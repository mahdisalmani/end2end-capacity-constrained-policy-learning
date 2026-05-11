"""
Re-evaluate the non-nested sweep with the ORACLE policy value
V_oracle(pi) = (1/N) sum_i Y_pot[i, pi(X_i)]
using the synthetic counterfactuals. This bypasses the noisy IPW estimator.

Retrains each cell (cheap — they're already cached as full per-cell CSVs,
but we need the per-individual arm vectors which weren't saved). Writes
augmented CSV with V_oracle_train and V_oracle_eval columns.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

from src.train import train_GF
from src.train_alt import train_alt
from src.s2_dual import (
    fit_outcome_models,
    get_mhat_matrix,
    solve_dual_lp,
)

from experiments.run_cell_nonnested import (
    NN_T, NN_D, NN_TAU, NN_B,
    _gen, load_or_build_eval_cache,
)
from experiments.n_sweep_criteo import (
    _f_arms_and_assigner,
    ipw_policy_value,
)
from experiments.real_queue_experiment import S2_METHODS


def _oracle_v(arms, Y_pot):
    return float(Y_pot[np.arange(len(arms)), arms].mean())


def _cell(args):
    N, seed, kwargs = args
    try:
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
        T = NN_T; D = NN_D; B = NN_B
        eval_data = load_or_build_eval_cache()
        td = _gen(N=N, seed=seed)
        torch.manual_seed(seed); np.random.seed(seed)

        out = []
        steps = kwargs["steps"]; lr = kwargs["lr"]
        f_tau = kwargs["f_tau"]; cap = kwargs["cap_buffer"]
        alt_freq = kwargs["alt_inner_freq"]

        # F
        model_F, _, _ = train_GF(
            kind="F", train_data=td, D=D, T=T, tau=f_tau, b=B,
            steps=steps, lr=lr, log_every=max(1, steps), seed=seed,
        )
        a_tr_F, a_ev_F, _ = _f_arms_and_assigner(model_F, td, eval_data, B, cap)
        out.append(("F", a_tr_F, a_ev_F))

        # Alt
        model_Alt, _, _ = train_alt(
            train_data=td, D=D, T=T, tau=f_tau, b=B,
            outer_steps=steps, inner_freq=alt_freq, lr=lr,
            log_every=max(1, steps), seed=seed,
        )
        a_tr_A, a_ev_A, _ = _f_arms_and_assigner(model_Alt, td, eval_data, B, cap)
        out.append(("Alt", a_tr_A, a_ev_A))

        # Gs
        model_Gs, _, _ = train_GF(
            kind="Gs", train_data=td, D=D, T=T, tau=NN_TAU, b=B,
            steps=steps, lr=lr, log_every=max(1, steps), seed=seed,
        )
        a_tr_G, a_ev_G, _ = _f_arms_and_assigner(model_Gs, td, eval_data, B, cap)
        out.append(("Gs", a_tr_G, a_ev_G))

        # S2 + S2-mlp
        for method in list(S2_METHODS) + ["mlp"]:
            models = fit_outcome_models(
                X_train=td["X"], T_train=td["T"], Y_train=td["Y"],
                T=T, method=method, E_train=td["E"],
            )
            M_tr = get_mhat_matrix(models, td["X"], T)
            mu_hat, _, _, _ = solve_dual_lp(M_tr, B, verbose=False)
            a_tr = (M_tr - mu_hat[None, :]).argmax(axis=1)
            M_ev = get_mhat_matrix(models, eval_data["X"], T)
            a_ev = (M_ev - mu_hat[None, :]).argmax(axis=1)
            out.append((f"S2-{method}", a_tr, a_ev))

        # Baselines on eval — random and treat_all
        rng_r = np.random.default_rng(seed * 7919 + 1)
        a_tr_r = rng_r.integers(T, size=len(td["T"]))
        a_ev_r = rng_r.integers(T, size=len(eval_data["T"]))
        out.append(("random", a_tr_r, a_ev_r))
        a_tr_ta = np.ones(len(td["T"]), dtype=np.int64)
        a_ev_ta = np.ones(len(eval_data["T"]), dtype=np.int64)
        out.append(("treat_all", a_tr_ta, a_ev_ta))

        rows = []
        for tag, a_tr, a_ev in out:
            rows.append({
                "method": tag,
                "N": int(N),
                "seed": int(seed),
                "V_oracle_train": _oracle_v(a_tr, td["Y_pot"]),
                "V_oracle_eval":  _oracle_v(a_ev, eval_data["Y_pot"]),
                "ipw_val":        ipw_policy_value(a_ev, eval_data),
                "ipw_train":      ipw_policy_value(a_tr, td),
            })
        return (N, seed, "ok", rows)
    except Exception:
        return (N, seed, traceback.format_exc(), [])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--n-values", type=int, nargs="+", default=[1000])
    p.add_argument("--out-csv", default="results/nonnested_oracle.csv")
    args = p.parse_args()

    load_or_build_eval_cache()
    work = [
        (N, s, dict(steps=args.steps, lr=5e-3, f_tau=0.03,
                    cap_buffer=0.92, alt_inner_freq=5))
        for N in args.n_values for s in range(args.seeds)
    ]
    print(f"[reeval] {len(work)} cells, {args.workers} workers", flush=True)
    rows = []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for i, (N, s, status, rs) in enumerate(pool.imap_unordered(_cell, work), 1):
            print(f"[{i:3d}/{len(work)}] N={N} seed={s} status={status[:20]}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
            rows.extend(rs)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[reeval] wrote {args.out_csv}  shape={df.shape}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

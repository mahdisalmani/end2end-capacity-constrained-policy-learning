"""
[LEGACY — superseded by the queue-sim cell/sweep pipeline; kept for reproducing the old results/sweep.csv artifacts]
NOTE: the *-mu variants re-solve mu on EVAL scores, which is an
eval-information peek — mu is part of the trained policy under this
project's bilevel-eval convention, and the project reverted the practice in
commit 3efaf40. They are therefore OFF by default; pass --with-eval-mu to
reproduce the old (leaky) numbers deliberately.

Single-cell worker for the (N, seed) sweep.

run_one_cell(N, seed) -> list[dict]:
    fresh train data for (N, seed) + the shared eval split, runs all 7 methods
    plus the random/oracle baselines, writes one per-cell CSV to
    results/cells/cell_N{N}_seed{seed}.csv. Resumable: if the CSV already
    exists, returns its rows without re-running.
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

from src import config
from src.data import generate_data
from src.inner_G import initialize_G_layer, solve_G_scipy
from src.inner_F import reset_F_state, _solve_F_inner
from src.train import train_GF
from src.evaluation import evaluate_GF_model, evaluate_greedy_no_cap_from_model
from src.baselines import evaluate_random_policy, evaluate_oracle_greedy_no_cap
from src.s2_dual import (
    run_dual_method,
    run_dual_method_with_eval_mu,
    fit_outcome_models,
    get_mhat_matrix,
)


DATA_DIR = "data"
CELL_CSV_DIR = "results/cells"
SHARED_EVAL_PATH = os.path.join(DATA_DIR, "eval.npz")
S2_METHODS = ["linear", "lasso", "tree", "knn", "dr"]


def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CELL_CSV_DIR, exist_ok=True)


def _cell_csv_path(N, seed):
    return os.path.join(CELL_CSV_DIR, f"cell_N{N}_seed{seed}.csv")


def _failed_path(N, seed):
    return os.path.join(CELL_CSV_DIR, f"cell_N{N}_seed{seed}.FAILED")


def _train_npz_path(N, seed):
    return os.path.join(DATA_DIR, f"train_N{N}_seed{seed}.npz")


def _atomic_write_csv(df, final_path):
    tmp = final_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, final_path)


def _atomic_savez(final_path, **kwargs):
    """Atomic np.savez_compressed. Per-PID tmp path so that concurrent workers
    writing the same final_path don't race on the tmp file. Tmp path ends in
    .npz so numpy doesn't auto-append the extension."""
    tmp = f"{final_path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **kwargs)
    os.replace(tmp, final_path)


def _load_or_make_eval():
    """Single shared eval split, generated once and cached on disk."""
    if os.path.exists(SHARED_EVAL_PATH):
        with np.load(SHARED_EVAL_PATH, allow_pickle=True) as f:
            return {k: f[k] for k in f.files}

    eval_data = generate_data(
        N=config.N_EVAL,
        seed=config.EVAL_SEED,
        d=config.D, T=config.T,
        sigma_y=config.SIGMA_Y,
        propensity_strength=config.PROPENSITY_STRENGTH,
        outcome_strength=config.OUTCOME_STRENGTH,
        treatment_effect_strength=config.TREATMENT_EFFECT_STRENGTH,
        clip_propensity=config.CLIP_PROPENSITY,
    )
    _atomic_savez(SHARED_EVAL_PATH, **eval_data)
    return eval_data


def _flatten_result(r, N, seed, T, wall_s):
    """Drop arrays into per-treatment columns + str-encode mu for archival."""
    alloc = np.asarray(r["alloc"])
    row = {
        "N": N,
        "seed": seed,
        "method": r["tag"],
        "V_IPW_train": r.get("V_IPW_train", np.nan),
        "V_IPW_eval": r.get("V_IPW_eval", np.nan),
        "V_DR_eval": r.get("V_DR_eval", np.nan),
        "V_oracle": r.get("V_oracle", np.nan),
        "cap_viol_sup": r.get("cap_viol_sup", np.nan),
        "cap_ok": r.get("cap_ok", False),
        "lp_status": r.get("lp_status", "NA"),
        "lp_time": r.get("lp_time", np.nan),
        "total_time": r.get("total_time", np.nan),
        "wall_s": wall_s,
    }
    for t in range(T):
        row[f"Alloc_{t}"] = alloc[t] if t < len(alloc) else np.nan

    mu = r.get("mu")
    if mu is None:
        row["mu_str"] = ""
    else:
        mu_arr = np.asarray(mu).ravel()
        row["mu_str"] = ",".join(f"{x:.6g}" for x in mu_arr)

    return row


def _run_cell_body(N, seed, steps, lr, skip_mu=True):
    """Inner body of run_one_cell — assumes setup already done."""
    T = config.T
    D = config.D
    TAU = config.TAU
    B = config.B

    # Per-cell setup of mutable module state.
    initialize_G_layer(N=N, T=T, tau=TAU, b=B)
    reset_F_state()

    # Train + eval data.
    train_data = generate_data(
        N=N, seed=seed,
        d=D, T=T,
        sigma_y=config.SIGMA_Y,
        propensity_strength=config.PROPENSITY_STRENGTH,
        outcome_strength=config.OUTCOME_STRENGTH,
        treatment_effect_strength=config.TREATMENT_EFFECT_STRENGTH,
        clip_propensity=config.CLIP_PROPENSITY,
    )
    train_path = _train_npz_path(N, seed)
    if not os.path.exists(train_path):
        _atomic_savez(train_path, **train_data)

    eval_data = _load_or_make_eval()

    # One DR-baseline outcome model (Lasso) for V_DR_eval across all methods.
    dr_models = fit_outcome_models(
        X_train=train_data["X"],
        T_train=train_data["T"],
        Y_train=train_data["Y"],
        T=T, method="lasso",
    )
    m_hat_eval = get_mhat_matrix(dr_models, eval_data["X"], T)

    results = []

    # Reference baselines.
    results.append(evaluate_random_policy(
        train_data, eval_data, m_hat_eval, b=B, T=T,
    ))
    results.append(evaluate_oracle_greedy_no_cap(
        train_data, eval_data, m_hat_eval, b=B, T=T,
    ))

    b_t = torch.tensor(B)
    X_eval_t = torch.tensor(eval_data["X"])

    # G: vanilla (μ from train) + G-mu (μ re-solved on eval).
    model_G, mu_G_train, _ = train_GF(
        kind="G", train_data=train_data,
        D=D, T=T, tau=TAU, b=B,
        steps=steps, lr=lr, log_every=max(1, steps),
        seed=seed,
    )
    results.append(evaluate_GF_model(
        model=model_G, mu_train=mu_G_train,
        train_data=train_data, eval_data=eval_data,
        m_hat_eval=m_hat_eval, b=B, tau=TAU, T=T, tag="G",
        policy="argmax",
    ))
    if not skip_mu:
        with torch.no_grad():
            M_eval_G = model_G(X_eval_t)
        mu_G_eval = solve_G_scipy(M_eval_G, b_t, TAU)
        results.append(evaluate_GF_model(
            model=model_G, mu_train=mu_G_eval,
            train_data=train_data, eval_data=eval_data,
            m_hat_eval=m_hat_eval, b=B, tau=TAU, T=T, tag="G-mu",
            policy="argmax",
        ))
    results.append(evaluate_greedy_no_cap_from_model(
        model=model_G, train_data=train_data, eval_data=eval_data,
        m_hat_eval=m_hat_eval, b=B, T=T,
    ))

    # F: vanilla + F-mu.
    model_F, mu_F_train, _ = train_GF(
        kind="F", train_data=train_data,
        D=D, T=T, tau=TAU, b=B,
        steps=steps, lr=lr, log_every=max(1, steps),
        seed=seed,
    )
    results.append(evaluate_GF_model(
        model=model_F, mu_train=mu_F_train,
        train_data=train_data, eval_data=eval_data,
        m_hat_eval=m_hat_eval, b=B, tau=TAU, T=T, tag="F",
    ))
    if not skip_mu:
        with torch.no_grad():
            M_eval_F = model_F(X_eval_t)
        mu_F_eval, _ = _solve_F_inner(M_eval_F, b_t, TAU)
        results.append(evaluate_GF_model(
            model=model_F, mu_train=mu_F_eval,
            train_data=train_data, eval_data=eval_data,
            m_hat_eval=m_hat_eval, b=B, tau=TAU, T=T, tag="F-mu",
        ))

    # S2 methods.
    for method in S2_METHODS:
        if skip_mu:
            results.append(run_dual_method(
                method_name=method,
                train_data=train_data,
                eval_data=eval_data,
                m_hat_eval=m_hat_eval,
                T=T, b=B, verbose_lp=False,
            ))
            continue
        pair = run_dual_method_with_eval_mu(
            method_name=method,
            train_data=train_data,
            eval_data=eval_data,
            m_hat_eval=m_hat_eval,
            T=T, b=B, verbose_lp=False,
        )
        results.extend(pair)

    return results


def run_one_cell(N, seed, steps=200, lr=5e-3, force=False, skip_mu=True):
    """Worker entry point. Returns list of row-dicts (also written to CSV)."""
    _ensure_dirs()

    # Worker-local determinism + thread settings (safe to call repeatedly).
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_path = _cell_csv_path(N, seed)

    if os.path.exists(out_path) and not force:
        df = pd.read_csv(out_path)
        return df.to_dict(orient="records")

    # Best-effort cleanup of stale FAILED marker.
    failed_path = _failed_path(N, seed)
    if os.path.exists(failed_path):
        os.remove(failed_path)

    t0 = time.time()
    try:
        results = _run_cell_body(N, seed, steps=steps, lr=lr, skip_mu=skip_mu)
    except Exception:
        with open(failed_path, "w") as f:
            f.write(traceback.format_exc())
        return []

    wall = time.time() - t0
    rows = [_flatten_result(r, N, seed, config.T, wall) for r in results]
    df = pd.DataFrame(rows)
    _atomic_write_csv(df, out_path)
    return rows


def _parse_args():
    p = argparse.ArgumentParser(description="Run one (N, seed) cell.")
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--force", action="store_true")
    p.add_argument("--with-eval-mu", dest="skip_mu", action="store_false",
                   help="Also emit the *-mu variants, which re-solve mu on the "
                        "EVAL split. This is an eval-information peek and the "
                        "project reverted it (commit 3efaf40); off by default.")
    return p.parse_args()


def main():
    args = _parse_args()
    rows = run_one_cell(N=args.N, seed=args.seed,
                        steps=args.steps, lr=args.lr, force=args.force,
                        skip_mu=args.skip_mu)
    if not rows:
        print(f"[run_cell] N={args.N} seed={args.seed}: cell FAILED, "
              f"see {_failed_path(args.N, args.seed)}", file=sys.stderr)
        sys.exit(1)
    df = pd.DataFrame(rows)
    print(df[["method", "V_IPW_train", "V_IPW_eval", "V_DR_eval",
              "V_oracle", "cap_viol_sup"]].to_string(index=False))
    print(f"\nN={args.N} seed={args.seed} wall={rows[0]['wall_s']:.1f}s")
    print(f"Wrote {_cell_csv_path(args.N, args.seed)}")


if __name__ == "__main__":
    main()

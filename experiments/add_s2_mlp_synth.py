"""
Add S2-mlp to the synthetic sweep results: train per-arm MLPs on each
(N, seed) cell of the synthetic grid, deploy via the cap-buffer LP, run
the queue simulator, and merge into `results/synth_sweep_seeds.csv`.

Run:
    python -m experiments.add_s2_mlp_synth --workers 20
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

from src import config
from src.s2_dual import (
    fit_outcome_models,
    get_mhat_matrix,
    solve_dual_lp,
)

from experiments.run_cell_synth import (
    load_or_build_eval_cache,
    _generate,
    _simulate_one,
)
from experiments.n_sweep_criteo import ipw_policy_value
from experiments.real_queue_experiment import make_s2_assigner


def _train_eval_mlp_cell(N, seed, kwargs):
    try:
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
        T = int(config.T); D = int(config.D); B = config.B

        eval_data = load_or_build_eval_cache()
        td = _generate(N=N, seed=seed, D=D, T=T)

        torch.manual_seed(seed)
        np.random.seed(seed)

        t0 = time.time()
        models = fit_outcome_models(
            X_train=td["X"], T_train=td["T"], Y_train=td["Y"],
            T=T, method="mlp", E_train=td["E"],
        )
        M_hat_train = get_mhat_matrix(models, td["X"], T)
        mu_hat, _, _, _ = solve_dual_lp(M_hat_train, B, verbose=False)
        a_train = (M_hat_train - mu_hat[None, :]).argmax(axis=1)
        M_hat_eval = get_mhat_matrix(models, eval_data["X"], T)
        a_eval = (M_hat_eval - mu_hat[None, :]).argmax(axis=1)
        ass = make_s2_assigner(models, mu_hat, eval_data, T)
        train_wall = time.time() - t0

        row = _simulate_one(
            ass, "S2-mlp", eval_data, T, B,
            kwargs["N_sim"], kwargs["lambda_people"], kwargs["max_time_mult"],
            seed,
        )
        row["method"] = "S2-mlp"
        row["ipw_train"] = ipw_policy_value(a_train, td)
        row["ipw_val"] = ipw_policy_value(a_eval, eval_data)
        row["N"] = int(N)
        row["seed"] = int(seed)
        return (N, seed, "ok", train_wall, row)
    except Exception:
        return (N, seed, f"EXC:{traceback.format_exc(limit=2)}", 0.0, None)


def _worker(args):
    N, seed, kwargs = args
    return _train_eval_mlp_cell(N, seed, kwargs)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-csv", default="results/synth_sweep_seeds.csv")
    p.add_argument("--out-csv", default="results/synth_sweep_seeds_mlp.csv")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--N-sim", type=int, default=1000, dest="N_sim")
    p.add_argument("--lambda-people", type=float, default=1.0)
    p.add_argument("--max-time-mult", type=float, default=1.5)
    return p.parse_args()


def main():
    args = _parse_args()
    base = pd.read_csv(args.in_csv)
    n_values = sorted(base["N"].unique().tolist())
    seeds = list(range(args.seeds))
    print(f"[add-mlp] base csv shape={base.shape}  N's={n_values}  seeds={len(seeds)}",
          flush=True)

    print("[add-mlp] preloading synthetic eval cache (one-shot)…", flush=True)
    load_or_build_eval_cache()

    kwargs = dict(
        N_sim=args.N_sim,
        lambda_people=args.lambda_people,
        max_time_mult=args.max_time_mult,
    )
    work = [(N, s, kwargs) for N in sorted(n_values, reverse=True) for s in seeds]
    total = len(work)
    print(f"[add-mlp] {total} cells, {args.workers} workers", flush=True)

    new_rows = []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for i, (N, seed, status, twall, row) in enumerate(
            pool.imap_unordered(_worker, work), 1
        ):
            elapsed = time.time() - t0
            print(f"[{i:4d}/{total}] N={N:5d} seed={seed:3d}  status={status}  "
                  f"train_wall={twall:6.1f}s  elapsed={elapsed:7.1f}s", flush=True)
            if row is not None:
                new_rows.append(row)

    print(f"[add-mlp] done in {time.time() - t0:.1f}s. rows added: {len(new_rows)}",
          flush=True)

    mlp_df = pd.DataFrame(new_rows)
    merged = pd.concat([base, mlp_df], ignore_index=True)
    merged = merged[base.columns.tolist() +
                    [c for c in merged.columns if c not in base.columns]]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"[add-mlp] wrote {args.out_csv}  shape={merged.shape}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

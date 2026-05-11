"""
Parallel (N, seed) sweep on Criteo Uplift.

Each cell = one (N, seed) pair → one process via multiprocessing → trains
random + treat_all + F + Alt + Gs + 5 S2 + queue sim + IPW. Cells are
resumable (per-cell CSV cache).

Run:
    python -m experiments.sweep_criteo --seeds 20
    python -m experiments.sweep_criteo --seeds 20 --workers 32

After all cells complete, aggregates into a single CSV and 1x3 plot
(IPW val, IPW train, mean wait) with per-method curves.
"""

import argparse
import datetime as dt
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.run_cell_criteo import (
    CELL_DIR,
    _cell_csv_path,
    _failed_path,
    load_or_build_criteo_cache,
    run_one_cell,
)
from experiments.real_queue_experiment import S2_METHODS
from experiments.n_sweep_criteo import plot_results


DEFAULT_PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _worker(args):
    (N, seed, kwargs) = args
    t0 = time.time()
    try:
        rows = run_one_cell(N=N, seed=seed, **kwargs)
        wall = time.time() - t0
        status = "ok" if rows else "FAILED"
        return (N, seed, status, wall)
    except Exception as e:  # noqa: BLE001
        return (N, seed, f"EXC:{type(e).__name__}:{e}", time.time() - t0)


def _resolve_ns(args, n_train_full):
    if args.n_values:
        return list(args.n_values)
    return [max(1, int(round(n_train_full * p / 100.0))) for p in args.n_pcts]


def _select_cells(N_values, seeds, force):
    cells = [(N, s) for N in N_values for s in seeds]
    if force:
        return sorted(cells, key=lambda ns: -ns[0])
    pending = [
        (N, s) for (N, s) in cells
        if not os.path.exists(_cell_csv_path(N, s))
    ]
    pending.sort(key=lambda ns: -ns[0])  # big-N first
    return pending


def _gather_results(N_values, seeds):
    paths = []
    for N in N_values:
        for s in seeds:
            p = _cell_csv_path(N, s)
            if os.path.exists(p):
                paths.append(p)
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _parse_args():
    p = argparse.ArgumentParser(description="Parallel Criteo (N, seed) sweep.")
    p.add_argument("--n-pcts", type=float, nargs="+", default=DEFAULT_PCTS)
    p.add_argument("--n-values", type=int, nargs="+", default=None,
                   help="Override --n-pcts with absolute N's.")
    p.add_argument("--seeds", type=int, default=20,
                   help="Number of seeds. Cells use seed=0..S-1.")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--f-tau", type=float, default=0.03)
    p.add_argument("--cap-buffer", type=float, default=0.92)
    p.add_argument("--alt-inner-freq", type=int, default=5)
    p.add_argument("--criteo-variant", type=str, default="10pct",
                   choices=["full", "10pct"])
    p.add_argument("--criteo-subsample", type=int, default=50_000)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--N-sim", type=int, default=1000, dest="N_sim")
    p.add_argument("--lambda-people", type=float, default=1.0)
    p.add_argument("--max-time-mult", type=float, default=1.5)
    p.add_argument("--out-csv", type=str,
                   default="results/criteo_sweep_seeds.csv")
    p.add_argument("--out-png", type=str,
                   default="results/criteo_sweep_seeds.png")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(CELL_DIR, exist_ok=True)

    # Build Criteo cache in the parent (one-shot heavy ops) before forking.
    print("[sweep] preloading Criteo cache (one-shot)…", flush=True)
    train_full, eval_data, cfg = load_or_build_criteo_cache(
        args.criteo_subsample, args.criteo_variant, args.split_seed,
    )
    n_train_full = len(train_full["T"])
    print(f"[sweep] criteo: train_full={n_train_full}, eval={len(eval_data['T'])}, "
          f"T={cfg['T']}, B={cfg['B']}", flush=True)

    N_values = _resolve_ns(args, n_train_full)
    seeds = list(range(args.seeds))
    cells = _select_cells(N_values, seeds, args.force)
    grid = len(N_values) * len(seeds)
    print(f"[sweep] grid {len(N_values)} N × {len(seeds)} seeds = {grid}", flush=True)
    print(f"[sweep] {len(cells)} cells to run, {args.workers} workers", flush=True)

    cell_kwargs = dict(
        subsample=args.criteo_subsample,
        variant=args.criteo_variant,
        split_seed=args.split_seed,
        steps=args.steps,
        lr=args.lr,
        f_tau=args.f_tau,
        cap_buffer=args.cap_buffer,
        alt_inner_freq=args.alt_inner_freq,
        N_sim=args.N_sim,
        lambda_people=args.lambda_people,
        max_time_mult=args.max_time_mult,
        force=args.force,
    )
    work = [(N, s, cell_kwargs) for (N, s) in cells]

    t_start = time.time()
    if work:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
            for i, (N, seed, status, wall) in enumerate(
                pool.imap_unordered(_worker, work), 1
            ):
                elapsed = time.time() - t_start
                print(f"[{i:4d}/{len(work)}] N={N:5d} seed={seed:3d}  "
                      f"status={status}  cell_wall={wall:6.1f}s  "
                      f"elapsed={elapsed:7.1f}s", flush=True)
    else:
        print("[sweep] all cells already cached.", flush=True)

    print(f"[sweep] done in {time.time() - t_start:.1f}s", flush=True)

    # Aggregate.
    df = _gather_results(N_values, seeds)
    if df.empty:
        print("[sweep] no results gathered — exiting.", flush=True)
        return
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[sweep] wrote {args.out_csv}  shape={df.shape}", flush=True)

    agg = df.groupby(["N", "method"]).agg(
        mean_wait_served_mean=("mean_wait_served", "mean"),
        ipw_val_mean=("ipw_val", "mean"),
        ipw_train_mean=("ipw_train", "mean"),
        frac_unserved_mean=("frac_unserved", "mean"),
    ).reset_index()

    method_order = ["random", "treat_all", "Gs", "F", "Alt"] + [
        f"S2-{m}" for m in S2_METHODS
    ]
    methods_present = [m for m in method_order if m in agg["method"].values]
    variant_label = {"full": "full ~14M rows",
                     "10pct": "10% sample, ~1.4M rows"}[args.criteo_variant]
    plot_results(agg, methods_present, args.out_png, variant_label=variant_label)

    # Print summary pivots.
    for col, label, fmt in [
        ("ipw_val_mean", "IPW(val) (mean over seeds)", "{:.4f}".format),
        ("ipw_train_mean", "IPW(train) (mean over seeds)", "{:.4f}".format),
        ("mean_wait_served_mean", "mean wait time (served)", "{:.2f}".format),
        ("frac_unserved_mean", "% unserved", lambda x: f"{100*x:.2f}"),
    ]:
        piv = agg.pivot(index="N", columns="method", values=col).sort_index()
        print("\n" + "=" * 100)
        print(f"Criteo sweep — {label}")
        print("=" * 100)
        with pd.option_context("display.width", 200):
            print(piv.map(fmt).to_string())


if __name__ == "__main__":
    sys.exit(main())

"""
Parallel (N, seed) sweep on LaLonde NSW + PSID-1.

Mirrors `experiments/sweep_criteo.py`. Each (N, seed) cell trains all
methods + queue sim; cells run via multiprocessing pool.

Run:
    python -m experiments.sweep_lalonde --seeds 20 --workers 20
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

from experiments.run_cell_lalonde import (
    CELL_DIR,
    _cell_csv_path,
    load_or_build_lalonde_cache,
    run_one_cell,
)
from experiments.real_queue_experiment import S2_METHODS


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
    pending = [(N, s) for (N, s) in cells if not os.path.exists(_cell_csv_path(N, s))]
    pending.sort(key=lambda ns: -ns[0])
    return pending


def _gather_results(N_values, seeds):
    paths = [_cell_csv_path(N, s) for N in N_values for s in seeds
             if os.path.exists(_cell_csv_path(N, s))]
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _parse_args():
    p = argparse.ArgumentParser(description="Parallel LaLonde (N, seed) sweep.")
    p.add_argument("--n-pcts", type=float, nargs="+", default=DEFAULT_PCTS)
    p.add_argument("--n-values", type=int, nargs="+", default=None)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--f-tau", type=float, default=0.03)
    p.add_argument("--cap-buffer", type=float, default=0.92)
    p.add_argument("--alt-inner-freq", type=int, default=5)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--N-sim", type=int, default=1000, dest="N_sim")
    p.add_argument("--lambda-people", type=float, default=1.0)
    p.add_argument("--max-time-mult", type=float, default=1.5)
    p.add_argument("--out-csv", type=str, default="results/lalonde_sweep_seeds.csv")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(CELL_DIR, exist_ok=True)

    print("[sweep] preloading LaLonde cache (one-shot)…", flush=True)
    train_full, eval_data, cfg = load_or_build_lalonde_cache(
        args.train_frac, args.split_seed,
    )
    n_train_full = len(train_full["T"])
    print(f"[sweep] lalonde: train_full={n_train_full}, eval={len(eval_data['T'])}, "
          f"T={cfg['T']}, B={cfg['B']}", flush=True)

    N_values = _resolve_ns(args, n_train_full)
    seeds = list(range(args.seeds))
    cells = _select_cells(N_values, seeds, args.force)
    grid = len(N_values) * len(seeds)
    print(f"[sweep] grid {len(N_values)} N × {len(seeds)} seeds = {grid}", flush=True)
    print(f"[sweep] {len(cells)} cells to run, {args.workers} workers", flush=True)

    cell_kwargs = dict(
        train_frac=args.train_frac,
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

    df = _gather_results(N_values, seeds)
    if df.empty:
        print("[sweep] no results gathered — exiting.", flush=True)
        return
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[sweep] wrote {args.out_csv}  shape={df.shape}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

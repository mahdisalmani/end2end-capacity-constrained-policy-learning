"""
Parallel (N, seed) sweep on the synthetic DGP from src.data + src.config.

Mirrors `sweep_criteo.py` / `sweep_lalonde.py`. Default N grid is
1000..10000 step 1000.

Run:
    python -m experiments.sweep_synth --seeds 20 --workers 20
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import pandas as pd

from experiments.run_cell_synth import (
    CELL_DIR,
    _cell_csv_path,
    load_or_build_eval_cache,
    run_one_cell,
)


DEFAULT_NS = list(range(1000, 10_001, 1000))


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
    p = argparse.ArgumentParser(description="Parallel synthetic-DGP (N, seed) sweep.")
    p.add_argument("--n-values", type=int, nargs="+", default=DEFAULT_NS)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--f-tau", type=float, default=0.03)
    p.add_argument("--cap-buffer", type=float, default=0.92)
    p.add_argument("--alt-inner-freq", type=int, default=5)
    p.add_argument("--N-sim", type=int, default=1000, dest="N_sim")
    p.add_argument("--lambda-people", type=float, default=1.0)
    p.add_argument("--max-time-mult", type=float, default=1.5)
    p.add_argument("--out-csv", type=str, default="results/synth_sweep_seeds.csv")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(CELL_DIR, exist_ok=True)

    print("[sweep] preloading synthetic eval cache (one-shot)…", flush=True)
    eval_data = load_or_build_eval_cache()
    print(f"[sweep] synth eval: N={len(eval_data['T'])}, T={eval_data['Y_pot'].shape[1]}",
          flush=True)

    N_values = list(args.n_values)
    seeds = list(range(args.seeds))
    cells = _select_cells(N_values, seeds, args.force)
    grid = len(N_values) * len(seeds)
    print(f"[sweep] grid {len(N_values)} N × {len(seeds)} seeds = {grid}", flush=True)
    print(f"[sweep] {len(cells)} cells to run, {args.workers} workers", flush=True)

    cell_kwargs = dict(
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

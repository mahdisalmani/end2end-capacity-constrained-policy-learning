"""
Generic parallel (N, seed) sweep driver.

Runs `run_one_cell` from a given cell module over an N x seed grid with a
spawn-context multiprocessing pool (`maxtasksperchild=1`, so the module-level
warm-start caches in `src.inner_F` / `src.inner_G` are process-fresh for
every cell), then gathers the per-cell CSVs into one dataframe.

The dataset-specific drivers (`sweep_synth`, `sweep_criteo`,
`sweep_lalonde`, `sweep_nonnested`) supply argparse defaults, preload their
data cache in the parent before forking, and optionally post-process
(aggregate / plot) the gathered dataframe.
"""

import importlib
import multiprocessing as mp
import os
import time

import pandas as pd

from experiments.cell_core import cell_csv_path


def _worker(args):
    """Pool worker: import the cell module by name (spawn-safe) and run one
    cell. Returns (N, seed, status, wall_seconds)."""
    cell_module, N, seed, kwargs = args
    t0 = time.time()
    try:
        mod = importlib.import_module(cell_module)
        rows = mod.run_one_cell(N=N, seed=seed, **kwargs)
        wall = time.time() - t0
        status = "ok" if rows else "FAILED"
        return (N, seed, status, wall)
    except Exception as e:  # noqa: BLE001
        return (N, seed, f"EXC:{type(e).__name__}:{e}", time.time() - t0)


def select_cells(cell_dir, N_values, seeds, force):
    """Pending (N, seed) pairs, big-N first (they take longest)."""
    cells = [(N, s) for N in N_values for s in seeds]
    if force:
        return sorted(cells, key=lambda ns: -ns[0])
    pending = [
        (N, s) for (N, s) in cells
        if not os.path.exists(cell_csv_path(cell_dir, N, s))
    ]
    pending.sort(key=lambda ns: -ns[0])
    return pending


def gather_results(cell_dir, N_values, seeds):
    paths = [
        cell_csv_path(cell_dir, N, s)
        for N in N_values for s in seeds
        if os.path.exists(cell_csv_path(cell_dir, N, s))
    ]
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def run_sweep(cell_module, cell_dir, N_values, seeds, cell_kwargs,
              workers, force, out_csv):
    """Run the grid, gather, and (if out_csv) write the combined CSV.

    Returns the gathered dataframe (possibly empty).
    """
    os.makedirs(cell_dir, exist_ok=True)

    cells = select_cells(cell_dir, N_values, seeds, force)
    grid = len(N_values) * len(seeds)
    print(f"[sweep] grid {len(N_values)} N × {len(seeds)} seeds = {grid}", flush=True)
    print(f"[sweep] {len(cells)} cells to run, {workers} workers", flush=True)

    work = [(cell_module, N, s, cell_kwargs) for (N, s) in cells]

    t_start = time.time()
    if work:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
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

    df = gather_results(cell_dir, N_values, seeds)
    if df.empty:
        print("[sweep] no results gathered — exiting.", flush=True)
        return df

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[sweep] wrote {out_csv}  shape={df.shape}", flush=True)

    return df


def add_common_args(p, steps_default=500, out_csv_default=None):
    """Argparse options shared by every sweep driver."""
    p.add_argument("--seeds", type=int, default=20,
                   help="Number of seeds. Cells use seed=0..S-1.")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--steps", type=int, default=steps_default)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--f-tau", type=float, default=0.03)
    p.add_argument("--cap-buffer", type=float, default=0.92)
    p.add_argument("--alt-inner-freq", type=int, default=5)
    p.add_argument("--N-sim", type=int, default=1000, dest="N_sim")
    p.add_argument("--lambda-people", type=float, default=1.0)
    p.add_argument("--max-time-mult", type=float, default=1.5)
    p.add_argument("--out-csv", type=str, default=out_csv_default)
    p.add_argument("--force", action="store_true")
    return p


def common_cell_kwargs(args):
    """The run_one_cell kwargs shared by every dataset."""
    return dict(
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


def resolve_ns_from_pcts(n_values, n_pcts, n_train_full):
    """Absolute N grid: explicit --n-values wins over --n-pcts."""
    if n_values:
        return list(n_values)
    return [max(1, int(round(n_train_full * p / 100.0))) for p in n_pcts]

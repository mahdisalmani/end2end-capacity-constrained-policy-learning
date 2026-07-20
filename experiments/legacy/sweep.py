"""
[LEGACY — superseded by the queue-sim cell/sweep pipeline; kept for reproducing the old results/sweep.csv artifacts]

Parallel sweep over (N, seed) cells.

CLI:
    python -m experiments.sweep --workers 32
    python -m experiments.sweep --N 100 200 --seeds 0 1 --workers 4
    python -m experiments.sweep --force

Skips cells whose per-cell CSV already exists unless --force. Schedules
large-N cells first so the long tail is short. maxtasksperchild=1 to free
CVXPYLayer / diffcp memory between cells.
"""

import argparse
import datetime as dt
import multiprocessing as mp
import os
import sys
import time

from experiments.legacy import aggregate
from experiments.legacy.run_cell import (
    _cell_csv_path,
    _failed_path,
    run_one_cell,
)


DEFAULT_NS = list(range(10, 51, 10))
DEFAULT_SEEDS = list(range(20))
LOGS_DIR = "logs"


def _is_done(N, seed):
    return os.path.exists(_cell_csv_path(N, seed))


def _is_failed(N, seed):
    return os.path.exists(_failed_path(N, seed))


def _select_cells(Ns, seeds, force, retry_failed):
    cells = [(N, s) for N in Ns for s in seeds]
    if force:
        return sorted(cells, key=lambda ns: -ns[0])
    pending = []
    for N, s in cells:
        if _is_done(N, s):
            continue
        if _is_failed(N, s) and not retry_failed:
            continue
        pending.append((N, s))
    pending.sort(key=lambda ns: -ns[0])  # large-N first
    return pending


def _worker(args):
    """multiprocessing-compatible adapter (no kwargs through Pool)."""
    N, seed, steps, lr, force, skip_mu = args
    t0 = time.time()
    try:
        rows = run_one_cell(N=N, seed=seed, steps=steps, lr=lr,
                            force=force, skip_mu=skip_mu)
        wall = time.time() - t0
        status = "ok" if rows else "FAILED"
        return (N, seed, status, wall)
    except Exception as e:  # noqa: BLE001
        return (N, seed, f"EXC:{type(e).__name__}:{e}", time.time() - t0)


def _parse_args():
    p = argparse.ArgumentParser(description="Sweep (N, seed) grid.")
    p.add_argument("--N", type=int, nargs="+", default=DEFAULT_NS,
                   help="Training sizes to sweep (default: 10..50 step 10)")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="Seeds (default: 0..19)")
    p.add_argument("--workers", type=int, default=32,
                   help="Worker processes (default 32; 384 CPUs available)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--force", action="store_true",
                   help="Re-run cells whose per-cell CSV already exists.")
    p.add_argument("--retry-failed", action="store_true",
                   help="Re-run cells with a .FAILED marker.")
    p.add_argument("--no-aggregate", action="store_true",
                   help="Skip the post-sweep aggregate step.")
    p.add_argument("--with-eval-mu", dest="skip_mu", action="store_false",
                   default=True,
                   help="Also emit the *-mu variants, which re-solve mu on the "
                        "EVAL split (an eval-information peek). Off by default.")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(
        LOGS_DIR,
        f"sweep_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log",
    )

    cells = _select_cells(args.N, args.seeds, args.force, args.retry_failed)
    total_grid = len(args.N) * len(args.seeds)
    print(f"[sweep] grid {len(args.N)}xN * {len(args.seeds)} seeds = {total_grid}")
    print(f"[sweep] {len(cells)} cells to run, {args.workers} workers")
    print(f"[sweep] log: {log_path}")

    if not cells:
        print("[sweep] nothing to do.")
        if not args.no_aggregate:
            aggregate.main()
        return

    work = [(N, s, args.steps, args.lr, args.force, args.skip_mu) for (N, s) in cells]

    t_start = time.time()
    n_done = 0
    with open(log_path, "w") as logf:
        logf.write(f"sweep start: {len(cells)} cells, "
                   f"{args.workers} workers, steps={args.steps}, lr={args.lr}\n")
        logf.flush()
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
            for (N, seed, status, wall) in pool.imap_unordered(_worker, work):
                n_done += 1
                elapsed = time.time() - t_start
                line = (
                    f"[{n_done:4d}/{len(cells)}] N={N:4d} seed={seed:3d}  "
                    f"status={status}  cell_wall={wall:7.1f}s  "
                    f"sweep_elapsed={elapsed:7.1f}s"
                )
                print(line, flush=True)
                logf.write(line + "\n")
                logf.flush()

    print(f"[sweep] done in {time.time() - t_start:.1f}s")

    if not args.no_aggregate:
        aggregate.main()


if __name__ == "__main__":
    sys.exit(main())

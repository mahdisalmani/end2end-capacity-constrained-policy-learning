"""
Parallel (N, seed) sweep on the non-nested synthetic DGP
(`experiments.data_nonnested`, config hardcoded in `run_cell_nonnested`).

Default grid: N=1000, seed=0..19. Writes `results/nonnested_sweep_seeds.csv`.

Run:
    python -m experiments.sweep_nonnested --seeds 20 --workers 20
"""

import argparse
import sys

from experiments import sweep_core
from experiments.run_cell_nonnested import CELL_DIR, load_or_build_eval_cache


def _parse_args():
    p = argparse.ArgumentParser(description="Parallel non-nested-DGP (N, seed) sweep.")
    p.add_argument("--n-values", type=int, nargs="+", default=[1000])
    sweep_core.add_common_args(p, steps_default=1500,
                               out_csv_default="results/nonnested_sweep_seeds.csv")
    return p.parse_args()


def main():
    args = _parse_args()

    print("[sweep] preloading non-nested eval cache (one-shot)…", flush=True)
    eval_data = load_or_build_eval_cache()
    print(f"[sweep] nonnested eval: N={len(eval_data['T'])}, T={eval_data['Y_pot'].shape[1]}",
          flush=True)

    sweep_core.run_sweep(
        cell_module="experiments.run_cell_nonnested",
        cell_dir=CELL_DIR,
        N_values=list(args.n_values),
        seeds=list(range(args.seeds)),
        cell_kwargs=sweep_core.common_cell_kwargs(args),
        workers=args.workers,
        force=args.force,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    sys.exit(main())

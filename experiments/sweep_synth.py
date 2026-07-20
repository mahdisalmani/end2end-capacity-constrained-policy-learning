"""
Parallel (N, seed) sweep on the nested synthetic DGP from src.data + src.config.

Mirrors `sweep_criteo.py` / `sweep_lalonde.py`. Default N grid is
1000..10000 step 1000. Writes `results/synth_sweep_seeds.csv`.

Run:
    python -m experiments.sweep_synth --seeds 20 --workers 20
"""

import argparse
import sys

from experiments import sweep_core
from experiments.run_cell_synth import CELL_DIR, load_or_build_eval_cache


DEFAULT_NS = list(range(1000, 10_001, 1000))


def _parse_args():
    p = argparse.ArgumentParser(description="Parallel synthetic-DGP (N, seed) sweep.")
    p.add_argument("--n-values", type=int, nargs="+", default=DEFAULT_NS)
    sweep_core.add_common_args(p, steps_default=500,
                               out_csv_default="results/synth_sweep_seeds.csv")
    return p.parse_args()


def main():
    args = _parse_args()

    print("[sweep] preloading synthetic eval cache (one-shot)…", flush=True)
    eval_data = load_or_build_eval_cache()
    print(f"[sweep] synth eval: N={len(eval_data['T'])}, T={eval_data['Y_pot'].shape[1]}",
          flush=True)

    sweep_core.run_sweep(
        cell_module="experiments.run_cell_synth",
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

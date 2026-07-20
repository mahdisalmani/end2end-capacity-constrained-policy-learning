"""
Parallel (N, seed) sweep on LaLonde NSW + PSID-1.

Mirrors `experiments/sweep_criteo.py`. Each (N, seed) cell trains all
methods + queue sim; cells run via multiprocessing pool. Writes
`results/lalonde_sweep_seeds.csv`.

Run:
    python -m experiments.sweep_lalonde --seeds 20 --workers 20
"""

import argparse
import sys

from experiments import sweep_core
from experiments.run_cell_lalonde import CELL_DIR, load_or_build_lalonde_cache


DEFAULT_PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _parse_args():
    p = argparse.ArgumentParser(description="Parallel LaLonde (N, seed) sweep.")
    p.add_argument("--n-pcts", type=float, nargs="+", default=DEFAULT_PCTS)
    p.add_argument("--n-values", type=int, nargs="+", default=None)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--split-seed", type=int, default=0)
    sweep_core.add_common_args(p, steps_default=500,
                               out_csv_default="results/lalonde_sweep_seeds.csv")
    return p.parse_args()


def main():
    args = _parse_args()

    print("[sweep] preloading LaLonde cache (one-shot)…", flush=True)
    train_full, eval_data, cfg = load_or_build_lalonde_cache(
        args.train_frac, args.split_seed,
    )
    n_train_full = len(train_full["T"])
    print(f"[sweep] lalonde: train_full={n_train_full}, eval={len(eval_data['T'])}, "
          f"T={cfg['T']}, B={cfg['B']}", flush=True)

    cell_kwargs = dict(
        train_frac=args.train_frac,
        split_seed=args.split_seed,
        **sweep_core.common_cell_kwargs(args),
    )

    sweep_core.run_sweep(
        cell_module="experiments.run_cell_lalonde",
        cell_dir=CELL_DIR,
        N_values=sweep_core.resolve_ns_from_pcts(args.n_values, args.n_pcts,
                                                 n_train_full),
        seeds=list(range(args.seeds)),
        cell_kwargs=cell_kwargs,
        workers=args.workers,
        force=args.force,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    sys.exit(main())

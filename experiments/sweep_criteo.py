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
import sys

import pandas as pd

from experiments import sweep_core
from experiments.common import S2_METHODS
from experiments.n_sweep_criteo import plot_results
from experiments.run_cell_criteo import CELL_DIR, load_or_build_criteo_cache


DEFAULT_PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _parse_args():
    p = argparse.ArgumentParser(description="Parallel Criteo (N, seed) sweep.")
    p.add_argument("--n-pcts", type=float, nargs="+", default=DEFAULT_PCTS)
    p.add_argument("--n-values", type=int, nargs="+", default=None,
                   help="Override --n-pcts with absolute N's.")
    p.add_argument("--criteo-variant", type=str, default="10pct",
                   choices=["full", "10pct"])
    p.add_argument("--criteo-subsample", type=int, default=50_000)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--out-png", type=str,
                   default="results/criteo_sweep_seeds.png")
    sweep_core.add_common_args(p, steps_default=500,
                               out_csv_default="results/criteo_sweep_seeds.csv")
    return p.parse_args()


def main():
    args = _parse_args()

    # Build Criteo cache in the parent (one-shot heavy ops) before forking.
    print("[sweep] preloading Criteo cache (one-shot)…", flush=True)
    train_full, eval_data, cfg = load_or_build_criteo_cache(
        args.criteo_subsample, args.criteo_variant, args.split_seed,
    )
    n_train_full = len(train_full["T"])
    print(f"[sweep] criteo: train_full={n_train_full}, eval={len(eval_data['T'])}, "
          f"T={cfg['T']}, B={cfg['B']}", flush=True)

    cell_kwargs = dict(
        subsample=args.criteo_subsample,
        variant=args.criteo_variant,
        split_seed=args.split_seed,
        **sweep_core.common_cell_kwargs(args),
    )

    df = sweep_core.run_sweep(
        cell_module="experiments.run_cell_criteo",
        cell_dir=CELL_DIR,
        N_values=sweep_core.resolve_ns_from_pcts(args.n_values, args.n_pcts,
                                                 n_train_full),
        seeds=list(range(args.seeds)),
        cell_kwargs=cell_kwargs,
        workers=args.workers,
        force=args.force,
        out_csv=args.out_csv,
    )
    if df.empty:
        return

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

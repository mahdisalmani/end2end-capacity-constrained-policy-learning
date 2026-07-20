"""
[LEGACY — superseded by the queue-sim cell/sweep pipeline; kept for reproducing the old results/sweep.csv artifacts]
NOTE: Alt is now a first-class method in every run_cell_* cell; this
one-off backfill script predates that.

Add the Alt (alternating block-coordinate) method to the existing Criteo
sweep CSV/plot without re-running F + S2.

Loads `results/criteo_pct_sweep.csv` (F + 5 S2 + random + treat_all rows at
10 N's), trains Alt at each N with the same seeds/subsamples the original
sweep used, runs the same queue simulator, appends the new rows, rewrites
the merged CSV, and regenerates the plot.

Reuses the deployment recipe (cap-buffered LP) and queue simulator from
`n_sweep_criteo.py`.
"""

import os
import time

import numpy as np
import pandas as pd
import torch

from src.train_alt import train_alt
from experiments.data_criteo import load_criteo
from experiments.n_sweep_criteo import (
    _f_arms_and_assigner,
    _subsample,
    ipw_policy_value,
    plot_results,
)
from experiments.real_queue_experiment import (
    make_streams,
    simulate,
    aggregate_one,
    S2_METHODS,
)


def main(
    in_csv="results/criteo_sweep.csv",
    out_csv="results/criteo_sweep_alt.csv",
    out_png="results/criteo_sweep_alt.png",
    criteo_variant="10pct",
    criteo_subsample=50000,
    split_seed=0,
    train_seed=1,
    steps=500,
    lr=5e-3,
    f_tau=0.03,
    cap_buffer=0.92,
    alt_inner_freq=5,
    n_pcts=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    num_sim_seeds=20,
    N_sim=1000,
    lambda_people=1.0,
    max_time_mult=1.5,
):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)

    train_full, eval_data, cfg = load_criteo(
        seed=split_seed, subsample=criteo_subsample, variant=criteo_variant,
    )
    T = int(cfg["T"])
    D = int(cfg["D"])
    B = cfg["B"]

    n_train_full = len(train_full["T"])
    n_values = [max(1, int(round(n_train_full * p / 100.0))) for p in n_pcts]
    print(f"[add-alt] n_pcts={list(n_pcts)} -> n_values={n_values}  "
          f"(train_full={n_train_full})")

    rows = []
    for N in n_values:
        print(f"\n========== Alt @ N = {N} ==========", flush=True)
        td = _subsample(train_full, N, seed=split_seed * 1000 + N)
        t0 = time.time()

        model_Alt, mu_Alt, _ = train_alt(
            train_data=td, D=D, T=T, tau=f_tau, b=B,
            outer_steps=steps, inner_freq=alt_inner_freq, lr=lr,
            log_every=max(1, steps), seed=train_seed,
        )
        a_train, a_eval, assigner = _f_arms_and_assigner(
            model_Alt, td, eval_data, B, cap_buffer,
        )
        ipw_train = ipw_policy_value(a_train, td)
        ipw_val = ipw_policy_value(a_eval, eval_data)
        train_wall = time.time() - t0
        print(f"[Alt N={N}] train_wall={train_wall:.1f}s  "
              f"IPW(train)={ipw_train:.4f}  IPW(val)={ipw_val:.4f}", flush=True)

        for sim_seed in range(num_sim_seeds):
            people_t, person_idx, T_max, resource_t = make_streams(
                eval_data, N_sim, lambda_people, B,
                max_time_mult, seed=sim_seed * 7 + 13,
            )
            t_sim = time.time()
            recs = simulate(
                people_t, person_idx, resource_t, assigner,
                T=T, T_max=T_max, eval_data=eval_data, sim_seed=sim_seed,
            )
            wall = time.time() - t_sim
            row = aggregate_one(recs, "Alt", sim_seed, B, N_sim, wall)
            row["N"] = int(N)
            row["ipw_train"] = ipw_train
            row["ipw_val"] = ipw_val
            rows.append(row)

    alt_df = pd.DataFrame(rows)
    print(f"\n[add-alt] Alt rows: {len(alt_df)}")

    base_df = pd.read_csv(in_csv)
    merged = pd.concat([base_df, alt_df], ignore_index=True)
    # Align column order to base_df.
    merged = merged[base_df.columns.tolist() +
                    [c for c in merged.columns if c not in base_df.columns]]

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[add-alt] wrote {out_csv}  ({merged.shape})")

    # Aggregate + plot using n_sweep_criteo.plot_results.
    agg = merged.groupby(["N", "method"]).agg(
        mean_wait_served_mean=("mean_wait_served", "mean"),
        ipw_val_mean=("ipw_val", "mean"),
        ipw_train_mean=("ipw_train", "mean"),
        frac_unserved_mean=("frac_unserved", "mean"),
    ).reset_index()

    method_order = ["random", "treat_all", "F", "Alt"] + [f"S2-{m}" for m in S2_METHODS]
    methods_present = [m for m in method_order if m in agg["method"].values]
    variant_label = {"full": "full ~14M rows",
                     "10pct": "10% sample, ~1.4M rows"}[criteo_variant]
    plot_results(agg, methods_present, out_png, variant_label=variant_label)

    # Print summary pivots for the new (Alt-included) results.
    print("\n=== merged IPW(val) ===")
    print(agg.pivot(index="N", columns="method",
                    values="ipw_val_mean").round(4).to_string())
    print("\n=== merged IPW(train) ===")
    print(agg.pivot(index="N", columns="method",
                    values="ipw_train_mean").round(4).to_string())
    print("\n=== merged mean wait (served) ===")
    print(agg.pivot(index="N", columns="method",
                    values="mean_wait_served_mean").round(2).to_string())
    print("\n=== merged % unserved ===")
    print((100 * agg.pivot(index="N", columns="method",
                           values="frac_unserved_mean")).round(2).to_string())


if __name__ == "__main__":
    main()

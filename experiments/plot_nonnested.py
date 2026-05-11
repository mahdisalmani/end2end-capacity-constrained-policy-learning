"""
Bar-chart summary of the non-nested DGP sweep (single N=1000, 20 seeds).

Reads:
    results/nonnested_oracle.csv         (V_oracle_eval per (method, seed))
    results/nonnested_sweep_seeds.csv    (wait, frac_unserved per (method, seed))

Plots 4 panels: V_oracle, IPW(val), mean_wait_served, %unserved.
Bars are sorted by V_oracle (best to worst).
Error bars are 95% CI from 20 seeds (1.96 * SEM).
Methods are color-coded: bilevel (F/Gs/Alt) blue family,
S2 family orange, baselines (random/treat_all) grey.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_GROUP = {
    "F":         "bilevel",
    "Gs":        "bilevel",
    "Alt":       "bilevel",
    "S2-linear": "s2",
    "S2-lasso":  "s2",
    "S2-tree":   "s2",
    "S2-knn":    "s2",
    "S2-dr":     "s2",
    "S2-mlp":    "s2",
    "random":    "baseline",
    "treat_all": "baseline",
}

GROUP_COLOR = {
    "bilevel":  "#0b5394",
    "s2":       "#cc6600",
    "baseline": "#888888",
}


def _agg(df_o, df_q, methods):
    o = df_o.groupby("method").agg(
        v_mean=("V_oracle_eval", "mean"),
        v_sem=("V_oracle_eval", lambda x: x.std() / np.sqrt(len(x))),
        ipw_mean=("ipw_val", "mean"),
        ipw_sem=("ipw_val", lambda x: x.std() / np.sqrt(len(x))),
    )
    q = df_q.groupby("method").agg(
        wait_mean=("mean_wait_served", "mean"),
        wait_sem=("mean_wait_served", lambda x: x.std() / np.sqrt(len(x))),
        unserved_mean=("frac_unserved", lambda x: 100 * x.mean()),
        unserved_sem=("frac_unserved", lambda x: 100 * x.std() / np.sqrt(len(x))),
    )
    out = o.join(q, how="left")
    return out.reindex(methods)


def _panel(ax, agg, mean_col, sem_col, methods, title, ylabel,
           log_y=False):
    colors = [GROUP_COLOR[METHOD_GROUP[m]] for m in methods]
    means = agg.loc[methods, mean_col].values
    sems = agg.loc[methods, sem_col].values
    err = 1.96 * sems
    xs = np.arange(len(methods))
    ax.bar(xs, means, yerr=err, color=colors, capsize=3,
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(True, axis="y", alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle-csv", default="results/nonnested_oracle.csv")
    p.add_argument("--queue-csv",  default="results/nonnested_sweep_seeds.csv")
    p.add_argument("--out-png",    default="results/nonnested_bars.png")
    p.add_argument("--title", default=(
        "Non-nested DGP v2 (T=10, D=30, arm-1 Gaussian peak)  —  "
        "N=1000, 20 seeds, 95% CI"))
    args = p.parse_args()

    df_o = pd.read_csv(args.oracle_csv)
    df_q = pd.read_csv(args.queue_csv)

    # Sort by V_oracle descending (best -> worst), ignoring treat_all/random
    # for ordering since they're cap-aware-irrelevant.
    ranking = (df_o.groupby("method")["V_oracle_eval"].mean()
               .sort_values(ascending=False))
    methods = [m for m in ranking.index if m in METHOD_GROUP]

    agg = _agg(df_o, df_q, methods)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    _panel(axes[0, 0], agg, "v_mean", "v_sem", methods,
           "Oracle policy value V_oracle (high = better)",
           "V_oracle (eval)")
    _panel(axes[0, 1], agg, "ipw_mean", "ipw_sem", methods,
           "IPW value (val) (high = better, but noisy under confounding)",
           "V_IPW (eval)")
    _panel(axes[1, 0], agg, "wait_mean", "wait_sem", methods,
           "Mean wait time for served individuals (low = better)",
           "wait (sim units)", log_y=True)
    _panel(axes[1, 1], agg, "unserved_mean", "unserved_sem", methods,
           "% unserved (low = better; caps respected)",
           "% unserved")

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR["bilevel"],
                      label="Bilevel (F / Gs / Alt)"),
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR["s2"],
                      label="S2 (per-arm regression)"),
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR["baseline"],
                      label="Baseline (random / treat_all)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.98))

    fig.suptitle(args.title, y=0.92)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.savefig(args.out_png, dpi=140)
    print(f"[plot] wrote {args.out_png}")


if __name__ == "__main__":
    main()

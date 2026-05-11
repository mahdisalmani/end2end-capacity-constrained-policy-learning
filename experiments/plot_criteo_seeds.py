"""
Re-plot `results/criteo_sweep_seeds.csv` with:
  - methods restricted to Gs, F, Alt, S2-linear, S2-tree, S2-knn
  - 95% confidence-interval bands (mean ± 1.96 * SEM over 20 seeds)
  - title without "G omitted"
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["Gs", "F", "Alt", "S2-linear", "S2-tree", "S2-knn", "S2-mlp"]
METHOD_STYLE = {
    "Gs":        dict(color="#0b5394", lw=2.4, marker="o", ms=7,
                      label="Gs (proposed, convex dual)"),
    "F":         dict(color="#cc0000", lw=2.0, marker="s", ms=6, label="F"),
    "Alt":       dict(color="#674ea7", lw=2.0, marker="^", ms=6, label="Alt"),
    "S2-linear": dict(color="#999999", lw=1.4, marker="x", ms=6),
    "S2-tree":   dict(color="#6aa84f", lw=1.4, marker="v", ms=6),
    "S2-knn":    dict(color="#e69138", lw=1.4, marker="D", ms=5),
    "S2-mlp":    dict(color="#9900cc", lw=1.4, marker="*", ms=7),
}


def _summarise(df):
    """Per (N, method): mean and 1.96-SEM for IPW val / IPW train / wait."""
    g = df.groupby(["N", "method"])
    out = g.agg(
        ipw_val_mean=("ipw_val", "mean"),
        ipw_val_sem=("ipw_val", lambda x: x.std() / np.sqrt(len(x))),
        ipw_train_mean=("ipw_train", "mean"),
        ipw_train_sem=("ipw_train", lambda x: x.std() / np.sqrt(len(x))),
        wait_mean=("mean_wait_served", "mean"),
        wait_sem=("mean_wait_served", lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()
    return out


def _draw_panel(ax, agg, mean_col, sem_col, methods, log_y=False):
    for m in methods:
        sub = agg[agg["method"] == m].sort_values("N")
        if sub.empty:
            continue
        style = METHOD_STYLE.get(m, {"label": m})
        kw = {k: v for k, v in style.items() if k != "label"}
        label = style.get("label", m)
        lo = sub[mean_col] - 1.96 * sub[sem_col]
        hi = sub[mean_col] + 1.96 * sub[sem_col]
        if log_y:
            lo = lo.clip(lower=1e-3)
        ax.fill_between(sub["N"], lo, hi, alpha=0.18, color=kw.get("color"))
        ax.plot(sub["N"], sub[mean_col], label=label, **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-csv", default="results/criteo_sweep_seeds.csv")
    p.add_argument("--out-png", default="results/criteo_sweep_seeds_ci.png")
    p.add_argument("--variant", default="10% sample, ~1.4M rows")
    p.add_argument("--dataset-name", default="Criteo Uplift",
                   help="Title prefix, e.g. 'LaLonde NSW + PSID-1'.")
    p.add_argument("--y-label-val",
                   default="IPW value, validation (P(visit))",
                   help="Y-axis label for the IPW(val) panel.")
    p.add_argument("--y-label-train",
                   default="IPW value, train (P(visit))",
                   help="Y-axis label for the IPW(train) panel.")
    args = p.parse_args()

    df = pd.read_csv(args.in_csv)
    df = df[df["method"].isin(METHODS)].copy()
    if df.empty:
        raise SystemExit(f"No rows for methods {METHODS} in {args.in_csv}")
    agg = _summarise(df)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    ax = axes[0]
    _draw_panel(ax, agg, "ipw_val_mean", "ipw_val_sem", METHODS)
    ax.set_xlabel("N (train subsample size)")
    ax.set_ylabel(args.y_label_val)
    ax.set_xscale("log")
    ax.set_title("IPW (val) vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    ax = axes[1]
    _draw_panel(ax, agg, "ipw_train_mean", "ipw_train_sem", METHODS)
    ax.set_xlabel("N (train subsample size)")
    ax.set_ylabel(args.y_label_train)
    ax.set_xscale("log")
    ax.set_title("IPW (train) vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    ax = axes[2]
    _draw_panel(ax, agg, "wait_mean", "wait_sem", METHODS, log_y=True)
    ax.set_xlabel("N (train subsample size)")
    ax.set_ylabel("Mean wait time (served)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Mean wait time vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle(
        f"{args.dataset_name} ({args.variant}) — trained-policy comparison "
        f"(20 seeds, 95% CI)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.savefig(args.out_png, dpi=140)
    print(f"[plot] wrote {args.out_png}")


if __name__ == "__main__":
    main()

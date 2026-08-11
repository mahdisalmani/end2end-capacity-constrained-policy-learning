"""Figure for the ported Dual-Guided Learning matching benchmark.

Three panels versus training size N: deployed ground-truth value, deployed
share of the scarcest arm against its cap, and median wait. Reads
results/dglmatch_cells/*.csv (written by experiments/run_cell_dglmatch.py).

The scarce arm here is location 1: their generator gives it the highest mean
outcome and their capacities give it the smallest budget (0.2), so it is the
arm every method wants and only a feasible method can hold.
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_style import METHOD_STYLE, apply as _apply_style  # noqa: E402
_apply_style()

CAP_SCARCE = 0.2          # their capacity for location 1
SCARCE_COL = "alloc_1"

ORDER = ["F", "Gs", "Alt", "S2-mlp", "S2-tree", "S2-dr",
         "S2-knn", "S2-linear", "S2-lasso"]
LABEL = {"F": "F (end-to-end)", "Gs": "G (end-to-end, convex)",
         "Alt": "Alt (dual refresh)", "S2-mlp": "PtO-mlp (capacity-matched)",
         "S2-tree": "PtO-tree", "S2-dr": "PtO-dr", "S2-knn": "PtO-knn",
         "S2-linear": "PtO-linear", "S2-lasso": "PtO-lasso"}


def load():
    frames = []
    for f in sorted(glob.glob("results/dglmatch_cells/cell_N*_seed*.csv")):
        frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit("no cells found in results/dglmatch_cells/")
    df = pd.concat(frames, ignore_index=True)
    return df[df.method.isin(ORDER)]


def panel(ax, df, col, ylabel, title, clip_positive=False):
    for m in ORDER:
        d = df[df.method == m]
        if d.empty:
            continue
        g = d.groupby("N")[col].agg(["mean", "sem"])
        st = dict(METHOD_STYLE.get(m, {}))
        for k in ("marker", "alpha", "label"):
            st.pop(k, None)
        ax.plot(g.index, g["mean"], label=LABEL[m], **st)
        lo = g["mean"] - 1.96 * g["sem"].fillna(0)
        hi = g["mean"] + 1.96 * g["sem"].fillna(0)
        if clip_positive:
            lo = lo.clip(lower=max(1e-2, float(g["mean"].min()) * 0.25))
        ax.fill_between(g.index, lo, hi, alpha=0.12,
                        color=st.get("color"), lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("training size $N$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)


def main():
    df = load()
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.35))

    panel(axes[0], df, "mean_oracle_outcome_all",
          "deployed ground-truth value", "(a)  value")

    panel(axes[1], df, SCARCE_COL,
          "deployed share, scarcest arm", "(b)  is it feasible?")
    axes[1].axhline(CAP_SCARCE, color="#a03a31", lw=1.0)
    axes[1].annotate(f"cap {CAP_SCARCE:g}",
                     xy=(df.N.min(), CAP_SCARCE), xytext=(2, 4),
                     textcoords="offset points", fontsize=7.5, color="#a03a31")

    panel(axes[2], df, "mean_wait_served",
          "mean wait (served)", "(c)  queueing cost", clip_positive=True)
    axes[2].set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.20), columnspacing=1.1, handlelength=1.8)
    fig.tight_layout(w_pad=2.0)
    os.makedirs("figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig_dglmatch.{ext}", bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    print("[fig] wrote figures/fig_dglmatch.pdf/.png")


if __name__ == "__main__":
    main()

"""Paper figure: the Deployment-Adjusted Policy Value index.

(a) Combined index across all five datasets as a function of the single
    parameter kappa (cost of one waiting period, in fractions of the
    value-at-stake). Every reader applies their own delay-aversion; the
    crossover is marked, not chosen.
(b) Per-dataset crossover kappa* — the delay cost at which the best
    end-to-end method overtakes the best two-stage. Datasets where it never
    crosses in [0, 0.2] are shown at the top edge as "never".

Reads results/deploy_index.json (produced by scripts/deploy_index.py).
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

STYLE = {
    "F":         dict(color="#2a78d6", lw=2.0, label="F (end-to-end, non-convex)"),
    "Gs":        dict(color="#008300", lw=2.0, label="G (end-to-end, convex)"),
    "Alt":       dict(color="#e87ba4", lw=1.4, label="Alt (dual refresh)"),
    "S2-linear": dict(color="#eda100", lw=1.1, label="S2-linear"),
    "S2-lasso":  dict(color="#1baf7a", lw=1.1, label="S2-lasso"),
    "S2-tree":   dict(color="#eb6834", lw=1.6, label="S2-tree"),
    "S2-knn":    dict(color="#4a3aa7", lw=1.1, label="S2-knn"),
    "S2-dr":     dict(color="#c8321e", lw=1.1, label="S2-dr"),
}
DS_LABEL = {"nonnested": "non-\nnested", "adultsemi": "Adult\nsemi",
            "actg": "ACTG\n175", "diabetes": "Diabetes\n130",
            "criteo": "Criteo", "lalonde": "LaLonde"}


def main():
    d = json.load(open("results/deploy_index.json"))
    K = np.array(d["kappas"])
    KMAX = 0.05
    sel = K <= KMAX

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9),
                             gridspec_kw={"width_ratios": [1.9, 1.0]})

    ax = axes[0]
    for m, st in STYLE.items():
        if m not in d["combined"]:
            continue
        v = np.array(d["combined"][m])
        ax.plot(K[sel], v[sel], **st, zorder=5 if m in ("F", "Gs") else 3)
    lo = -1.6
    ax.set_ylim(lo, 1.05)
    deep = min(min(d["combined"][m]) for m in ("S2-lasso", "S2-tree")
               if m in d["combined"])
    ax.annotate(f"S2-lasso / S2-tree continue to {deep:+.1f}",
                xy=(0.0415, lo + 0.06), ha="right", fontsize=6.6,
                color="#8a6a5a")
    cross = d["combined_crossover"]
    if cross == 0.0:
        ax.annotate("end-to-end leads at every κ,\nincluding κ = 0 (delay-free)",
                    xy=(0.0055, 0.99), fontsize=7, va="top",
                    color="#444444")
    elif cross is not None:
        ax.axvline(cross, color="#666666", lw=0.9, ls=(0, (4, 3)))
        ax.annotate(f"end-to-end leads\nfor all κ ≥ {cross:.4f}",
                    xy=(cross, 1.02), xytext=(6, -2),
                    textcoords="offset points", fontsize=7, va="top",
                    color="#444444")
    ax.set_xlabel("κ — cost of one waiting period (fraction of value-at-stake)")
    ax.set_ylabel("combined deployability index")
    n_ds = len(d["datasets"])
    ax.set_title(f"(a)  DAPV(κ) averaged over all {n_ds} datasets",
                 loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=6.3, frameon=False, ncol=1, loc="lower right",
              bbox_to_anchor=(1.0, 0.02))

    ax = axes[1]
    ds_order = ["diabetes", "nonnested", "adultsemi", "actg", "criteo",
                "lalonde"]
    ds_order = [ds for ds in ds_order if ds in d["datasets"]]
    ys, labels = [], []
    NEVER_Y = 0.09
    for i, ds in enumerate(ds_order):
        c = d["datasets"][ds]["crossover"]
        labels.append(DS_LABEL[ds])
        if c is None:
            ax.scatter([i], [NEVER_Y], marker="^", s=42, color="#a03a31", zorder=5)
            ax.annotate("never", xy=(i, NEVER_Y), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=6.6, color="#a03a31")
            ys.append(np.nan)
        else:
            ax.plot([i, i], [0, c], color="#2a78d6", lw=1.6, zorder=3)
            ax.scatter([i], [c], s=42, color="#2a78d6", zorder=5)
            lbl = "0" if c == 0.0 else f"{c:.4f}"
            ax.annotate(lbl, xy=(i, c), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=6.6,
                        color="#1c4b7a")
            ys.append(c)
    ax.set_xticks(range(len(ds_order)), labels, fontsize=6.2)
    ax.set_xlim(-0.55, len(ds_order) - 0.45)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylim(0, 0.105)
    ax.set_ylabel("crossover κ*")
    ax.set_title("(b)  delay cost at which end-to-end overtakes",
                 loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Deployment-Adjusted Policy Value: value, feasibility and delay "
                 "in one number — with its only parameter on the axis",
                 y=1.04, fontsize=9.3, fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    os.makedirs("figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig_index.{ext}", dpi=300 if ext == "png" else None)
    print("[fig] wrote figures/fig_index.pdf/.png")


if __name__ == "__main__":
    main()

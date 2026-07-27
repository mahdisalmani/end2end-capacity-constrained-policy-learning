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

from paper_style import FIG_WIDE_2, METHOD_STYLE, apply as _apply_style
_apply_style()

_KEEP = ("F", "Gs", "Alt", "S2-linear", "S2-lasso", "S2-tree", "S2-knn",
         "S2-dr", "S2-mlp")
STYLE = {}
for _k in _KEEP:
    _st = dict(METHOD_STYLE[_k])
    _st.pop("marker", None)          # index panels are pure line charts
    STYLE[_k] = _st

DS_LABEL = {"nonnested": "non-\nnested", "adultsemi": "Adult\nsemi",
            "actg": "ACTG\n175", "diabetes": "Diabetes\n130",
            "criteo": "Criteo", "lalonde": "\nLaLonde"}


def main():
    d = json.load(open("results/deploy_index.json"))
    K = np.array(d["kappas"])
    KMAX = 0.05
    sel = K <= KMAX

    fig, axes = plt.subplots(1, 2, figsize=FIG_WIDE_2,
                             gridspec_kw={"width_ratios": [1.9, 1.0]})

    ax = axes[0]
    for m, st in STYLE.items():
        if m not in d["combined"]:
            continue
        v = np.array(d["combined"][m])
        ax.plot(K[sel], v[sel], **st)
    lo = -1.6
    ax.set_ylim(lo, 1.05)
    deep = min(min(d["combined"][m]) for m in ("S2-lasso", "S2-tree")
               if m in d["combined"])
    ax.annotate(f"PtO-lasso / PtO-tree reach {deep:+.0f}",
                xy=(0.0415, lo + 0.17), ha="right", fontsize=7.5,
                color="#8a6a5a")
    cross = d["combined_crossover"]
    if cross == 0.0:
        ax.annotate("end-to-end leads at every κ, including κ = 0 (delay-free)",
                    xy=(0.5, 1.02), xycoords="axes fraction", ha="center",
                    va="bottom", fontsize=8, color="#444444")
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
                 loc="left", fontweight="bold", pad=16)
    ax.grid(True, axis="y", alpha=0.3)
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.24), columnspacing=1.1,
               handlelength=1.8)

    ax = axes[1]
    ds_order = ["diabetes", "nonnested", "adultsemi", "actg", "criteo",
                "lalonde"]
    ds_order = [ds for ds in ds_order if ds in d["datasets"]]
    ys, labels, ctrl_ix = [], [], []
    NEVER_Y = 0.09
    for i, ds in enumerate(ds_order):
        c = d["datasets"][ds]["crossover"]
        labels.append(DS_LABEL[ds])
        if c is None:
            ax.scatter([i], [NEVER_Y], marker="o", s=42, color="#8a8a8a",
                       zorder=5)
            ctrl_ix.append(i)
            ys.append(np.nan)
        else:
            ax.plot([i, i], [0, c], color="#2a78d6", lw=1.6, zorder=3)
            ax.scatter([i], [c], s=42, color="#2a78d6", zorder=5)
            lbl = "0" if c == 0.0 else f"{c:.4f}"
            ax.annotate(lbl, xy=(i, c), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color="#1c4b7a")
            ys.append(c)
    if ctrl_ix:
        _lbl = "no gap (control)" if len(ctrl_ix) == 1 else "no gap (controls)"
        ax.annotate(_lbl, xy=(float(np.mean(ctrl_ix)), NEVER_Y),
                    xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=7.5, color="#6a6a6a")
    ax.set_xticks(range(len(ds_order)), labels, fontsize=7.5)
    ax.set_xlim(-0.55, len(ds_order) - 0.45)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylim(0, 0.105)
    ax.set_ylabel("crossover κ*")
    ax.set_title("(b)  delay cost at which end-to-end overtakes",
                 loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # No suptitle: the caption carries it in the paper.
    fig.tight_layout(w_pad=2.0)
    os.makedirs("figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig_index.{ext}", dpi=300 if ext == "png" else None)
    print("[fig] wrote figures/fig_index.pdf/.png")


if __name__ == "__main__":
    main()

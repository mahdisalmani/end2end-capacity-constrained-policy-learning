"""Paper figure: the mechanism DGP (dense decision margin, tight capacity).

(a) Deployed ground-truth value vs N, with the cap-feasible oracle.
(b) Deployed share of the capped arm vs N, against the cap and the
    buffered-LP line: two-stage thresholds noisy estimates at a price and
    overshoots; end-to-end concentrates under the cap.
(c) Median wait vs N (log scale): the queue converts the overshoot into
    10-100x waits.

Reads results/mechanism_sweep_seeds.csv (gather_cells.py mechanism).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_style import FIG_WIDE_3, METHOD_STYLE, apply as _apply_style
_apply_style()

_KEEP = ("F", "Gs", "Alt", "S2-linear", "S2-tree", "S2-knn", "S2-mlp")
STYLE = {}
for _k in _KEEP:
    _st = dict(METHOD_STYLE[_k])
    _st.pop("marker", None)          # line panels; the frontier scatter sets its own markers
    _st.pop("alpha", None)
    STYLE[_k] = _st
STYLE["F"]["label"] = "F (end-to-end)"

CAP, BUF = 0.25, 0.92


def curves(df, col, agg="mean"):
    g = df.groupby(["method", "N"])[col]
    m = (g.median() if agg == "median" else g.mean()).unstack()
    return m


def main():
    df = pd.read_csv("results/mechanism_sweep_seeds.csv")
    Ns = sorted(df.N.unique())
    Nd = Ns[-1]

    from experiments.data_mechanism import generate_mechanism, oracle_value
    oracle = float(np.mean([oracle_value(generate_mechanism(1000, s)[1])
                            for s in range(10)]))

    fig, axes = plt.subplots(1, 3, figsize=FIG_WIDE_3)

    # (a) the deployment frontier at N = Nd: value vs wait, feasibility as ring
    ax = axes[0]
    d = df[df.N == Nd]
    ag = d.groupby("method").agg(V=("mean_oracle_outcome_all", "mean"),
                                 W=("mean_wait_all", "median"),
                                 al=("alloc_1", "mean"))
    for m, st in STYLE.items():
        if m not in ag.index:
            continue
        w = max(float(ag.loc[m, "W"]), 0.05)
        over = ag.loc[m, "al"] > CAP
        ax.scatter([w], [ag.loc[m, "V"]], s=52, color=st["color"], zorder=5,
                   marker="s" if over else "o",
                   edgecolor="#a03a31" if over else "none", linewidth=1.4)
        dy = {"Gs": -9, "Alt": 5, "F": 5, "S2-linear": -9,
              "S2-knn": 5, "S2-tree": -9}.get(m, 5)
        dx, ha = ((-5, "right") if m == "S2-linear" else (5, "left"))
        ax.annotate(st["label"].split(" (")[0], xy=(w, ag.loc[m, "V"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                    ha=ha, color=st["color"])
    ax.axhline(oracle, color="#555555", lw=0.9, ls=(0, (2, 2)))
    ax.annotate("cap-feasible oracle", xy=(0.05, oracle), xytext=(2, -8),
                textcoords="offset points", fontsize=7.5, color="#555555")
    ax.set_xscale("log")
    ax.set_xlabel("median wait (periods, log)")
    ax.set_ylabel("deployed ground-truth value")
    ax.set_title("(a)  the deployment frontier",
                 loc="left", fontweight="bold")
    ax.annotate(f"$N$ = {Nd//1000}k;  squares deploy\nOVER the cap",
                xy=(0.97, 0.52), xycoords="axes fraction", ha="right",
                fontsize=7.5, color="#a03a31")
    ax.grid(True, alpha=0.3)

    # (b) deployed share of the capped arm vs N
    ax = axes[1]
    A = df.groupby(["method", "N"])["alloc_1"].mean().unstack()
    for m, st in STYLE.items():
        if m in A.index:
            ax.plot(Ns, A.loc[m, Ns], **st)
    ax.axhline(CAP, color="#a03a31", lw=1.0)
    ax.annotate("cap 0.25", xy=(Ns[0], CAP), xytext=(2, 4), ha="left",
                textcoords="offset points", fontsize=7.5, color="#a03a31")
    ax.axhline(CAP * BUF, color="#a03a31", lw=0.7, ls=(0, (3, 3)))
    ax.annotate("buffered LP 0.23", xy=(Ns[0], CAP * BUF), xytext=(2, -9),
                ha="left", textcoords="offset points", fontsize=7.5,
                color="#a03a31")
    ax.set_xscale("log")
    ax.set_xticks(Ns, [f"{n/1000:.3g}k" for n in Ns])
    ax.set_xlabel("training samples $N$")
    ax.set_ylabel("deployed share, capped arm")
    ax.set_title("(b)  who respects the cap", loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # (c) deployed value vs N
    ax = axes[2]
    V = df.groupby(["method", "N"])["mean_oracle_outcome_all"].mean().unstack()
    for m, st in STYLE.items():
        if m in V.index:
            ax.plot(Ns, V.loc[m, Ns], **st)
    ax.axhline(oracle, color="#555555", lw=0.9, ls=(0, (2, 2)))
    ax.axhline(0.0, color="#aaaaaa", lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks(Ns, [f"{n/1000:.3g}k" for n in Ns])
    ax.set_xlabel("training samples $N$")
    ax.set_ylabel("deployed ground-truth value")
    ax.set_title("(c)  value as data grows", loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.14), columnspacing=1.1,
               handlelength=1.8)

    # No suptitle: the caption carries it in the paper.
    fig.tight_layout(w_pad=1.6)
    os.makedirs("figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig_mechanism.{ext}",
                    dpi=300 if ext == "png" else None)
    print("[fig] wrote figures/fig_mechanism.pdf/.png")


if __name__ == "__main__":
    main()

"""Supplementary figure: the buffer-sweep ablation.

Can a tighter deployment buffer substitute for decision-aware training?
Four panels: deployed share of the capped arm and deployed value versus
the buffer, on the mechanism dataset (ground-truth value) and ACTG 175
(held-out IPW value). Reads results/buffer_cells/*.csv.
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_style import METHOD_STYLE, apply as _apply_style  # noqa: E402
_apply_style()

STYLE = {
    "F":        dict(METHOD_STYLE["F"], label="F (end-to-end)"),
    "G":        dict(METHOD_STYLE["Gs"], label="G (end-to-end, convex)"),
    "PtO-mlp":  dict(METHOD_STYLE["S2-mlp"]),
    "PtO-tree": dict(METHOD_STYLE["S2-tree"]),
}
for _st in STYLE.values():
    _st.pop("marker", None)
    _st.pop("alpha", None)

CAP = {"mechanism": 0.25, "actg": 0.30}
VCOL = {"mechanism": "mean_oracle_outcome_all", "actg": "ipw_val"}
VLAB = {"mechanism": "deployed ground-truth value",
        "actg": "held-out IPW value"}
DSLAB = {"mechanism": "mechanism", "actg": "ACTG 175"}


def main():
    frames = []
    for f in glob.glob("results/buffer_cells/*.csv"):
        ds = os.path.basename(f).split("_N")[0]
        d = pd.read_csv(f)
        d["dataset"] = ds
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(1, 4, figsize=(8.2, 2.1))
    for j, ds in enumerate(("mechanism", "actg")):
        d = df[df.dataset == ds]
        ax = axes[2 * j]
        A = d.groupby(["method", "buffer"])["alloc_1"].mean().unstack()
        for m, st in STYLE.items():
            if m in A.index:
                ax.plot(A.columns, A.loc[m], **st)
        ax.axhline(CAP[ds], color="#a03a31", lw=1.0)
        ax.annotate(f"cap {CAP[ds]:g}", xy=(A.columns[0], CAP[ds]),
                    xytext=(2, 4), textcoords="offset points", ha="left",
                    fontsize=7.5, color="#a03a31")
        ax.set_xlabel("deployment buffer")
        ax.set_ylabel("deployed share, capped arm")
        ax.set_title(f"({'ac'[j]})  {DSLAB[ds]}: share",
                     loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[2 * j + 1]
        V = d.groupby(["method", "buffer"])[VCOL[ds]].mean().unstack()
        for m, st in STYLE.items():
            if m in V.index:
                ax.plot(V.columns, V.loc[m], **st)
        ax.set_xlabel("deployment buffer")
        ax.set_ylabel(VLAB[ds])
        ax.set_title(f"({'bd'[j]})  {DSLAB[ds]}: value",
                     loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.16), columnspacing=1.1,
               handlelength=1.8)
    fig.tight_layout(w_pad=2.2)
    os.makedirs("figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig_buffer.{ext}",
                    dpi=300 if ext == "png" else None)
    print("[fig] wrote figures/fig_buffer.pdf/.png")


if __name__ == "__main__":
    main()

"""Publication figures: one semi-synthetic, one real-data.

Each figure is a 1x3 row over training-set size N:

  (a) IPW policy value on the validation split  — what the policy is worth
  (b) deployed mass on the capped arm vs its capacity — whether it is legal
  (c) mean wait time in the queue simulation      — what it costs operationally

Panel (b) is the one that earns the figure: on both datasets the value panel
alone would rank an infeasible policy first, and (b) shows why that ranking is
wrong. Baselines that ignore capacity (treat-all) are drawn as thin grey
reference lines rather than as competing series.

Outputs vector PDF (for LaTeX) plus a 300-dpi PNG (for slides/preview).

Usage:
    python scripts/make_paper_figures.py
    python scripts/make_paper_figures.py --real lalonde --outdir figures
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- house style ----------------------------------------------------------
# One sans face throughout, hairline spines, no chartjunk. Sizes are chosen
# for a ~3.4in-wide column reproduction (single column) without further
# scaling, so text stays >= 7pt on the printed page.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.5,
    "lines.markersize": 3.6,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,      # TrueType: editable/searchable text in the PDF
    "ps.fonttype": 42,
})

# Ours emphasised; baselines recede. Colours are the CVD-checked categorical
# slots used throughout the project's reporting.
STYLE = {
    "F":         dict(color="#2a78d6", lw=1.9, marker="o", zorder=6, label="F (end-to-end, non-convex)"),
    "Gs":        dict(color="#008300", lw=1.9, marker="s", zorder=6, label="G (end-to-end, convex)"),
    "Alt":       dict(color="#e87ba4", lw=1.6, marker="^", zorder=5, label="Alt (dual refresh)"),
    "S2-linear": dict(color="#eda100", lw=1.1, marker="v", zorder=3, alpha=0.9, label="S2-linear"),
    "S2-tree":   dict(color="#eb6834", lw=1.1, marker="D", zorder=3, alpha=0.9, label="S2-tree"),
    "S2-knn":    dict(color="#4a3aa7", lw=1.1, marker="P", zorder=3, alpha=0.9, label="S2-knn"),
    "S2-mlp":    dict(color="#e34948", lw=1.1, marker="X", zorder=3, alpha=0.9, label="S2-mlp"),
}
ORDER = ["F", "Gs", "Alt", "S2-linear", "S2-tree", "S2-knn", "S2-mlp"]
REFS = {"treat_all": "treat-all", "random": "random"}

CAP1 = {"criteo": 0.50, "lalonde": 0.30, "nonnested": 0.10}
VALUE_LABEL = {
    "criteo":    "IPW policy value  (P(visit))",
    "lalonde":   "IPW policy value  (\\$1k)",
    "nonnested": "IPW policy value",
}
TITLE = {
    "criteo":    "Criteo Uplift (real; 2 arms, capacity 0.50)",
    "lalonde":   "LaLonde NSW + PSID-1 (real; 2 arms, capacity 0.30)",
    "nonnested": "Non-nested synthetic DGP (10 arms, capacity 0.10 each)",
}


def summarize(csv, metric, skewed=False):
    """Seed-level means first, then across seeds. Returns {method: (N, mid, lo, hi)}.

    Value/allocation use mean +- 1.96 SEM. Wait time is heavily right-skewed
    across seeds (a policy either clears its queue or it does not), so it uses
    median + interquartile range, which is robust and cannot go negative.
    """
    df = pd.read_csv(csv)
    per_seed = df.groupby(["N", "seed", "method"], as_index=False)[metric].mean()
    out = {}
    for m, g in per_seed.groupby("method"):
        Ns, mid, lo, hi = [], [], [], []
        for N, gg in g.groupby("N"):
            v = gg[metric].dropna().values
            if not len(v):
                continue
            Ns.append(N)
            if skewed:
                mid.append(np.median(v))
                lo.append(np.percentile(v, 25))
                hi.append(np.percentile(v, 75))
            else:
                mu = v.mean()
                sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                mid.append(mu); lo.append(mu - 1.96 * sem); hi.append(mu + 1.96 * sem)
        if Ns:
            o = np.argsort(Ns)
            out[m] = tuple(np.asarray(a)[o] for a in (Ns, mid, lo, hi))
    return out


# Uncertainty bands are drawn for the proposed methods only. With seven
# series every band overlapping every other one produces mud and hides the
# lines it is meant to qualify; restricting them keeps the figure legible and
# puts the uncertainty where the claim is. The full per-method intervals are
# in the data tables.
BAND_FOR = ("F", "Gs")


def draw(ax, data, methods, band=True, logy=False, xticks=None):
    for m in methods:
        if m not in data:
            continue
        N, mid, lo, hi = data[m]
        st = dict(STYLE[m])
        lbl = st.pop("label")
        if band and m in BAND_FOR:
            ax.fill_between(N, lo, hi, color=st["color"], alpha=0.15, lw=0, zorder=2)
        ax.plot(N, mid, label=lbl, markeredgecolor="white", markeredgewidth=0.5, **st)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t//1000}k" if t >= 1000 else str(t) for t in xticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(True, which="major", axis="y")
    ax.set_axisbelow(True)


def add_reference(ax, data, key, label, logy=False, fmt="{}"):
    """Infeasible / trivial baselines as a flat annotated line, not a series."""
    if key not in data:
        return None
    N, mid, _, _ = data[key]
    ax.plot(N, mid, color="#8a8a8a", lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.annotate(label, xy=(N[-1], mid[-1]), xytext=(-2, 3),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=6.8, color="#6a6a6a")
    return mid[-1]


def make_figure(dataset, csv, outdir, panel_c="wait"):
    val = summarize(csv, "ipw_val")
    alloc = summarize(csv, "alloc_1")
    wait = summarize(csv, "mean_wait_served", skewed=True)
    methods = [m for m in ORDER if m in val]

    xticks = sorted({int(n) for v in val.values() for n in v[0]})

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.85))

    # (a) policy value
    ax = axes[0]
    draw(ax, val, methods, xticks=xticks)
    add_reference(ax, val, "treat_all", "treat-all")
    add_reference(ax, val, "random", "random")
    ax.set_xlabel("training size $N$")
    ax.set_ylabel(VALUE_LABEL[dataset])
    ax.set_title("(a)  policy value on held-out data", loc="left", fontweight="bold")

    # (b) allocation vs capacity — the panel that decides feasibility
    ax = axes[1]
    draw(ax, alloc, methods, xticks=xticks)
    cap = CAP1[dataset]
    ax.axhline(cap, color="#c0392b", lw=1.1, ls=(0, (5, 3)), zorder=4)
    # Shade the infeasible region and label the REGION rather than the line:
    # "over capacity" is a state of the curve, and a region label cannot
    # collide with whichever series happens to run near the threshold.
    ax.axhspan(cap, 1.02, color="#c0392b", alpha=0.05, lw=0, zorder=0)
    ax.annotate(f"infeasible  (> $b={cap:g}$)", xy=(0.035, 0.955),
                xycoords="axes fraction", ha="left", va="top",
                fontsize=7, color="#c0392b", fontweight="bold", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ax.set_xlabel("training size $N$")
    ax.set_ylabel("deployed mass on capped arm")
    ax.set_title("(b)  is the policy feasible?", loc="left", fontweight="bold")
    # Scale to the drawn series only. treat-all sits at 1.0 on this panel and
    # is not plotted here, so including it would flatten every real curve
    # against the axis (badly so when the cap is 0.1).
    drawn_hi = [np.max(alloc[m][3]) for m in methods if m in alloc and len(alloc[m][3])]
    top = max(cap * 1.35, (max(drawn_hi) * 1.10) if drawn_hi else cap * 1.35)
    ax.set_ylim(0, min(top, 1.02))

    # (c) operational cost
    ax = axes[2]
    draw(ax, wait, methods, logy=True, xticks=xticks)
    add_reference(ax, wait, "treat_all", "treat-all", logy=True)
    ax.set_xlabel("training size $N$")
    ax.set_ylabel("mean wait time (served)")
    ax.set_title("(c)  queueing cost of deployment", loc="left", fontweight="bold")

    # one legend for the row, below the panels
    handles = [Line2D([], [], color=STYLE[m]["color"], lw=STYLE[m]["lw"],
                      marker=STYLE[m]["marker"], markersize=3.6,
                      markeredgecolor="white", markeredgewidth=0.5,
                      label=STYLE[m]["label"]) for m in methods]
    handles.append(Line2D([], [], color="#8a8a8a", lw=0.9, ls=(0, (4, 2)),
                          label="uncapped / trivial baselines"))
    handles.append(Patch(facecolor="#2a78d6", alpha=0.15, edgecolor="none",
                         label="95% CI (proposed methods; IQR in (c))"))
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)),
               frameon=False, bbox_to_anchor=(0.5, -0.13), columnspacing=1.4,
               handlelength=2.0)
    fig.suptitle(TITLE[dataset], y=1.03, fontsize=9.5, fontweight="bold")
    fig.tight_layout(w_pad=1.6)

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"fig_{dataset}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"[fig] wrote {stem}.pdf and {stem}.png  ({len(methods)} methods)")
    return stem


def main():
    p = argparse.ArgumentParser(description="Build the two paper figures.")
    p.add_argument("--synthetic", default="nonnested",
                   choices=["nonnested", "synth"])
    p.add_argument("--real", default="criteo", choices=["criteo", "lalonde"])
    p.add_argument("--outdir", default="figures")
    args = p.parse_args()

    for tag, ds in (("synthetic", args.synthetic), ("real", args.real)):
        csv = f"results/{ds}_sweep_seeds.csv"
        if not os.path.exists(csv):
            print(f"[fig] SKIP {tag} ({ds}): {csv} not found")
            continue
        make_figure(ds, csv, args.outdir)


if __name__ == "__main__":
    main()

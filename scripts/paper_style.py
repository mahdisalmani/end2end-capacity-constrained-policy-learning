"""Single source of truth for the paper's figure typography and palette.

Design constants:

- The paper is set in Times (AAAI newtx). Figures use matplotlib's STIX
  faces, which are metric-compatible with Times, so figure text reads as
  one system with the body text. Math inside figures uses the same face.
- Figures are designed at their FINAL printed width (7.0 in for a
  two-column-spanning figure*), so a point in the script is a point on
  the page: base 9 pt matches the AAAI caption size, ticks 8 pt.
- One palette for every figure, the CVD-checked categorical slots used
  across the project's reporting. Emphasis by weight, not hue: proposed
  methods are heavier, baselines lighter.

Usage:
    from paper_style import apply, METHOD_STYLE, FIG_WIDE_3, FIG_WIDE_2
    apply()
"""

import matplotlib

FIG_WIDE_3 = (7.4, 2.2)     # three-panel figure*
FIG_WIDE_2 = (7.0, 2.3)    # two-panel figure* (index)
FIG_COL = (3.35, 2.5)      # single-column figure

METHOD_STYLE = {
    "F":         dict(color="#2a78d6", lw=1.9, marker="o", zorder=6,
                      label="F (end-to-end, non-convex)"),
    "Gs":        dict(color="#008300", lw=1.9, marker="s", zorder=6,
                      label="G (end-to-end, convex)"),
    "Alt":       dict(color="#e87ba4", lw=1.6, marker="^", zorder=5,
                      label="Alt (dual refresh)"),
    "S2-linear": dict(color="#eda100", lw=1.1, marker="v", zorder=3,
                      alpha=0.9, label="PtO-linear"),
    "S2-lasso":  dict(color="#1baf7a", lw=1.1, marker="<", zorder=3,
                      alpha=0.9, label="PtO-lasso"),
    "S2-tree":   dict(color="#eb6834", lw=1.1, marker="D", zorder=3,
                      alpha=0.9, label="PtO-tree"),
    "S2-knn":    dict(color="#4a3aa7", lw=1.1, marker="P", zorder=3,
                      alpha=0.9, label="PtO-knn"),
    "S2-dr":     dict(color="#e34948", lw=1.1, marker="*", zorder=3,
                      alpha=0.9, label="PtO-dr"),
    "S2-mlp":    dict(color="#c8321e", lw=1.4, marker="X", zorder=4,
                      ls=(0, (5, 2)), label="PtO-mlp (capacity-matched)"),
}


def apply():
    matplotlib.rcParams.update({
        # one type system with the paper body (Times-compatible STIX)
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # sizes at final printed scale (figure* = 7.0 in wide)
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        # geometry
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
        "grid.alpha": 0.3,
        # output
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

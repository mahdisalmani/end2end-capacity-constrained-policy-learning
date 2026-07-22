"""Paper figure: the regime maps.

(a) capacity tightness x effect nonlinearity, (b) outcome noise x
nonlinearity — each cell the mean over seeds of Delta = oracle value of
end-to-end F minus the best two-stage baseline at that configuration.
Blue = end-to-end ahead, red = two-stage ahead, printed value in units of
the outcome. The honest headline is that red dominates at low noise: with a
learnable surface and generous signal, tree+LP is the better estimator, and
end-to-end's edge (if any) lives in the high-noise band and in feasibility,
not raw value.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.titlesize": 9, "figure.dpi": 150,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})

S2 = ["S2-linear", "S2-lasso", "S2-tree"]


def grid(df, row_col, row_vals, lam_vals, fixed=None):
    g = df.groupby([row_col, "lam", "method"])["oracle_val"].mean()
    out = np.full((len(row_vals), len(lam_vals)), np.nan)
    for i, r in enumerate(row_vals):
        for j, l in enumerate(lam_vals):
            try:
                sub = g.loc[(r, l)]
            except KeyError:
                continue
            out[i, j] = sub["F"] - max(sub[m] for m in S2 if m in sub)
    return out


def panel(ax, M, row_vals, lam_vals, row_label, title):
    vmax = np.nanmax(np.abs(M)) or 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    ax.imshow(M, cmap="RdBu", norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                    fontweight="bold",
                    color="white" if abs(v) > 0.55 * vmax else "#222222")
    ax.set_xticks(range(len(lam_vals)), [f"{l:g}" for l in lam_vals])
    ax.set_yticks(range(len(row_vals)), [f"{r:g}" for r in row_vals])
    ax.set_xlabel("effect nonlinearity λ")
    ax.set_ylabel(row_label)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    df = pd.read_csv("results/regime_map.csv")
    if "sigma" not in df.columns:
        df["sigma"] = 0.5
    df["sigma"] = df["sigma"].fillna(0.5)

    base = df[df.sigma == 0.5]
    caps = sorted(base.cap_scale.unique())
    lams = sorted(base.lam.unique())
    M1 = grid(base, "cap_scale", caps, lams)

    noise = df[df.cap_scale == 1.0]
    sigs = sorted(noise.sigma.unique())
    lamsN = sorted(noise[noise.sigma != 0.5].lam.unique()) or lams
    M2 = grid(noise, "sigma", sigs, lamsN)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    panel(axes[0], M1, caps, lams,
          "capacity tightness (×base caps)",
          "(a)  Δ(F − best two-stage), by tightness — σ = 0.5")
    panel(axes[1], M2, sigs, lamsN,
          "outcome noise σ  (effects ≈ 1–4)",
          "(b)  Δ(F − best two-stage), by noise — caps ×1")
    fig.suptitle("When does end-to-end beat two-stage? (ground-truth value, "
                 "Adult semi-synthetic, N=2000)", y=1.04, fontsize=9.5,
                 fontweight="bold")
    fig.tight_layout(w_pad=2.2)
    os.makedirs("figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig_regime.{ext}", dpi=300 if ext == "png" else None)
    print("[fig] wrote figures/fig_regime.pdf/.png")
    print("      panel (a) range:", np.nanmin(M1).round(2), "..", np.nanmax(M1).round(2))
    print("      panel (b) range:", np.nanmin(M2).round(2), "..", np.nanmax(M2).round(2))


if __name__ == "__main__":
    main()

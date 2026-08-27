"""Per-dataset delay-adjusted reward (replaces the pooled index figure).

One panel per dataset. Each panel shows the upper envelope of the price-trained
family against the upper envelope of the decision-blind family -- the best method
available to each side at each delay cost -- with a rule at the crossover where
the price-trained frontier overtakes. Per-method detail lives in the results table;
this figure answers one question, so it carries two lines.

No averaging across datasets: they differ in arms, caps and outcome scale, so only
the within-dataset comparison means anything.
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_style import METHOD_STYLE, apply as _apply
_apply()

NICE = {"nonnested": "ten-arm synthetic", "mechanism": "dense-margin synthetic",
        "adultsemi": "Adult semi-synthetic", "actg": "ACTG 175",
        "diabetes": "Diabetes 130-US", "criteo": "Criteo (control)"}
ORDER = ["ten", "mech", "adult", "diab", "actg", "crit"]
ORDER = ["nonnested", "mechanism", "adultsemi", "diabetes", "actg", "criteo"]
SHOW = ["F", "Gs", "Alt", "S2-mlp"]   # named methods only -- no "best of family"
LBL  = {"F": "DPL-nc", "Gs": "DPL-cvx", "Alt": "DPL-alt", "S2-mlp": "PtO-mlp"}
KMAX = 0.06
C_DPL = METHOD_STYLE.get("F", {}).get("color", "#2c6fbb")
C_PTO = METHOD_STYLE.get("S2-mlp", {}).get("color", "#a03a31")

d = json.load(open("results/deploy_index.json"))
kap = np.array(d["kappas"]); sel = kap <= KMAX
CX = {}
fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.6), sharex=True)
axes = axes.ravel()

for ax, ds in zip(axes, ORDER):
    e = d["datasets"][ds]; cur = e["curves"]
    # back to the dataset's own outcome units: the index normalised only so that
    # datasets could be pooled, and we no longer pool. On a control dataset the
    # normalisation divides by a near-zero span and manufactures a large-looking gap.
    a = lambda v: e["v_rand"] + e["span"] * np.asarray(v)
    y = {m: a(cur[m])[sel] for m in SHOW if m in cur}
    dpl, pto, alt = y.get("F"), y.get("S2-mlp"), y.get("Alt")
    k = kap[sel]
    ax.fill_between(k, pto, dpl, where=dpl >= pto, color=C_DPL, alpha=0.08, lw=0)
    for m in SHOW:
        if m not in y:
            continue
        st = dict(METHOD_STYLE.get(m, {}))
        for kk in ("marker", "alpha", "label"):
            st.pop(kk, None)
        ax.plot(k, y[m], label=LBL[m], **st)
    # crossover under THIS definition of "ours" (excludes the shortcut), so the
    # rule and the curves cannot disagree
    ahead = np.where(dpl >= pto)[0]   # DPL-nc vs the capacity-matched PtO-mlp
    cx = float(k[ahead[0]]) if len(ahead) else None
    if cx and 0 < cx <= KMAX:
        ax.axvline(cx, color="#8a8f98", lw=0.9, ls=(0, (3, 2)), zorder=0)
        ha = "right" if cx > 0.6 * KMAX else "left"
        ax.annotate(rf"$\kappa^\star={cx:.3f}$", xy=(cx, 0.03),
                    xycoords=("data", "axes fraction"),
                    xytext=(-3 if ha == "right" else 3, 0), ha=ha,
                    textcoords="offset points", fontsize=6.5, color="#5f6570")
    elif cx == 0 and ds != "criteo":
        ax.annotate("DPL-nc ahead at every $\\kappa$", xy=(0.05, 0.06),
                    xycoords="axes fraction", fontsize=6.5, color="#5f6570")
    if ds == "criteo":
        stack = [dpl, pto] + ([alt] if alt is not None else [])
        spread = float(np.max(np.max(stack, axis=0) - np.min(stack, axis=0)))
        ax.annotate(f"no separation:\nall four within {spread:.3f}",
                    xy=(0.05, 0.06), xycoords="axes fraction",
                    fontsize=6.5, color="#5f6570")
    CX[ds] = cx
    ax.set_title(NICE[ds], loc="left", fontweight="bold", fontsize=8)
    if ds in ORDER[3:]:
        ax.set_xlabel(r"delay cost $\kappa$")
    ax.grid(True, axis="y", alpha=0.28)
    ax.margins(x=0)
    ax.set_xticks([0, 0.03, 0.06])
for a in (axes[0], axes[3]):
    a.set_ylabel("delay-adjusted reward")
for ax in axes:
    ax.ticklabel_format(axis="y", style="plain")

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.06), columnspacing=1.4, handlelength=1.8)
fig.tight_layout(w_pad=1.6, h_pad=1.2)
os.makedirs("figures", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig_index_perdataset.{ext}", bbox_inches="tight",
                dpi=300 if ext == "png" else None)
print("[fig] wrote figures/fig_index_perdataset.pdf/.png")
print("crossovers (DPL-nc vs PtO-mlp):", {NICE[x]: CX[x] for x in ORDER})

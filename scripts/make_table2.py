"""Regenerate the Experiments 'deployment scale' table (Table 2) from the
current *_sweep_seeds.csv files, at the largest N, 10 seeds.

Reference baseline = PtO-mlp (the capacity-matched network), per the
2026-08 decision. Columns, per method DPL-nc (F), DPL-cvx (Gs),
DPL-alt (Alt), PtO-mlp (S2-mlp):
  * Average horizon usage of the most-loaded scarce arm, as % of its cap
    (was mislabelled 'peak use'); >100 is a capacity violation.
  * Mean wait (interarrival periods).
And one value column: DPL-nc improvement over PtO-mlp (%), on the metric
each dataset is scored by (oracle value where ground truth exists, else
held-out IPW value).

Prints a human table and a ready-to-paste LaTeX tabular.

Usage:  PYTHONPATH=. python3 scripts/make_table2.py
"""

import os
import numpy as np
import pandas as pd

# dataset key -> (display name, latex row label, cap, scarce arms, score col)
DATASETS = [
    ("nonnested", "ten-arm synthetic",    "ten-arm synthetic",    0.10, list(range(1, 10)), "oracle_val"),
    ("mechanism", "dense-margin",          "dense-margin synth.",  0.25, [1],                 "oracle_val"),
    ("adultsemi", "Adult semi-synthetic",  "Adult semi-synthetic", 0.08, list(range(1, 8)),  "oracle_val"),
    ("diabetes",  "Diabetes 130-US",       "Diabetes 130-US",      0.25, [1, 2],              "ipw_val"),
    ("actg",      "ACTG 175",              "ACTG 175",             0.30, [1, 3],              "ipw_val"),
    ("criteo",    "Criteo Uplift",         "Criteo Uplift",        0.50, [1],                 "ipw_val"),
]
# paper method label -> csv method tag
METHODS = [("nc", "F"), ("cvx", "Gs"), ("alt", "Alt"), ("mlp", "S2-mlp")]


def _rows_at_max_N(key):
    df = pd.read_csv(f"results/{key}_sweep_seeds.csv")
    return df[df["N"] == df["N"].max()]


def _usage(g, cap, scarce):
    peak = 0.0
    for a in scarce:
        c = f"alloc_{a}"
        if c in g.columns:
            peak = max(peak, g[c].mean())
    return 100.0 * peak / cap


def _value(g, score):
    col = score if (score in g.columns and g[score].notna().any()) else "ipw_val"
    return g[col].mean()


def main():
    table = []
    for key, name, label, cap, scarce, score in DATASETS:
        d = _rows_at_max_N(key)
        per = {}
        for lab, tag in METHODS:
            g = d[d["method"] == tag]
            per[lab] = {"use": _usage(g, cap, scarce), "wait": g["mean_wait_all"].mean(),
                        "val": _value(g, score)}
        impr = 100.0 * (per["nc"]["val"] - per["mlp"]["val"]) / abs(per["mlp"]["val"])
        table.append((label, name, per, impr, cap))

    # ---- human-readable ----
    print("\n=== Table 2 numbers (max N, 10 seeds; ref = PtO-mlp) ===")
    print(f"{'dataset':22s} | usage% nc/cvx/alt/mlp        | wait nc/cvx/alt/mlp            | DPL-nc val vs mlp")
    for label, name, per, impr, cap in table:
        u = "/".join(f"{per[m]['use']:.0f}" for m in ("nc", "cvx", "alt", "mlp"))
        w = "/".join(f"{per[m]['wait']:.1f}" for m in ("nc", "cvx", "alt", "mlp"))
        print(f"{name:22s} | {u:24s} | {w:28s} | {impr:+.0f}%   "
              f"(nc {per['nc']['val']:.4f} vs mlp {per['mlp']['val']:.4f})")

    # ---- LaTeX (matches tab:results layout) ----
    print("\n=== LaTeX (paste into tab:results body) ===")
    for label, name, per, impr, cap in table:
        u = " & ".join(f"${per[m]['use']:.0f}$" for m in ("nc", "cvx", "alt", "mlp"))
        w = " & ".join(f"${per[m]['wait']:.1f}$" for m in ("nc", "cvx", "alt", "mlp"))
        print(f"{label:22s} & {u} && {w} && ${impr:+.0f}\\%$ \\\\")


if __name__ == "__main__":
    main()

"""Deployment-Adjusted Policy Value (DAPV): one number per method that
combines policy value, feasibility and queueing delay WITHOUT arbitrary
weights.

Construction. A policy is scored by what it actually delivers under the
deployment queue:

    DAPV(kappa) = V_dep - kappa * W

    V_dep = (1 - u) * V_served + u * V0      (feasibility enters here)
    W     = mean wait over ALL arrivals       (delay enters here)
    u     = fraction never served; V0 = value of no intervention

There is exactly ONE free parameter, kappa >= 0: the cost of one unit of
waiting time (one mean interarrival period; every simulation uses the same
arrival rate, so W is comparable across datasets). Instead of choosing
kappa, we SWEEP it — every reader applies their own delay-aversion and the
crossover points are reported explicitly. Feasibility needs no weight at
all: an infeasible policy mechanically loses value through unserved
arrivals (priced at V0) and queue spill-over.

For semi-/fully-synthetic datasets V_dep is the queue-realized GROUND-TRUTH
outcome (mean_oracle_outcome_all). For real datasets V_served is the IPW
value of the deployed assignment (delivered-as-assigned approximation,
noted) and V0 is the IPW value of the control arm on the eval split.

Cross-dataset combination: per dataset, values are normalized so that
random = 0 and the best method at kappa=0 equals 1; kappa is then in units
of "fraction of the value-at-stake per waiting period". The combined index
is the unweighted mean over datasets.

Outputs results/deploy_index.csv (long form) and results/deploy_index.json
(per-dataset + combined curves, crossovers, leaderboards).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["adultsemi", "actg", "diabetes", "criteo", "lalonde", "nonnested"]
ORACLE = {"adultsemi", "nonnested"}     # ground-truth deployed value exists
METHODS = ["F", "Gs", "Alt", "S2-linear", "S2-lasso", "S2-tree", "S2-knn",
           "S2-dr", "S2-mlp", "random", "treat_all"]
E2E = ("F", "Gs", "Alt")
KAPPAS = [0.0] + list(np.round(np.geomspace(1e-4, 0.2, 45), 6))


def control_value(ds):
    """V0: value of no intervention, on the eval split."""
    import torch
    torch.set_default_dtype(torch.float64)
    if ds == "adultsemi":
        from experiments.data_adult_semi import load_adult_semi
        _, ev, _ = load_adult_semi()
        return float(ev["Y_pot"][:, 0].mean())
    if ds == "nonnested":
        from experiments.run_cell_nonnested import load_or_build_eval_cache
        ev = load_or_build_eval_cache()
        return float(ev["Y_pot"][:, 0].mean())
    if ds == "actg":
        from experiments.data_actg import load_actg
        _, ev, _ = load_actg()
    elif ds == "diabetes":
        from experiments.run_cell_diabetes import load_or_build_cache
        _, ev, _ = load_or_build_cache(0.7, 0)
    elif ds == "criteo":
        from experiments.run_cell_criteo import load_or_build_criteo_cache
        _, ev, _ = load_or_build_criteo_cache(50_000, "full", 0)
    elif ds == "lalonde":
        from experiments.run_cell_lalonde import load_or_build_lalonde_cache
        _, ev, _ = load_or_build_lalonde_cache(0.7, 0)
    m = (ev["T"] == 0).astype(float)
    return float((ev["Y"] * m / ev["e_T"]).mean())   # IPW value of arm 0


def main():
    rows = []
    v0s = {}
    for ds in DATASETS:
        csv = f"results/{ds}_sweep_seeds.csv"
        if not os.path.exists(csv):
            print(f"[index] SKIP {ds}: no {csv}")
            continue
        v0s[ds] = control_value(ds)
        df = pd.read_csv(csv)
        df = df[df.N == df.N.max()]          # deployment scenario: max training data
        per = df.groupby(["method", "seed"], as_index=False).agg(
            oracle_all=("mean_oracle_outcome_all", "mean"),
            ipw_val=("ipw_val", "mean"),
            u=("frac_unserved", "mean"),
            W=("mean_wait_all", "mean"))
        for _, r in per.iterrows():
            if ds in ORACLE:
                V = r["oracle_all"]           # queue-realized ground truth
            else:
                V = (1 - r["u"]) * r["ipw_val"] + r["u"] * v0s[ds]
            rows.append({"dataset": ds, "method": r["method"],
                         "seed": int(r["seed"]), "V": float(V),
                         "W": float(r["W"]), "u": float(r["u"])})
        print(f"[index] {ds}: V0={v0s[ds]:.3f}, N_dep={df.N.max()}, "
              f"{per['seed'].nunique()} seeds")

    long = pd.DataFrame(rows)
    long.to_csv("results/deploy_index.csv", index=False)

    out = {"kappas": KAPPAS, "datasets": {}, "v0": v0s,
           "e2e": list(E2E)}
    # per-dataset normalization constants (method-level means)
    combined_acc = {}
    for ds, g in long.groupby("dataset"):
        mm = g.groupby("method")[["V", "W", "u"]].mean()
        v_rand = mm.loc["random", "V"] if "random" in mm.index else mm["V"].min()
        v_best0 = mm["V"].max()
        span = max(v_best0 - v_rand, 1e-9)
        methods = [m for m in METHODS if m in mm.index]
        curves, sems = {}, {}
        for m in methods:
            gm = g[g.method == m]
            vn = (gm["V"].values - v_rand) / span
            w = gm["W"].values
            idx = np.array([[v - k * wi for v, wi in zip(vn, w)] for k in KAPPAS])
            curves[m] = [round(float(x), 4) for x in idx.mean(1)]
            sems[m] = [round(float(x.std(ddof=1) / np.sqrt(len(x))), 4)
                       if len(x) > 1 else 0.0 for x in idx]
        # crossover: smallest kappa where best e2e >= best S2 (method-mean curves)
        s2m = [m for m in methods if m.startswith("S2")]
        cross = None
        for i, k in enumerate(KAPPAS):
            be = max(curves[m][i] for m in E2E if m in curves)
            bs = max(curves[m][i] for m in s2m) if s2m else -9e9
            if be >= bs:
                cross = k
                break
        out["datasets"][ds] = {
            "methods": methods, "curves": curves, "sems": sems,
            "span": round(span, 4), "v_rand": round(float(v_rand), 4),
            "crossover": cross,
            "means": {m: {"V": round(float(mm.loc[m, "V"]), 4),
                          "Vn": round(float((mm.loc[m, "V"] - v_rand) / span), 4),
                          "W": round(float(mm.loc[m, "W"]), 3),
                          "u": round(float(mm.loc[m, "u"]), 4)}
                      for m in methods},
        }
        for m in methods:
            combined_acc.setdefault(m, []).append(np.array(curves[m]))
        print(f"[index] {ds}: crossover kappa = {cross}")

    # combined: unweighted mean over datasets where the method exists in all
    n_ds = len(out["datasets"])
    comb = {m: np.mean(a, axis=0) for m, a in combined_acc.items()
            if len(a) == n_ds}
    out["combined"] = {m: [round(float(x), 4) for x in v] for m, v in comb.items()}
    s2m = [m for m in comb if m.startswith("S2")]
    cross = None
    for i, k in enumerate(KAPPAS):
        be = max(comb[m][i] for m in E2E if m in comb)
        bs = max(comb[m][i] for m in s2m) if s2m else -9e9
        if be >= bs:
            cross = k
            break
    out["combined_crossover"] = cross
    print(f"[index] COMBINED ({n_ds} datasets) crossover kappa = {cross}")
    for i, k in enumerate(KAPPAS):
        if k in (0.0, 0.01, 0.03, 0.1):
            rank = sorted(comb.items(), key=lambda kv: -kv[1][i])[:4]
            print(f"   kappa={k:<5} top: " +
                  ", ".join(f"{m}={v[i]:+.3f}" for m, v in rank))

    with open("results/deploy_index.json", "w") as f:
        json.dump(out, f)
    print("[index] wrote results/deploy_index.csv + .json")


if __name__ == "__main__":
    main()

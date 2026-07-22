"""Robustness of the queue-derived quantities (u, W) to the simulation
horizon.

The unserved fraction u and mean wait W come from a discrete-event
simulation that ends at T_max = max_time_mult x (last arrival time); an
arrival still queued at T_max counts as unserved and its wait is censored
at T_max - t_arrival. `max_time_mult = 1.5` is the one free horizon choice
in the whole DAPV construction, so this script re-simulates every trained
method at several multipliers and reports u, W and deployed value per
(method, multiplier): if the method ORDERING is stable across multipliers,
no conclusion hinges on the 1.5.

Trains on Adult semi-synthetic at deployment scale (N = 16,000, seed 0)
once, then sweeps the horizon with 5 paired arrival streams each.

Writes results/horizon_sensitivity.csv.

Run inside an allocation:
    PYTHONPATH=. python3 scripts/horizon_sensitivity.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (            # noqa: E402
    S2_METHODS,
    arms_and_assigner_from_model,
    make_random_assigner,
    s2_arms_and_assigner,
    simulate_one_method,
)
from src.train import train_GF              # noqa: E402
from src.train_alt import train_alt         # noqa: E402

N, SEED, STEPS = 16_000, 0, 800
MULTS = [1.25, 1.5, 2.0, 3.0]
SIM_SEEDS = [0, 1, 2, 3, 4]


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    # same prepare as the sweep cells (loader defaults + subsample formula)
    from experiments.common import subsample_rows            # noqa: E402
    from experiments.data_adult_semi import load_adult_semi  # lazy: heavy
    train_full, ev, cfg = load_adult_semi(seed=0)
    td = subsample_rows(train_full, N, seed=0 * 1_000_003 + SEED * 9973 + N)
    T, D, TAU, B = int(cfg["T"]), int(cfg["D"]), float(cfg["TAU"]), cfg["B"]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    assigners = {}
    mF, _, _ = train_GF(kind="F", train_data=td, D=D, T=T, tau=0.03, b=B,
                        steps=STEPS, lr=5e-3, log_every=STEPS, seed=SEED)
    assigners["F"] = arms_and_assigner_from_model(mF, td, ev, B, 0.92)[2]
    mA, _, _ = train_alt(train_data=td, D=D, T=T, tau=0.03, b=B,
                         outer_steps=STEPS, inner_freq=5, lr=5e-3,
                         log_every=STEPS, seed=SEED)
    assigners["Alt"] = arms_and_assigner_from_model(mA, td, ev, B, 0.92)[2]
    mG, _, _ = train_GF(kind="Gs", train_data=td, D=D, T=T, tau=TAU, b=B,
                        steps=STEPS, lr=5e-3, log_every=STEPS, seed=SEED)
    assigners["Gs"] = arms_and_assigner_from_model(mG, td, ev, B, 0.92)[2]
    for m in S2_METHODS + ["mlp"]:
        assigners[f"S2-{m}"] = s2_arms_and_assigner(td, ev, T, B, m,
                                                    mlp_steps=STEPS)[2]
    assigners["random"] = make_random_assigner(T)

    rows = []
    for mult in MULTS:
        for ss in SIM_SEEDS:
            for name, asg in assigners.items():
                r = simulate_one_method(asg, name, ev, T, B, 1000, 1.0,
                                        mult, sim_seed=ss)
                r["max_time_mult"] = mult
                rows.append(r)
        df = pd.DataFrame([r for r in rows if r["max_time_mult"] == mult])
        agg = df.groupby("method")[["frac_unserved", "mean_wait_all",
                                    "mean_oracle_outcome_all"]].mean()
        print(f"\n=== max_time_mult = {mult} ===")
        print(agg.sort_values("mean_oracle_outcome_all",
                              ascending=False).round(4).to_string())

    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/horizon_sensitivity.csv", index=False)
    print("\nwrote results/horizon_sensitivity.csv", out.shape)


if __name__ == "__main__":
    main()

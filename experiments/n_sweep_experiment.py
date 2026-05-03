"""
N-sweep real-queue experiment (no G).

For each training-set size N in --n-values, generate a fresh train split
of size N, train F + the five S2 variants (skip G — its CVXPYLayer is too
slow to sweep), simulate the queueing system, and report per-method
metrics. Eval split is fixed (N_EVAL from src.config) so cross-N values
are comparable.

Outputs:
    results/n_sweep.csv  - per (N, method, sim_seed) raw rows
    results/n_sweep.png  - 2x2 panels of {total_wait, mean_wait_served,
                           oracle_served, frac_unserved} vs N

Run:
    python -m experiments.n_sweep_experiment
    python -m experiments.n_sweep_experiment --n-values 100 200 1000
"""

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import generate_data
from src.train import train_GF
from src.s2_dual import (
    fit_outcome_models,
    get_mhat_matrix,
    solve_dual_lp,
)

from experiments.real_queue_experiment import (
    make_random_assigner,
    make_oracle_greedy_assigner,
    make_gf_assigner,
    make_s2_assigner,
    make_streams,
    simulate,
    aggregate_one,
    S2_METHODS,
)


def _gen(N, seed, D, T):
    return generate_data(
        N=N, seed=seed, d=D, T=T,
        sigma_y=config.SIGMA_Y,
        propensity_strength=config.PROPENSITY_STRENGTH,
        outcome_strength=config.OUTCOME_STRENGTH,
        treatment_effect_strength=config.TREATMENT_EFFECT_STRENGTH,
        clip_propensity=config.CLIP_PROPENSITY,
    )


def train_policies_no_G(train_data, eval_data, T, D, TAU, B, steps, lr, seed):
    policies = {}
    policies["random"] = make_random_assigner(T)
    policies["oracle_greedy_no_cap"] = make_oracle_greedy_assigner(eval_data)

    print("[train] F (implicit diff)")
    model_F, mu_F, _ = train_GF(
        kind="F", train_data=train_data,
        D=D, T=T, tau=TAU, b=B,
        steps=steps, lr=lr, log_every=max(1, steps), seed=seed,
    )
    policies["F"] = make_gf_assigner(
        model_F, mu_F.detach().cpu().numpy(), eval_data, TAU,
    )

    for method in S2_METHODS:
        print(f"[train] S2-{method}")
        outcome_models = fit_outcome_models(
            X_train=train_data["X"],
            T_train=train_data["T"],
            Y_train=train_data["Y"],
            T=T, method=method,
            E_train=train_data["E"],
        )
        M_hat_train = get_mhat_matrix(outcome_models, train_data["X"], T)
        mu_hat, _, _, _ = solve_dual_lp(M_hat_train, B, verbose=False)
        policies[f"S2-{method}"] = make_s2_assigner(
            outcome_models, mu_hat, eval_data, T,
        )

    return policies


def plot_results(agg, methods, out_png):
    metrics = [
        ("total_wait_mean", "Total wait (sum across N_sim)", True),
        ("mean_wait_served_mean", "Mean wait | served", True),
        ("oracle_served_mean", "Mean oracle outcome | served", False),
        ("frac_unserved_mean", "Fraction unserved", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, label, log) in zip(axes.flat, metrics):
        for m in methods:
            sub = agg[agg["method"] == m].sort_values("N")
            ax.plot(sub["N"], sub[col], marker="o", label=m)
        ax.set_xlabel("N (train size)")
        ax.set_ylabel(label)
        if log:
            ax.set_yscale("log")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8, ncol=2)
    fig.suptitle("Real-queue performance vs training size N (G omitted)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130)
    print(f"[plot] wrote {out_png}")


def parse_args():
    p = argparse.ArgumentParser(description="N-sweep real-queue experiment (no G).")
    p.add_argument("--n-values", type=int, nargs="+", default=[100, 200, 1000])
    p.add_argument("--N-sim", type=int, default=5000, dest="N_sim")
    p.add_argument("--lambda-people", type=float, default=1.0)
    p.add_argument("--num-sim-seeds", type=int, default=5)
    p.add_argument("--max-time-mult", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--train-seed", type=int, default=1)
    p.add_argument("--out-csv", type=str, default="results/n_sweep.csv")
    p.add_argument("--out-png", type=str, default="results/n_sweep.png")
    return p.parse_args()


def main():
    args = parse_args()
    config.setup_global_state()

    T = int(config.T)
    D = int(config.D)
    TAU = float(config.TAU)
    B = config.B

    print(f"[main] T={T} D={D} TAU={TAU} B={B}")
    print(f"[main] N values: {args.n_values}")

    eval_data = _gen(N=config.N_EVAL, seed=config.EVAL_SEED, D=D, T=T)

    rows = []
    for N in args.n_values:
        print(f"\n========== N = {N} ==========")
        t_train_0 = time.time()
        train_data = _gen(N=N, seed=config.TRAIN_SEED, D=D, T=T)
        policies = train_policies_no_G(
            train_data=train_data,
            eval_data=eval_data,
            T=T, D=D, TAU=TAU, B=B,
            steps=args.steps, lr=args.lr, seed=args.train_seed,
        )
        print(f"[N={N}] train wall: {time.time() - t_train_0:.1f}s")

        for sim_seed in range(args.num_sim_seeds):
            people_t, person_idx, T_max, resource_t = make_streams(
                eval_data, args.N_sim, args.lambda_people, B,
                args.max_time_mult, seed=sim_seed * 7 + 13,
            )
            for method, assigner in policies.items():
                t0 = time.time()
                recs = simulate(
                    people_t, person_idx, resource_t, assigner,
                    T=T, T_max=T_max, eval_data=eval_data, sim_seed=sim_seed,
                )
                wall = time.time() - t0
                row = aggregate_one(recs, method, sim_seed, B, args.N_sim, wall)
                row["N"] = int(N)
                rows.append(row)
                print(f"  N={N:5d}  seed={sim_seed}  {method:25s}  "
                      f"total_wait={row['total_wait']:10.2f}  "
                      f"oracle={row['mean_oracle_outcome_served']: .4f}  "
                      f"unserved={row['frac_unserved']:6.2%}")

    df = pd.DataFrame(rows)

    agg = df.groupby(["N", "method"]).agg(
        total_wait_mean=("total_wait", "mean"),
        mean_wait_served_mean=("mean_wait_served", "mean"),
        oracle_served_mean=("mean_oracle_outcome_served", "mean"),
        frac_unserved_mean=("frac_unserved", "mean"),
    ).reset_index()

    print("\n" + "=" * 100)
    print("N-SWEEP SUMMARY (mean across sim_seeds)")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x: .4f}",
                           "display.width", 200):
        print(agg.to_string(index=False))
    print("=" * 100)

    if args.out_csv:
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"[main] wrote {args.out_csv}")

    if args.out_png:
        out_dir = os.path.dirname(args.out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        method_order = (
            ["random", "oracle_greedy_no_cap", "F"]
            + [f"S2-{m}" for m in S2_METHODS]
        )
        methods_present = [m for m in method_order if m in agg["method"].values]
        plot_results(agg, methods_present, args.out_png)


if __name__ == "__main__":
    main()

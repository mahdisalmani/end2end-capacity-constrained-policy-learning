"""
Real-queue deployment experiment (snapshot-based, softmax-sampling G/F).

Phase 1: train every policy method (random, oracle-greedy-no-cap, G, F, and
the five S2 variants) on the train snapshot produced by `generate_data.py`.
Artifacts are then frozen.

Phase 2: simulate real-world deployment as a queueing system. People arrive
as a Poisson process at rate `lambda_people`. Per-arm resources arrive as
independent Poisson processes at rate `b_t * lambda_people`, so a method
that assigns more than `b_t` mass to arm t will see arm-t's queue grow
without bound. Idle resources are held in inventory until claimed.

DEPLOYMENT NOTE: this script deploys G/F by *sampling* from the softmax
policy (`make_gf_assigner` below) with no capacity buffer — the project's
original stochastic deployment convention. The newer sweep/cell harnesses
instead deploy argmax(M - mu) with a sub-capacity buffer
(`experiments.common.arms_and_assigner_from_model`), so wait-time numbers
from this script are NOT directly comparable to theirs.

The queue simulator itself lives in `experiments.common`; this module
re-exports it for backward compatibility.

Run:
    python -m experiments.real_queue_experiment
    python -m experiments.real_queue_experiment --N-sim 2000 --num-sim-seeds 3
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from src import config
from src.data import load_experiment
from src.inner_G import initialize_G_layer
from src.train import train_GF
from src.s2_dual import (
    fit_outcome_models,
    get_mhat_matrix,
    solve_dual_lp,
)

# Shared primitives (canonical home: experiments/common.py). Re-exported
# here because many older scripts import them from this module.
from experiments.common import (  # noqa: F401  (re-exports)
    S2_METHODS,
    aggregate_one,
    make_oracle_greedy_assigner,
    make_random_assigner,
    make_s2_assigner,
    make_streams,
    simulate,
)


def make_gf_assigner(model, mu_train, eval_data, tau):
    """Sample from softmax((M(x) - mu)/tau). Cumulative probs precomputed.

    Stochastic deployment (this script only): expected allocation equals the
    softmax policy mass, so caps can be transiently exceeded — the queues
    absorb the excess. The sweep harnesses use the deterministic
    argmax + cap-buffer deployment instead.
    """
    with torch.no_grad():
        M = model(torch.tensor(eval_data["X"])).numpy()
    logits = (M - mu_train[None, :]) / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cumprobs = probs.cumsum(axis=1)

    def assign(rng, person_idx):
        u = rng.random()
        return int(np.searchsorted(cumprobs[person_idx], u))

    return assign


def train_all_policies(train_data, eval_data, cfg, steps, lr, seed):
    T = int(cfg["T"])
    D = int(cfg["D"])
    TAU = float(cfg["TAU"])
    B = cfg["B"]

    policies = {}

    print("[train] random + oracle baselines")
    policies["random"] = make_random_assigner(T)
    policies["oracle_greedy_no_cap"] = make_oracle_greedy_assigner(eval_data)

    print("[train] G (CVXPYLayer, convex dual)")
    model_G, mu_G, _ = train_GF(
        kind="G", train_data=train_data,
        D=D, T=T, tau=TAU, b=B,
        steps=steps, lr=lr, log_every=max(1, steps), seed=seed,
    )
    policies["G"] = make_gf_assigner(
        model_G, mu_G.detach().cpu().numpy(), eval_data, TAU,
    )

    print("[train] F (implicit diff, non-convex literal)")
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


def print_summary(df, N_sim, num_sim_seeds, B):
    agg = df.groupby("method").agg(
        total_wait_mean=("total_wait", "mean"),
        total_wait_std=("total_wait", "std"),
        mean_wait_served_mean=("mean_wait_served", "mean"),
        oracle_served_mean=("mean_oracle_outcome_served", "mean"),
        oracle_served_std=("mean_oracle_outcome_served", "std"),
        frac_unserved_mean=("frac_unserved", "mean"),
    ).reset_index()
    rank = {"random": 0, "oracle_greedy_no_cap": 1, "G": 2, "F": 3}
    agg["_rk"] = agg["method"].map(lambda m: rank.get(m, 10))
    agg = agg.sort_values(["_rk", "method"]).drop(columns=["_rk"]).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("REAL-QUEUE EXPERIMENT SUMMARY")
    print(f"N_sim={N_sim}  sim_seeds={num_sim_seeds}  "
          f"B={np.array2string(np.asarray(B), precision=3)}")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x: .4f}",
                           "display.width", 200):
        print(agg.to_string(index=False))
    print("=" * 100)


def parse_args():
    p = argparse.ArgumentParser(description="Real-queue deployment experiment.")
    p.add_argument("--N-sim", type=int, default=10_000, dest="N_sim")
    p.add_argument("--lambda-people", type=float, default=1.0)
    p.add_argument("--num-sim-seeds", type=int, default=10)
    p.add_argument("--max-time-mult", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--train-seed", type=int, default=1)
    p.add_argument("--out-csv", type=str, default="results/real_queue.csv")
    return p.parse_args()


def main():
    args = parse_args()
    config.setup_global_state()

    print("[main] loading snapshot")
    train_data, eval_data, cfg = load_experiment(
        config.TRAIN_DATA_PATH,
        config.EVAL_DATA_PATH,
        config.CONFIG_PATH,
    )

    N = int(cfg["N"])
    T = int(cfg["T"])
    TAU = float(cfg["TAU"])
    B = cfg["B"]
    print(f"[main] N={N} T={T} TAU={TAU} B={B}")

    initialize_G_layer(N=N, T=T, tau=TAU, b=B)

    print("\n[main] training all policies (one shot)")
    policies = train_all_policies(
        train_data, eval_data, cfg,
        steps=args.steps, lr=args.lr, seed=args.train_seed,
    )

    rows = []
    for sim_seed in range(args.num_sim_seeds):
        print(f"\n=== sim_seed={sim_seed} ===")
        people_t, person_idx, T_max, resource_t = make_streams(
            eval_data, args.N_sim, args.lambda_people, B,
            args.max_time_mult, seed=sim_seed * 7 + 13,
        )
        for method, assigner in policies.items():
            t0 = time.time()
            recs = simulate(
                people_t, person_idx, resource_t, assigner,
                T=T, T_max=T_max,
                eval_data=eval_data, sim_seed=sim_seed,
            )
            wall = time.time() - t0
            row = aggregate_one(recs, method, sim_seed, B, args.N_sim, wall)
            rows.append(row)
            print(f"  {method:25s}  total_wait={row['total_wait']:10.2f}  "
                  f"mean_wait_served={row['mean_wait_served']:7.4f}  "
                  f"oracle_served={row['mean_oracle_outcome_served']: .4f}  "
                  f"unserved={row['frac_unserved']:6.2%}  "
                  f"({wall:.2f}s)")

    df = pd.DataFrame(rows)
    print_summary(df, args.N_sim, args.num_sim_seeds, B)

    if args.out_csv:
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\n[main] wrote {args.out_csv}")


if __name__ == "__main__":
    main()

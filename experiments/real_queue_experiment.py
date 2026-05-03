"""
Real-queue deployment experiment.

Phase 1: train every policy method (random, oracle-greedy-no-cap, G, F, and the
five S2 variants) on the train snapshot. Artifacts are then frozen.

Phase 2: simulate real-world deployment as a queueing system. People arrive as
a Poisson process at rate `lambda_people`. Per-arm resources arrive as
independent Poisson processes at rate `b_t * lambda_people`, so a method that
assigns more than `b_t` mass to arm t will see arm-t's queue grow without
bound. Idle resources are held in inventory until claimed.

For each (method, sim_seed) we report total waiting time and average oracle
outcome for served people, plus the number unserved (censored at T_max).

Run:
    python -m experiments.real_queue_experiment
    python -m experiments.real_queue_experiment --N-sim 2000 --num-sim-seeds 3
"""

import argparse
import os
import time
from collections import deque

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


S2_METHODS = ["linear", "lasso", "tree", "knn", "dr"]


# === Block A: trained-policy assignment functions ============================
# Every assigner has signature (rng, person_idx_in_eval) -> arm. Trained
# artifacts are precomputed at construction time so the inner simulation loop
# does no torch / sklearn calls.

def make_random_assigner(T):
    def assign(rng, person_idx):
        return int(rng.integers(T))
    return assign


def make_oracle_greedy_assigner(eval_data):
    a_star = eval_data["Y_pot"].argmax(axis=1)
    def assign(rng, person_idx):
        return int(a_star[person_idx])
    return assign


def make_gf_assigner(model, mu_train, eval_data, tau):
    """Sample from softmax((M(x) - mu)/tau). Cumulative probs precomputed."""
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


def make_s2_assigner(outcome_models, mu_hat, eval_data, T):
    """Deterministic argmax over (m_hat(x) - mu_hat)."""
    M_hat = get_mhat_matrix(outcome_models, eval_data["X"], T)
    a_star = (M_hat - mu_hat[None, :]).argmax(axis=1)
    def assign(rng, person_idx):
        return int(a_star[person_idx])
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


# === Block B: arrival streams ================================================
# One paired stream per sim_seed, shared across all methods so the method
# delta is not contaminated by Poisson noise.

def make_streams(eval_data, N_sim, lambda_people, B, max_time_mult, seed):
    rng = np.random.default_rng(seed)
    N_eval = eval_data["X"].shape[0]
    T = len(B)

    inter = rng.exponential(scale=1.0 / lambda_people, size=N_sim)
    people_t = np.cumsum(inter)
    person_idx = rng.integers(0, N_eval, size=N_sim)

    T_max = float(people_t[-1] * max_time_mult)

    resource_t = []
    for t in range(T):
        rate = float(B[t]) * lambda_people
        if rate <= 0.0:
            resource_t.append(np.empty(0, dtype=np.float64))
            continue
        expected = rate * T_max
        n_init = max(16, int(expected + 8.0 * np.sqrt(expected) + 16.0))
        ts = np.cumsum(rng.exponential(scale=1.0 / rate, size=n_init))
        while ts[-1] < T_max:
            extra = np.cumsum(rng.exponential(scale=1.0 / rate, size=n_init))
            ts = np.concatenate([ts, ts[-1] + extra])
        ts = ts[ts <= T_max]
        resource_t.append(ts)

    return people_t, person_idx, T_max, resource_t


# === Block C: discrete-event simulator =======================================
def simulate(people_t, person_idx, resource_t, assigner, T, T_max,
             eval_data, sim_seed):
    """One queueing simulation. Returns dict of per-person arrays of length N_sim."""
    Y_pot = eval_data["Y_pot"]
    rng = np.random.default_rng(sim_seed * 9_973_337 + 1)
    N_sim = len(people_t)

    inventory = np.zeros(T, dtype=np.int64)
    queues = [deque() for _ in range(T)]
    next_r_idx = np.zeros(T, dtype=np.int64)

    arms = np.zeros(N_sim, dtype=np.int64)
    waits = np.zeros(N_sim, dtype=np.float64)
    served = np.zeros(N_sim, dtype=bool)
    outcomes = np.full(N_sim, np.nan, dtype=np.float64)

    def serve_resources_until(t_now):
        for a in range(T):
            ts = resource_t[a]
            idx = next_r_idx[a]
            n = len(ts)
            while idx < n and ts[idx] <= t_now:
                t_r = ts[idx]
                idx += 1
                if queues[a]:
                    t_arr, k, p_idx = queues[a].popleft()
                    waits[k] = t_r - t_arr
                    served[k] = True
                    arms[k] = a
                    outcomes[k] = Y_pot[p_idx, a]
                else:
                    inventory[a] += 1
            next_r_idx[a] = idx

    for k in range(N_sim):
        t_p = people_t[k]
        serve_resources_until(t_p)

        a = assigner(rng, person_idx[k])
        if inventory[a] > 0:
            inventory[a] -= 1
            arms[k] = a
            waits[k] = 0.0
            served[k] = True
            outcomes[k] = Y_pot[person_idx[k], a]
        else:
            queues[a].append((t_p, k, person_idx[k]))

    serve_resources_until(T_max)

    for a in range(T):
        while queues[a]:
            t_arr, k, p_idx = queues[a].popleft()
            arms[k] = a
            waits[k] = T_max - t_arr
            served[k] = False

    return {
        "arm": arms,
        "person_idx": person_idx,
        "wait": waits,
        "served": served,
        "oracle_outcome": outcomes,
    }


# === Block D: aggregate + report =============================================

def aggregate_one(records, method, sim_seed, B, N_sim, sim_wall):
    arms = records["arm"]
    waits = records["wait"]
    served = records["served"]
    outcomes = records["oracle_outcome"]
    T = len(B)

    n_unserved = int((~served).sum())
    served_waits = waits[served]
    served_outcomes = outcomes[served]

    row = {
        "method": method,
        "sim_seed": sim_seed,
        "N_sim": N_sim,
        "total_wait": float(waits.sum()),
        "mean_wait_all": float(waits.mean()),
        "mean_wait_served": (float(served_waits.mean())
                             if served_waits.size > 0 else float("nan")),
        "mean_oracle_outcome_served": (float(served_outcomes.mean())
                                       if served_outcomes.size > 0 else float("nan")),
        "num_unserved": n_unserved,
        "frac_unserved": n_unserved / N_sim,
        "sim_wall_s": sim_wall,
    }
    counts = np.bincount(arms, minlength=T)
    fracs = counts / N_sim
    for t in range(T):
        row[f"alloc_{t}"] = float(fracs[t])
    return row


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


# === Block E: main ===========================================================
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

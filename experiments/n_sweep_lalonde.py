"""
N-sweep experiment on the LaLonde NSW + PSID-1 real dataset (single-process).

Harness:
  - Loads real data via `experiments.data_lalonde.load_lalonde`.
  - Train / eval is a fixed 70 / 30 split of the LaLonde data.
  - Per `--n-values N`, subsample N rows from the train split and
    retrain F + each S2 method; deploy on the SAME eval split for
    every N. (The plot's x-axis is "training subsample size".)
  - Reports IPW-estimated policy value on the eval split (not the
    synthetic oracle outcome, which is unknown for real data).
  - Replaces `oracle_greedy_no_cap` with `treat_all` (always assign
    T=1), since oracle-greedy needs counterfactual Y_pot.

The multi-seed, multiprocessing version is `experiments.sweep_lalonde`.
Shared machinery lives in `experiments.common`.

Run:
    python -m experiments.n_sweep_lalonde --n-values 100 200 500 1000 1800
"""

import argparse
import os
import time
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.train import train_GF

from experiments.common import (
    aggregate_one,
    arms_and_assigner_from_model,
    ipw_policy_value,
    make_random_assigner,
    make_streams,
    make_treat_all_assigner,
    precompute_arms,
    s2_arms_and_assigner,
    simulate,
    subsample_rows as _subsample,
)
from experiments.data_lalonde import load_lalonde


# S2 baselines run on LaLonde. tree and dr are skipped: with N as small as
# a few hundred, the per-arm tree overfits pathologically and the DR
# pseudo-outcome LassoCV is unstable.
LALONDE_S2_METHODS = ("linear", "lasso", "knn")


def make_gf_assigner(model, mu_train, eval_data, tau, B, train_data,
                     cap_buffer=0.92):
    """Deterministic deployment for F: argmax(M - mu) with the dual LP
    re-solved on train scores at `cap_buffer * B`. See
    `experiments.common.arms_and_assigner_from_model` (mu_train and tau are
    accepted for signature compatibility; deployment uses the calibrated
    LP prices, not the training-time mu)."""
    _, _, assigner = arms_and_assigner_from_model(
        model, train_data, eval_data, B, cap_buffer,
    )
    return assigner


def _stable_seed(*parts):
    """Deterministic 32-bit seed from string parts (unlike builtin hash(),
    which is salted per process and breaks run-to-run reproducibility)."""
    return zlib.crc32("|".join(map(str, parts)).encode())


# === Block A: train all methods ==============================================

def train_policies(train_data, eval_data, T, D, TAU, B, steps, lr, seed,
                   f_tau=0.03, cap_buffer=0.92):
    policies = {}
    policies["random"] = make_random_assigner(T)
    policies["treat_all"] = make_treat_all_assigner()

    f_tau_use = float(f_tau)
    print(f"[train] F (implicit diff)  tau={f_tau_use}  cap_buffer={cap_buffer}")
    model_F, mu_F, _ = train_GF(
        kind="F", train_data=train_data,
        D=D, T=T, tau=f_tau_use, b=B,
        steps=steps, lr=lr, log_every=max(1, steps), seed=seed,
    )
    policies["F"] = make_gf_assigner(
        model_F, mu_F.detach().cpu().numpy(), eval_data, f_tau_use, B,
        train_data=train_data, cap_buffer=cap_buffer,
    )

    for method in LALONDE_S2_METHODS:
        print(f"[train] S2-{method}")
        _, _, policies[f"S2-{method}"] = s2_arms_and_assigner(
            train_data, eval_data, T, B, method,
        )

    return policies


# === Block B: plot ===========================================================

def plot_results(agg, methods, out_png):
    """1x3 panel: mean wait time, IPW policy value, unserved %.
    F gets a thicker dark-blue line, drawn on top."""

    def _style(m):
        if m == "F":
            return dict(color="#0b5394", linewidth=2.6, marker="o",
                        markersize=8, markeredgecolor="white",
                        markeredgewidth=0.8, zorder=5,
                        label="F (proposed)")
        return dict(linewidth=1.0, marker="o", markersize=4.5,
                    alpha=0.85, zorder=2, label=m)

    def _draw(ax, ycol, ytrans=lambda y: y):
        for m in methods:
            if m == "F":
                continue
            sub = agg[agg["method"] == m].sort_values("N")
            ax.plot(sub["N"], ytrans(sub[ycol]), **_style(m))
        if "F" in methods:
            sub = agg[agg["method"] == "F"].sort_values("N")
            ax.plot(sub["N"], ytrans(sub[ycol]), **_style("F"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    _draw(ax, "mean_wait_served_mean")
    ax.set_xlabel("N (train subsample size)")
    ax.set_ylabel("Mean wait time (served)")
    ax.set_yscale("log")
    ax.set_title("Mean wait time vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    ax = axes[1]
    _draw(ax, "ipw_value_mean")
    ax.set_xlabel("N (train subsample size)")
    ax.set_ylabel("IPW policy value (eval, $1k)")
    ax.set_title("IPW value vs N")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    ax = axes[2]
    _draw(ax, "frac_unserved_mean", ytrans=lambda y: 100.0 * y)
    ax.set_xlabel("N (train subsample size)")
    ax.set_ylabel("Unserved (%)")
    ax.set_title("Unserved fraction vs N")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle("LaLonde NSW + PSID-1: F vs S2 (G omitted)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    print(f"[plot] wrote {out_png}")


# === Block C: main ===========================================================

def parse_args():
    p = argparse.ArgumentParser(description="LaLonde N-sweep (real data).")
    p.add_argument("--n-values", type=int, nargs="+",
                   default=[100, 200, 500, 1000, 1800])
    p.add_argument("--N-sim", type=int, default=1000, dest="N_sim",
                   help="Simulator arrival count (drawn with replacement "
                        "from eval split).")
    p.add_argument("--lambda-people", type=float, default=1.0)
    p.add_argument("--num-sim-seeds", type=int, default=100)
    p.add_argument("--max-time-mult", type=float, default=1.5)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--train-seed", type=int, default=1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--f-tau", type=float, default=0.03)
    p.add_argument("--cap-buffer", type=float, default=0.92)
    p.add_argument("--out-csv", type=str, default="results/lalonde_sweep.csv")
    p.add_argument("--out-png", type=str, default="results/lalonde_sweep.png")
    p.add_argument("--methods", type=str, nargs="+", default=None,
                   help="Subset of method names to keep in plot/table.")
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducible torch + numpy state.
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)

    train_full, eval_data, cfg = load_lalonde(seed=args.split_seed)
    T = int(cfg["T"])
    D = int(cfg["D"])
    TAU = float(cfg["TAU"])
    B = cfg["B"]

    n_eval = len(eval_data["T"])

    rows = []
    for N in args.n_values:
        print(f"\n========== N = {N} ==========")
        td = _subsample(train_full, N, seed=args.split_seed * 1000 + N)
        t0 = time.time()
        policies = train_policies(
            train_data=td,
            eval_data=eval_data,
            T=T, D=D, TAU=TAU, B=B,
            steps=args.steps, lr=args.lr, seed=args.train_seed,
            f_tau=args.f_tau, cap_buffer=args.cap_buffer,
        )
        if args.methods is not None:
            missing = [m for m in args.methods if m not in policies]
            if missing:
                raise ValueError(f"Unknown methods: {missing}. "
                                 f"Available: {list(policies)}")
            policies = {m: policies[m] for m in args.methods}
        print(f"[N={N}] train wall: {time.time() - t0:.1f}s, "
              f"methods: {list(policies)}")

        # IPW policy value (precompute once per method on the eval split).
        ipw_per_method = {}
        for method, assigner in policies.items():
            arms_eval = precompute_arms(
                assigner, n_eval, rng_seed=_stable_seed(method, "ipw"))
            ipw_per_method[method] = ipw_policy_value(arms_eval, eval_data)
            print(f"  {method:25s}  IPW = {ipw_per_method[method]:8.3f}")

        for sim_seed in range(args.num_sim_seeds):
            people_t, person_idx, T_max, resource_t = make_streams(
                eval_data, args.N_sim, args.lambda_people, B,
                args.max_time_mult, seed=sim_seed * 7 + 13,
            )
            for method, assigner in policies.items():
                t_sim = time.time()
                recs = simulate(
                    people_t, person_idx, resource_t, assigner,
                    T=T, T_max=T_max, eval_data=eval_data, sim_seed=sim_seed,
                )
                wall = time.time() - t_sim
                row = aggregate_one(recs, method, sim_seed, B, args.N_sim, wall)
                row["N"] = int(N)
                row["ipw_value"] = ipw_per_method[method]
                rows.append(row)

    df = pd.DataFrame(rows)

    agg = df.groupby(["N", "method"]).agg(
        mean_wait_served_mean=("mean_wait_served", "mean"),
        ipw_value_mean=("ipw_value", "mean"),
        frac_unserved_mean=("frac_unserved", "mean"),
    ).reset_index()

    wait_pivot = agg.pivot(index="N", columns="method",
                           values="mean_wait_served_mean").sort_index()
    ipw_pivot = agg.pivot(index="N", columns="method",
                          values="ipw_value_mean").sort_index()
    unserved_pivot = (
        100.0 * agg.pivot(index="N", columns="method",
                          values="frac_unserved_mean")
    ).sort_index()

    print("\n" + "=" * 100)
    print("LaLonde N-SWEEP: mean wait time (served)")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x:10.2f}",
                           "display.width", 200):
        print(wait_pivot.to_string())
    print("=" * 100)
    print("LaLonde N-SWEEP: IPW policy value (eval, thousands of $)")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x:8.3f}",
                           "display.width", 200):
        print(ipw_pivot.to_string())
    print("=" * 100)
    print("LaLonde N-SWEEP: unserved (%)")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x:7.2f}",
                           "display.width", 200):
        print(unserved_pivot.to_string())
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
        method_order = ["random", "treat_all", "F"] + [
            f"S2-{m}" for m in LALONDE_S2_METHODS
        ]
        methods_present = [m for m in method_order
                           if m in agg["method"].values]
        plot_results(agg, methods_present, args.out_png)


if __name__ == "__main__":
    main()

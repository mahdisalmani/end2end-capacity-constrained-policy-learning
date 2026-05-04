"""
N-sweep experiment on the Criteo Uplift (10% sample) real dataset.

Same harness as `experiments/n_sweep_lalonde.py` but:
  - Loads real data via `experiments.data_criteo.load_criteo`.
  - Subsamples ~200k rows from the 1.4M total, then 70 / 30 train/eval.
  - Per `--n-values N`, subsample N rows from train and retrain F +
    each S2 method; deploy on the same eval split for every N.
  - Reports IPW-estimated policy value (using the binary `visit`
    outcome as Y).
  - Same `treat_all` baseline (no oracle_greedy without Y_pot).

Run:
    python -m experiments.n_sweep_criteo --n-values 200 1000 5000 20000 100000
"""

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.train import train_GF
from src.s2_dual import (
    fit_outcome_models,
    get_mhat_matrix,
    solve_dual_lp,
)

from experiments.real_queue_experiment import (
    make_random_assigner,
    make_s2_assigner,
    make_streams,
    simulate,
    aggregate_one,
    S2_METHODS,
)
from experiments.data_criteo import load_criteo


# === Block A: assigners ======================================================

def make_treat_all_assigner():
    """Always pick the treatment arm (T=1)."""
    def assign(rng, person_idx):
        return 1
    return assign


def make_gf_assigner(model, mu_train, eval_data, tau, B, train_data,
                     cap_buffer=0.92):
    """Deterministic deployment for F.

    Re-solves the dual LP on F's M(X_train) with cap vector
    `cap_buffer * B` (default 0.92 -> sub-cap), then deploys
    `argmax(M_eval - mu_calibrated)`. No peek at the eval distribution.
    """
    B_arr = np.asarray(B, dtype=float)
    B_shrunk = cap_buffer * B_arr

    with torch.no_grad():
        M_train = model(torch.tensor(train_data["X"])).numpy()
    mu_calibrated, _, _, _ = solve_dual_lp(M_train, B_shrunk, verbose=False)

    with torch.no_grad():
        M_eval = model(torch.tensor(eval_data["X"])).numpy()
    a_star = (M_eval - mu_calibrated[None, :]).argmax(axis=1)

    def assign(rng, person_idx):
        return int(a_star[person_idx])

    return assign


# === Block B: train all methods ==============================================

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

    for method in S2_METHODS[:4]:    # linear, lasso, tree, knn (skip dr)
        if method not in {"linear", "lasso", "knn"}:
            continue
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


# === Block C: IPW policy value ===============================================

def precompute_arms(assigner, n_eval, rng_seed=0):
    """Apply assigner to every eval index. RNG is shared across all
    calls so stochastic policies (random) get a single coherent draw."""
    rng = np.random.default_rng(rng_seed)
    return np.fromiter(
        (assigner(rng, i) for i in range(n_eval)),
        dtype=np.int64, count=n_eval,
    )


def ipw_policy_value(arms_eval, eval_data):
    """V_IPW = (1 / N_eval) * sum_i  Y_i * 1{a_eval[i] = T_i} / e_T_i.

    For deterministic policies the indicator is 0/1. e_T_i is the
    propensity of the OBSERVED treatment, already clipped in the
    LaLonde loader.
    """
    Y = eval_data["Y"]
    T = eval_data["T"]
    e_T = eval_data["e_T"]
    matched = (arms_eval == T).astype(np.float64)
    return float((Y * matched / e_T).mean())


# === Block D: plot ===========================================================

def plot_results(agg, methods, out_png, variant_label="full"):
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
    ax.set_ylabel("IPW policy value (eval, P(visit))")
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

    fig.suptitle(f"Criteo Uplift ({variant_label}): F vs S2 (G omitted)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    print(f"[plot] wrote {out_png}")


# === Block E: main ===========================================================

def parse_args():
    p = argparse.ArgumentParser(description="Criteo Uplift N-sweep (real data).")
    p.add_argument("--n-values", type=int, nargs="+",
                   default=[200, 1000, 5000, 20000, 100000])
    p.add_argument("--criteo-subsample", type=int, default=200_000,
                   help="Random subsample size from the source file before "
                        "train/eval splitting.")
    p.add_argument("--criteo-variant", type=str, default="full",
                   choices=["full", "10pct"],
                   help="Which Criteo source file to load. 'full' = ~14M rows, "
                        "'10pct' = ~1.4M rows.")
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
    p.add_argument("--out-csv", type=str, default="results/criteo_sweep.csv")
    p.add_argument("--out-png", type=str, default="results/criteo_sweep.png")
    p.add_argument("--methods", type=str, nargs="+", default=None,
                   help="Subset of method names to keep in plot/table.")
    return p.parse_args()


_PER_ROW_KEYS = {"X", "T", "Y", "e_T", "E", "Y_pot"}


def _subsample(train_data, n, seed):
    """Take a permuted subset of n rows from train_data. Beta / Alpha
    are not per-row; pass them through unchanged."""
    rng = np.random.default_rng(seed)
    N_train = len(train_data["T"])
    n = min(n, N_train)
    idx = rng.permutation(N_train)[:n]
    return {
        k: (v[idx] if k in _PER_ROW_KEYS else v)
        for k, v in train_data.items()
    }


def main():
    args = parse_args()

    # Reproducible torch + numpy state.
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)

    subsample = args.criteo_subsample if args.criteo_subsample > 0 else None
    train_full, eval_data, cfg = load_criteo(
        seed=args.split_seed, subsample=subsample,
        variant=args.criteo_variant,
    )
    cfg["variant"] = args.criteo_variant
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
                assigner, n_eval, rng_seed=hash((method, "ipw")) & 0xFFFFFFFF)
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
    print("Criteo N-SWEEP: mean wait time (served)")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x:10.2f}",
                           "display.width", 200):
        print(wait_pivot.to_string())
    print("=" * 100)
    print("Criteo N-SWEEP: IPW policy value (eval, P(visit) basis)")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x:8.3f}",
                           "display.width", 200):
        print(ipw_pivot.to_string())
    print("=" * 100)
    print("Criteo N-SWEEP: unserved (%)")
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
            f"S2-{m}" for m in ("linear", "lasso", "knn")
        ]
        methods_present = [m for m in method_order
                           if m in agg["method"].values]
        variant_label = {"full": "full ~14M rows",
                         "10pct": "10% sample, ~1.4M rows"}[args.criteo_variant]
        plot_results(agg, methods_present, args.out_png,
                     variant_label=variant_label)


if __name__ == "__main__":
    main()

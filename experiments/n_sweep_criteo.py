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


def _f_arms_and_assigner(model, train_data, eval_data, B, cap_buffer):
    """Return (arms_train, arms_eval, assigner) for F.

    Re-solves the dual LP on F's M(X_train) with cap vector
    `cap_buffer * B` (default 0.92 -> sub-cap), then deploys
    `argmax(M - mu_calibrated)` on both train and eval.
    """
    B_arr = np.asarray(B, dtype=float)
    B_shrunk = cap_buffer * B_arr

    with torch.no_grad():
        M_train = model(torch.tensor(train_data["X"])).numpy()
    mu_calibrated, _, _, _ = solve_dual_lp(M_train, B_shrunk, verbose=False)
    arms_train = (M_train - mu_calibrated[None, :]).argmax(axis=1)

    with torch.no_grad():
        M_eval = model(torch.tensor(eval_data["X"])).numpy()
    arms_eval = (M_eval - mu_calibrated[None, :]).argmax(axis=1)

    def assigner(rng, person_idx):
        return int(arms_eval[person_idx])

    return arms_train, arms_eval, assigner


# === Block B: train all methods ==============================================

def train_policies(train_data, eval_data, T, D, TAU, B, steps, lr, seed,
                   f_tau=0.03, cap_buffer=0.92):
    """Return (policies, arms_train, arms_eval).

    `policies` is method -> (rng, person_idx) -> arm (for the simulator).
    `arms_train` and `arms_eval` are method -> int64 array of policy
    arm choices on train_data["X"] / eval_data["X"], used for the IPW
    policy-value computation.
    """
    policies = {}
    arms_train = {}
    arms_eval = {}
    n_train = len(train_data["T"])
    n_eval = len(eval_data["T"])

    policies["random"] = make_random_assigner(T)
    rng_r = np.random.default_rng(seed * 7919 + 1)
    arms_train["random"] = rng_r.integers(T, size=n_train)
    arms_eval["random"] = rng_r.integers(T, size=n_eval)

    policies["treat_all"] = make_treat_all_assigner()
    arms_train["treat_all"] = np.ones(n_train, dtype=np.int64)
    arms_eval["treat_all"] = np.ones(n_eval, dtype=np.int64)

    f_tau_use = float(f_tau)
    print(f"[train] F (implicit diff)  tau={f_tau_use}  cap_buffer={cap_buffer}")
    model_F, mu_F, _ = train_GF(
        kind="F", train_data=train_data,
        D=D, T=T, tau=f_tau_use, b=B,
        steps=steps, lr=lr, log_every=max(1, steps), seed=seed,
    )
    a_train_F, a_eval_F, assigner_F = _f_arms_and_assigner(
        model_F, train_data, eval_data, B, cap_buffer,
    )
    arms_train["F"] = a_train_F
    arms_eval["F"] = a_eval_F
    policies["F"] = assigner_F

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
        arms_train[f"S2-{method}"] = (
            M_hat_train - mu_hat[None, :]
        ).argmax(axis=1)
        M_hat_eval = get_mhat_matrix(outcome_models, eval_data["X"], T)
        arms_eval[f"S2-{method}"] = (
            M_hat_eval - mu_hat[None, :]
        ).argmax(axis=1)
        policies[f"S2-{method}"] = make_s2_assigner(
            outcome_models, mu_hat, eval_data, T,
        )

    return policies, arms_train, arms_eval


# === Block C: IPW policy value ===============================================

def precompute_arms(assigner, n_eval, rng_seed=0):
    """Apply assigner to every eval index. RNG is shared across all
    calls so stochastic policies (random) get a single coherent draw."""
    rng = np.random.default_rng(rng_seed)
    return np.fromiter(
        (assigner(rng, i) for i in range(n_eval)),
        dtype=np.int64, count=n_eval,
    )


def ipw_policy_value(arms, data):
    """V_IPW = (1 / N) * sum_i  Y_i * 1{arms[i] = T_i} / e_T_i.

    Works for any (arms, data) where the lengths match. Use it for
    both train-side and eval-side IPW values.
    """
    Y = data["Y"]
    T = data["T"]
    e_T = data["e_T"]
    matched = (arms == T).astype(np.float64)
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
    _draw(ax, "ipw_val_mean")
    ax.set_xlabel("N (train subsample size)")
    ax.set_ylabel("IPW value, validation (P(visit))")
    ax.set_xscale("log")
    ax.set_title("IPW (val) vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    ax = axes[1]
    _draw(ax, "ipw_train_mean")
    ax.set_xlabel("N (train subsample size)")
    ax.set_ylabel("IPW value, train (P(visit))")
    ax.set_xscale("log")
    ax.set_title("IPW (train) vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    ax = axes[2]
    _draw(ax, "mean_wait_served_mean")
    ax.set_xlabel("N (train subsample size)")
    ax.set_ylabel("Mean wait time (served)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Mean wait time vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle(f"Criteo Uplift ({variant_label}): F vs S2 (G omitted)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    print(f"[plot] wrote {out_png}")


# === Block E: main ===========================================================

def parse_args():
    p = argparse.ArgumentParser(description="Criteo Uplift N-sweep (real data).")
    # 20 log-spaced N values from 100 to 30000.
    p.add_argument("--n-values", type=int, nargs="+",
                   default=[100, 150, 200, 270, 370, 500, 700, 950, 1300,
                            1800, 2400, 3300, 4500, 6100, 8300, 11300,
                            15400, 20900, 28400, 30000])
    p.add_argument("--criteo-subsample", type=int, default=0,
                   help="Random subsample size from the source file before "
                        "train/eval splitting. 0 = use the whole file.")
    p.add_argument("--criteo-variant", type=str, default="10pct",
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
        policies, arms_train_d, arms_eval_d = train_policies(
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
            arms_train_d = {m: arms_train_d[m] for m in args.methods}
            arms_eval_d  = {m: arms_eval_d[m]  for m in args.methods}
        print(f"[N={N}] train wall: {time.time() - t0:.1f}s, "
              f"methods: {list(policies)}")

        # IPW policy value on TRAIN and on EVAL (no eval peek; arms are
        # the deterministic policy outputs from the trained model).
        ipw_train_per_method = {}
        ipw_eval_per_method = {}
        for method in policies:
            ipw_train_per_method[method] = ipw_policy_value(
                arms_train_d[method], td)
            ipw_eval_per_method[method] = ipw_policy_value(
                arms_eval_d[method], eval_data)
            print(f"  {method:25s}  IPW(train)={ipw_train_per_method[method]:7.4f}  "
                  f"IPW(val)={ipw_eval_per_method[method]:7.4f}")

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
                row["ipw_train"] = ipw_train_per_method[method]
                row["ipw_val"] = ipw_eval_per_method[method]
                rows.append(row)

    df = pd.DataFrame(rows)

    agg = df.groupby(["N", "method"]).agg(
        mean_wait_served_mean=("mean_wait_served", "mean"),
        ipw_val_mean=("ipw_val", "mean"),
        ipw_train_mean=("ipw_train", "mean"),
        frac_unserved_mean=("frac_unserved", "mean"),
    ).reset_index()

    wait_pivot = agg.pivot(index="N", columns="method",
                           values="mean_wait_served_mean").sort_index()
    ipw_val_pivot = agg.pivot(index="N", columns="method",
                              values="ipw_val_mean").sort_index()
    ipw_train_pivot = agg.pivot(index="N", columns="method",
                                values="ipw_train_mean").sort_index()

    print("\n" + "=" * 100)
    print("Criteo N-SWEEP: IPW policy value (validation, P(visit) basis)")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x:8.4f}",
                           "display.width", 200):
        print(ipw_val_pivot.to_string())
    print("=" * 100)
    print("Criteo N-SWEEP: IPW policy value (train, P(visit) basis)")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x:8.4f}",
                           "display.width", 200):
        print(ipw_train_pivot.to_string())
    print("=" * 100)
    print("Criteo N-SWEEP: mean wait time (served)")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x:10.2f}",
                           "display.width", 200):
        print(wait_pivot.to_string())
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

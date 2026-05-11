"""
Single (N, seed) Criteo cell.

Trains F + Alt + Gs + 5 S2 + random + treat_all baselines on a seeded
subsample of size N, runs one queue simulation, and writes per-cell rows
to `results/criteo_cells/cell_N{N}_seed{seed}.csv`.

Designed to be called from a multiprocessing pool (`experiments.sweep_criteo`).
Caches the heavy Criteo load to disk so workers don't repeat the propensity
fit and standardization.

Resumable: if the cell CSV already exists, returns it without re-running.
"""

import os
import pickle
import time
import traceback

import numpy as np
import pandas as pd
import torch

from src.train import train_GF
from src.train_alt import train_alt
from src.s2_dual import (
    fit_outcome_models,
    get_mhat_matrix,
    solve_dual_lp,
)

from experiments.data_criteo import load_criteo
from experiments.n_sweep_criteo import (
    _f_arms_and_assigner,
    _subsample,
    ipw_policy_value,
    make_treat_all_assigner,
)
from experiments.real_queue_experiment import (
    make_random_assigner,
    make_s2_assigner,
    make_streams,
    simulate,
    aggregate_one,
    S2_METHODS,
)


CELL_DIR = "results/criteo_cells"
CRITEO_CACHE = "data/criteo/cached.pkl"


def _ensure_dirs():
    os.makedirs(CELL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CRITEO_CACHE), exist_ok=True)


def _cell_csv_path(N, seed):
    return os.path.join(CELL_DIR, f"cell_N{N}_seed{seed}.csv")


def _failed_path(N, seed):
    return os.path.join(CELL_DIR, f"cell_N{N}_seed{seed}.FAILED")


def load_or_build_criteo_cache(subsample, variant, split_seed):
    """Once per machine: build pickle of (train_full, eval_data, cfg).
    Workers thereafter just `pickle.load` (instant)."""
    if os.path.exists(CRITEO_CACHE):
        with open(CRITEO_CACHE, "rb") as f:
            cache = pickle.load(f)
        key = (subsample, variant, split_seed)
        if cache.get("_key") == key:
            return cache["train_full"], cache["eval_data"], cache["cfg"]

    train_full, eval_data, cfg = load_criteo(
        seed=split_seed, subsample=subsample, variant=variant,
    )
    cfg["variant"] = variant
    payload = {
        "_key": (subsample, variant, split_seed),
        "train_full": train_full,
        "eval_data": eval_data,
        "cfg": cfg,
    }
    tmp = CRITEO_CACHE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CRITEO_CACHE)
    return train_full, eval_data, cfg


def _eval_arms(method, arms_train, arms_eval, train_data, eval_data):
    return {
        "method": method,
        "ipw_train": ipw_policy_value(arms_train, train_data),
        "ipw_val": ipw_policy_value(arms_eval, eval_data),
    }


def _simulate_one(assigner, method, eval_data, T, B, N_sim, lambda_people,
                  max_time_mult, sim_seed):
    people_t, person_idx, T_max, resource_t = make_streams(
        eval_data, N_sim, lambda_people, B, max_time_mult,
        seed=sim_seed * 7 + 13,
    )
    t0 = time.time()
    recs = simulate(
        people_t, person_idx, resource_t, assigner,
        T=T, T_max=T_max, eval_data=eval_data, sim_seed=sim_seed,
    )
    wall = time.time() - t0
    return aggregate_one(recs, method, sim_seed, B, N_sim, wall)


def run_one_cell(
    N, seed,
    subsample=50_000,
    variant="10pct",
    split_seed=0,
    steps=500,
    lr=5e-3,
    f_tau=0.03,
    cap_buffer=0.92,
    alt_inner_freq=5,
    N_sim=1000,
    lambda_people=1.0,
    max_time_mult=1.5,
    force=False,
):
    """Run one (N, seed) cell. Returns list of row-dicts; also writes CSV."""
    _ensure_dirs()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    out_path = _cell_csv_path(N, seed)
    if os.path.exists(out_path) and not force:
        return pd.read_csv(out_path).to_dict(orient="records")

    failed = _failed_path(N, seed)
    if os.path.exists(failed):
        os.remove(failed)

    try:
        train_full, eval_data, cfg = load_or_build_criteo_cache(
            subsample, variant, split_seed,
        )
        T = int(cfg["T"]); D = int(cfg["D"]); B = cfg["B"]

        td = _subsample(train_full, N, seed=split_seed * 1_000_003 + seed * 9973 + N)
        n_train = len(td["T"])
        n_eval = len(eval_data["T"])

        torch.manual_seed(seed)
        np.random.seed(seed)

        rows = []

        # baselines
        random_assigner = make_random_assigner(T)
        rng_r = np.random.default_rng(seed * 7919 + 1)
        a_r_train = rng_r.integers(T, size=n_train)
        a_r_eval = rng_r.integers(T, size=n_eval)
        rrow = _eval_arms("random", a_r_train, a_r_eval, td, eval_data)
        srow = _simulate_one(random_assigner, "random", eval_data,
                             T, B, N_sim, lambda_people, max_time_mult, seed)
        srow.update(rrow)
        rows.append(srow)

        treat_all_assigner = make_treat_all_assigner()
        a_ta_train = np.ones(n_train, dtype=np.int64)
        a_ta_eval = np.ones(n_eval, dtype=np.int64)
        trow = _eval_arms("treat_all", a_ta_train, a_ta_eval, td, eval_data)
        srow = _simulate_one(treat_all_assigner, "treat_all", eval_data,
                             T, B, N_sim, lambda_people, max_time_mult, seed)
        srow.update(trow)
        rows.append(srow)

        # F
        model_F, _, _ = train_GF(
            kind="F", train_data=td, D=D, T=T, tau=f_tau, b=B,
            steps=steps, lr=lr, log_every=max(1, steps), seed=seed,
        )
        a_train_F, a_eval_F, ass_F = _f_arms_and_assigner(
            model_F, td, eval_data, B, cap_buffer,
        )
        row = _eval_arms("F", a_train_F, a_eval_F, td, eval_data)
        srow = _simulate_one(ass_F, "F", eval_data,
                             T, B, N_sim, lambda_people, max_time_mult, seed)
        srow.update(row)
        rows.append(srow)

        # Alt
        model_Alt, _, _ = train_alt(
            train_data=td, D=D, T=T, tau=f_tau, b=B,
            outer_steps=steps, inner_freq=alt_inner_freq, lr=lr,
            log_every=max(1, steps), seed=seed,
        )
        a_train_A, a_eval_A, ass_A = _f_arms_and_assigner(
            model_Alt, td, eval_data, B, cap_buffer,
        )
        row = _eval_arms("Alt", a_train_A, a_eval_A, td, eval_data)
        srow = _simulate_one(ass_A, "Alt", eval_data,
                             T, B, N_sim, lambda_people, max_time_mult, seed)
        srow.update(row)
        rows.append(srow)

        # Gs (scipy + IFT)
        model_Gs, _, _ = train_GF(
            kind="Gs", train_data=td, D=D, T=T, tau=float(cfg["TAU"]), b=B,
            steps=steps, lr=lr, log_every=max(1, steps), seed=seed,
        )
        a_train_G, a_eval_G, ass_G = _f_arms_and_assigner(
            model_Gs, td, eval_data, B, cap_buffer,
        )
        row = _eval_arms("Gs", a_train_G, a_eval_G, td, eval_data)
        srow = _simulate_one(ass_G, "Gs", eval_data,
                             T, B, N_sim, lambda_people, max_time_mult, seed)
        srow.update(row)
        rows.append(srow)

        # S2 methods
        for method in S2_METHODS:
            models = fit_outcome_models(
                X_train=td["X"], T_train=td["T"], Y_train=td["Y"],
                T=T, method=method, E_train=td["E"],
            )
            M_hat_train = get_mhat_matrix(models, td["X"], T)
            mu_hat, _, _, _ = solve_dual_lp(M_hat_train, B, verbose=False)
            a_train_s2 = (M_hat_train - mu_hat[None, :]).argmax(axis=1)
            M_hat_eval = get_mhat_matrix(models, eval_data["X"], T)
            a_eval_s2 = (M_hat_eval - mu_hat[None, :]).argmax(axis=1)
            ass_s2 = make_s2_assigner(models, mu_hat, eval_data, T)

            tag = f"S2-{method}"
            row = _eval_arms(tag, a_train_s2, a_eval_s2, td, eval_data)
            srow = _simulate_one(ass_s2, tag, eval_data,
                                 T, B, N_sim, lambda_people, max_time_mult, seed)
            srow.update(row)
            rows.append(srow)

        # Annotate rows with cell coords.
        for r in rows:
            r["N"] = int(N)
            r["seed"] = int(seed)

        df = pd.DataFrame(rows)
        tmp = out_path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, out_path)
        return rows

    except Exception:
        with open(failed, "w") as f:
            f.write(traceback.format_exc())
        return []


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    rows = run_one_cell(N=args.N, seed=args.seed, steps=args.steps, force=args.force)
    if rows:
        print(pd.DataFrame(rows)[["method", "ipw_train", "ipw_val",
                                  "mean_wait_served", "frac_unserved"]].to_string(index=False))
    else:
        print("CELL FAILED — see", _failed_path(args.N, args.seed))

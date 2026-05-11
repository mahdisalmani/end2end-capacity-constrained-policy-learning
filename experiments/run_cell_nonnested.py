"""
Single (N, seed) cell on the NON-NESTED synthetic DGP from
`experiments/data_nonnested.py`. Mirrors run_cell_synth.py.

Config (hardcoded — non-nested DGP is intentionally a separate experiment):
    T   = 10 arms
    D   = 20 features
    TAU = 0.1
    B   = [1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    sigma_y = 0.3
    propensity_strength = 2.5
    outcome_strength    = 1.0
    treatment_effect_strength = 2.0
    clip_propensity = 0.02
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

from experiments.data_nonnested import generate_data_nonnested
from experiments.n_sweep_criteo import (
    _f_arms_and_assigner,
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


CELL_DIR = "results/nonnested_cells"
EVAL_CACHE = "data/nonnested_eval.pkl"

# Fixed DGP / problem config for the non-nested experiment.
NN_T = 10
NN_D = 30
NN_TAU = 0.1
# Standard 10-arm setting: control cap 1.0, scarce non-control arms 0.1 each
NN_B = np.array([1.0] + [0.1] * 9, dtype=np.float64)
NN_SIGMA_Y = 0.05
NN_PROP_STR = 0.7
NN_OUT_STR = 2.0   # ignored by DGP v2 (overridden internally to 0)
NN_TE_STR = 6.0    # ignored by DGP v2
NN_CLIP = 0.02
NN_HIDDEN = 64     # ignored by DGP v2
NN_N_EVAL = 10_000
NN_EVAL_SEED = 12_345


def _ensure_dirs():
    os.makedirs(CELL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(EVAL_CACHE) or ".", exist_ok=True)


def _cell_csv_path(N, seed):
    return os.path.join(CELL_DIR, f"cell_N{N}_seed{seed}.csv")


def _failed_path(N, seed):
    return os.path.join(CELL_DIR, f"cell_N{N}_seed{seed}.FAILED")


def _gen(N, seed):
    return generate_data_nonnested(
        N=N, seed=seed, d=NN_D, T=NN_T,
        sigma_y=NN_SIGMA_Y,
        propensity_strength=NN_PROP_STR,
        outcome_strength=NN_OUT_STR,
        treatment_effect_strength=NN_TE_STR,
        clip_propensity=NN_CLIP,
        hidden=NN_HIDDEN,
    )


def load_or_build_eval_cache():
    if os.path.exists(EVAL_CACHE):
        with open(EVAL_CACHE, "rb") as f:
            return pickle.load(f)
    eval_data = _gen(N=NN_N_EVAL, seed=NN_EVAL_SEED)
    tmp = EVAL_CACHE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(eval_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, EVAL_CACHE)
    return eval_data


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
    steps=5000,                 # bumped from synth default 500 — shared MLP needs more steps to learn this DGP
    lr=5e-3,
    f_tau=0.03,
    cap_buffer=0.92,
    alt_inner_freq=5,
    N_sim=1000,
    lambda_people=1.0,
    max_time_mult=1.5,
    force=False,
):
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
        T = NN_T; D = NN_D; B = NN_B; TAU = NN_TAU
        eval_data = load_or_build_eval_cache()
        td = _gen(N=N, seed=seed)

        n_train = len(td["T"]); n_eval = len(eval_data["T"])
        torch.manual_seed(seed); np.random.seed(seed)

        rows = []

        random_assigner = make_random_assigner(T)
        rng_r = np.random.default_rng(seed * 7919 + 1)
        a_r_train = rng_r.integers(T, size=n_train)
        a_r_eval = rng_r.integers(T, size=n_eval)
        rrow = _eval_arms("random", a_r_train, a_r_eval, td, eval_data)
        srow = _simulate_one(random_assigner, "random", eval_data,
                             T, B, N_sim, lambda_people, max_time_mult, seed)
        srow.update(rrow); rows.append(srow)

        treat_all_assigner = make_treat_all_assigner()
        a_ta_train = np.ones(n_train, dtype=np.int64)
        a_ta_eval = np.ones(n_eval, dtype=np.int64)
        trow = _eval_arms("treat_all", a_ta_train, a_ta_eval, td, eval_data)
        srow = _simulate_one(treat_all_assigner, "treat_all", eval_data,
                             T, B, N_sim, lambda_people, max_time_mult, seed)
        srow.update(trow); rows.append(srow)

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
        srow.update(row); rows.append(srow)

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
        srow.update(row); rows.append(srow)

        # Gs (scipy + IFT)
        model_Gs, _, _ = train_GF(
            kind="Gs", train_data=td, D=D, T=T, tau=TAU, b=B,
            steps=steps, lr=lr, log_every=max(1, steps), seed=seed,
        )
        a_train_G, a_eval_G, ass_G = _f_arms_and_assigner(
            model_Gs, td, eval_data, B, cap_buffer,
        )
        row = _eval_arms("Gs", a_train_G, a_eval_G, td, eval_data)
        srow = _simulate_one(ass_G, "Gs", eval_data,
                             T, B, N_sim, lambda_people, max_time_mult, seed)
        srow.update(row); rows.append(srow)

        # S2 methods (including mlp)
        for method in list(S2_METHODS) + ["mlp"]:
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
            srow.update(row); rows.append(srow)

        for r in rows:
            r["N"] = int(N); r["seed"] = int(seed)

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
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    rows = run_one_cell(N=args.N, seed=args.seed, steps=args.steps, force=args.force)
    if rows:
        print(pd.DataFrame(rows)[["method", "ipw_train", "ipw_val",
                                  "mean_wait_served", "frac_unserved"]].to_string(index=False))
    else:
        print("CELL FAILED — see", _failed_path(args.N, args.seed))

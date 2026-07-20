"""
Single (N, seed) cell on the NON-NESTED synthetic DGP from
`experiments/data_nonnested.py` (arm 1 carries a sharp 3-D Gaussian peak;
arms 2..9 are weak single-coordinate basis effects). Mirrors
run_cell_synth.py but with its own hardcoded config.

Config (hardcoded — the non-nested DGP is intentionally a separate
experiment; values below are the NN_* constants, the single source of
truth):
    T   = 10 arms
    D   = 30 features
    TAU = 0.1
    B   = [1.0] + [0.1] * 9
    sigma_y = 0.05
    propensity_strength = 0.7
    outcome_strength / treatment_effect_strength / hidden are accepted by
    the DGP signature but ignored by it (the peak DGP fixes its own
    magnitudes internally).

Trains the full method suite (S2 additionally includes "mlp" here) via
`experiments.cell_core.run_cell_generic`; writes per-cell rows to
`results/nonnested_cells/cell_N{N}_seed{seed}.csv`. The eval cache is
keyed on the NN_* knobs, so config edits invalidate it instead of silently
reusing a stale eval split.
"""

import os
import pickle

import numpy as np

from experiments import cell_core
from experiments.common import S2_METHODS, atomic_pickle_dump
from experiments.data_nonnested import generate_data_nonnested


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
NN_OUT_STR = 2.0   # ignored by the peak DGP (overridden internally to 0)
NN_TE_STR = 6.0    # ignored by the peak DGP
NN_CLIP = 0.02
NN_HIDDEN = 64     # ignored by the peak DGP
NN_N_EVAL = 10_000
NN_EVAL_SEED = 12_345


def _cell_csv_path(N, seed):
    return cell_core.cell_csv_path(CELL_DIR, N, seed)


def _failed_path(N, seed):
    return cell_core.failed_path(CELL_DIR, N, seed)


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


def _eval_cache_key():
    return (
        NN_N_EVAL, NN_EVAL_SEED, NN_D, NN_T, NN_SIGMA_Y, NN_PROP_STR,
        NN_OUT_STR, NN_TE_STR, NN_CLIP,
    )


def load_or_build_eval_cache():
    """Fixed eval split (NN_N_EVAL rows at NN_EVAL_SEED), cached to disk and
    keyed on the NN_* knobs so config changes rebuild it."""
    os.makedirs(os.path.dirname(EVAL_CACHE) or ".", exist_ok=True)
    key = _eval_cache_key()
    if os.path.exists(EVAL_CACHE):
        with open(EVAL_CACHE, "rb") as f:
            cache = pickle.load(f)
        if isinstance(cache, dict) and cache.get("_key") == key:
            return cache["eval_data"]
    eval_data = _gen(N=NN_N_EVAL, seed=NN_EVAL_SEED)
    atomic_pickle_dump({"_key": key, "eval_data": eval_data}, EVAL_CACHE)
    return eval_data


def _prepare(N, seed):
    eval_data = load_or_build_eval_cache()
    td = _gen(N=N, seed=seed)
    return td, eval_data, NN_T, NN_D, NN_TAU, NN_B


# Back-compat alias used by the add_s2_mlp_* scripts: one (method, seed)
# queue simulation. Canonical impl: experiments.common.simulate_one_method.
from experiments.common import simulate_one_method as _simulate_one  # noqa: E402,F401


def run_one_cell(
    N, seed,
    steps=1500,
    lr=5e-3,
    f_tau=0.03,
    cap_buffer=0.92,
    alt_inner_freq=5,
    N_sim=1000,
    lambda_people=1.0,
    max_time_mult=1.5,
    force=False,
):
    return cell_core.run_cell_generic(
        CELL_DIR, _prepare, N, seed,
        steps=steps, lr=lr, f_tau=f_tau, cap_buffer=cap_buffer,
        alt_inner_freq=alt_inner_freq, N_sim=N_sim,
        lambda_people=lambda_people, max_time_mult=max_time_mult,
        force=force,
        s2_methods=list(S2_METHODS) + ["mlp"],
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    rows = run_one_cell(N=args.N, seed=args.seed, steps=args.steps, force=args.force)
    cell_core.print_cell_rows(rows, CELL_DIR, args.N, args.seed)

"""
Single (N, seed) cell on the nested synthetic DGP (`src.data.generate_data`,
knobs from `src.config`).

Training data is generated fresh per (N, seed); eval data is one fixed split
(config.N_EVAL rows at config.EVAL_SEED), cached to disk after first
construction. The cache is keyed on the DGP knobs, so changing `src.config`
invalidates it instead of silently reusing a stale eval split.

The method suite, IPW scoring, queue simulation and CSV layout live in
`experiments.cell_core.run_cell_generic`; this file only binds the dataset.
Writes `results/synth_cells/cell_N{N}_seed{seed}.csv`.
"""

import os
import pickle

from src import config
from src.data import generate_data

from experiments import cell_core


CELL_DIR = "results/synth_cells"
EVAL_CACHE = "data/synth_eval.pkl"


def _cell_csv_path(N, seed):
    return cell_core.cell_csv_path(CELL_DIR, N, seed)


def _failed_path(N, seed):
    return cell_core.failed_path(CELL_DIR, N, seed)


def _generate(N, seed, D, T):
    return generate_data(
        N=N, seed=seed, d=D, T=T,
        sigma_y=config.SIGMA_Y,
        propensity_strength=config.PROPENSITY_STRENGTH,
        outcome_strength=config.OUTCOME_STRENGTH,
        treatment_effect_strength=config.TREATMENT_EFFECT_STRENGTH,
        clip_propensity=config.CLIP_PROPENSITY,
    )


def _eval_cache_key():
    return (
        int(config.N_EVAL), int(config.EVAL_SEED), int(config.D),
        int(config.T), float(config.SIGMA_Y),
        float(config.PROPENSITY_STRENGTH), float(config.OUTCOME_STRENGTH),
        float(config.TREATMENT_EFFECT_STRENGTH), float(config.CLIP_PROPENSITY),
    )


def load_or_build_eval_cache():
    """Eval split is fixed (config.N_EVAL, config.EVAL_SEED). Cache to disk,
    keyed on the DGP knobs so config changes rebuild it."""
    os.makedirs(os.path.dirname(EVAL_CACHE) or ".", exist_ok=True)
    key = _eval_cache_key()
    if os.path.exists(EVAL_CACHE):
        with open(EVAL_CACHE, "rb") as f:
            cache = pickle.load(f)
        if isinstance(cache, dict) and cache.get("_key") == key:
            return cache["eval_data"]
    eval_data = _generate(
        N=config.N_EVAL, seed=config.EVAL_SEED,
        D=config.D, T=config.T,
    )
    tmp = EVAL_CACHE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"_key": key, "eval_data": eval_data}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, EVAL_CACHE)
    return eval_data


def _prepare(N, seed):
    T = int(config.T); D = int(config.D); B = config.B; TAU = float(config.TAU)
    eval_data = load_or_build_eval_cache()
    td = _generate(N=N, seed=seed, D=D, T=T)
    return td, eval_data, T, D, TAU, B


# Back-compat alias used by the add_s2_mlp_* scripts: one (method, seed)
# queue simulation. Canonical impl: experiments.common.simulate_one_method.
from experiments.common import simulate_one_method as _simulate_one  # noqa: E402,F401


def run_one_cell(
    N, seed,
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
    return cell_core.run_cell_generic(
        CELL_DIR, _prepare, N, seed,
        steps=steps, lr=lr, f_tau=f_tau, cap_buffer=cap_buffer,
        alt_inner_freq=alt_inner_freq, N_sim=N_sim,
        lambda_people=lambda_people, max_time_mult=max_time_mult,
        force=force,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    rows = run_one_cell(N=args.N, seed=args.seed, steps=args.steps, force=args.force)
    cell_core.print_cell_rows(rows, CELL_DIR, args.N, args.seed)

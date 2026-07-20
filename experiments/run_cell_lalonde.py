"""
Single (N, seed) LaLonde NSW + PSID-1 cell.

Mirrors `experiments/run_cell_criteo.py` but loads the LaLonde dataset.
Trains the full method suite on a seeded subsample of size N via
`experiments.cell_core.run_cell_generic`; writes per-cell CSV to
`results/lalonde_cells/cell_N{N}_seed{seed}.csv`.

Resumable: returns cached rows if the cell CSV already exists.
"""

import os
import pickle

from experiments import cell_core
from experiments.common import atomic_pickle_dump, subsample_rows
from experiments.data_lalonde import load_lalonde


CELL_DIR = "results/lalonde_cells"
LALONDE_CACHE = "data/lalonde/cached.pkl"


def _cell_csv_path(N, seed):
    return cell_core.cell_csv_path(CELL_DIR, N, seed)


def _failed_path(N, seed):
    return cell_core.failed_path(CELL_DIR, N, seed)


def load_or_build_lalonde_cache(train_frac, split_seed):
    """One-shot load. Identical (train_full, eval_data, cfg) reused
    across cells (workers just pickle-load)."""
    os.makedirs(os.path.dirname(LALONDE_CACHE), exist_ok=True)
    if os.path.exists(LALONDE_CACHE):
        with open(LALONDE_CACHE, "rb") as f:
            cache = pickle.load(f)
        if cache.get("_key") == (train_frac, split_seed):
            return cache["train_full"], cache["eval_data"], cache["cfg"]

    train_full, eval_data, cfg = load_lalonde(
        train_frac=train_frac, seed=split_seed,
    )
    payload = {
        "_key": (train_frac, split_seed),
        "train_full": train_full,
        "eval_data": eval_data,
        "cfg": cfg,
    }
    atomic_pickle_dump(payload, LALONDE_CACHE)
    return train_full, eval_data, cfg


# Back-compat alias used by the add_s2_mlp_* scripts: one (method, seed)
# queue simulation. Canonical impl: experiments.common.simulate_one_method.
from experiments.common import simulate_one_method as _simulate_one  # noqa: E402,F401


def run_one_cell(
    N, seed,
    train_frac=0.7,
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
    def prepare(N_, seed_):
        train_full, eval_data, cfg = load_or_build_lalonde_cache(
            train_frac, split_seed,
        )
        T = int(cfg["T"]); D = int(cfg["D"]); B = cfg["B"]
        td = subsample_rows(
            train_full, N_, seed=split_seed * 1_000_003 + seed_ * 9973 + N_,
        )
        return td, eval_data, T, D, float(cfg["TAU"]), B

    return cell_core.run_cell_generic(
        CELL_DIR, prepare, N, seed,
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

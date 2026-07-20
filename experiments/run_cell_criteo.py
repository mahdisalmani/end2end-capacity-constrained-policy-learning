"""
Single (N, seed) Criteo Uplift cell.

Loads (and disk-caches) the Criteo split once, then trains the full method
suite on a seeded subsample of size N via
`experiments.cell_core.run_cell_generic`. Designed to be called from a
multiprocessing pool (`experiments.sweep_criteo`); the pickle cache means
workers skip the heavy propensity fit and standardization.

Writes `results/criteo_cells/cell_N{N}_seed{seed}.csv`. Resumable: if the
cell CSV already exists, returns it without re-running.
"""

import os
import pickle

from experiments import cell_core
from experiments.common import subsample_rows
from experiments.data_criteo import load_criteo


CELL_DIR = "results/criteo_cells"
CRITEO_CACHE = "data/criteo/cached.pkl"


def _cell_csv_path(N, seed):
    return cell_core.cell_csv_path(CELL_DIR, N, seed)


def _failed_path(N, seed):
    return cell_core.failed_path(CELL_DIR, N, seed)


def load_or_build_criteo_cache(subsample, variant, split_seed):
    """Once per machine: build pickle of (train_full, eval_data, cfg).
    Workers thereafter just `pickle.load` (instant)."""
    os.makedirs(os.path.dirname(CRITEO_CACHE), exist_ok=True)
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


# Back-compat alias used by the add_s2_mlp_* scripts: one (method, seed)
# queue simulation. Canonical impl: experiments.common.simulate_one_method.
from experiments.common import simulate_one_method as _simulate_one  # noqa: E402,F401


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

    def prepare(N_, seed_):
        train_full, eval_data, cfg = load_or_build_criteo_cache(
            subsample, variant, split_seed,
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
    # Loader options must be reachable from the CLI: the cache is keyed on
    # them, so a batch runner that can't set them would silently rebuild a
    # different split per job.
    p.add_argument("--variant", type=str, default="10pct",
                   choices=["full", "10pct"])
    p.add_argument("--subsample", type=int, default=50_000)
    p.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    p.add_argument("--f-tau", type=float, default=0.03, dest="f_tau")
    p.add_argument("--cap-buffer", type=float, default=0.92, dest="cap_buffer")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    rows = run_one_cell(N=args.N, seed=args.seed, steps=args.steps,
                        variant=args.variant, subsample=args.subsample,
                        split_seed=args.split_seed, f_tau=args.f_tau,
                        cap_buffer=args.cap_buffer, force=args.force)
    cell_core.print_cell_rows(rows, CELL_DIR, args.N, args.seed)

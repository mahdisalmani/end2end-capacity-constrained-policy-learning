"""
Single (N, seed) cell on Diabetes 130-US Hospitals (real, ~70k patients,
3 arms: no HbA1c test / test / test + med change — see
`experiments/data_diabetes.py`). Y_pot unknown (real data).

Writes `results/diabetes_cells/cell_N{N}_seed{seed}.csv`.
"""

import pickle

from experiments import cell_core
from experiments.common import atomic_pickle_dump, subsample_rows
from experiments.data_diabetes import load_diabetes
import os


CELL_DIR = "results/diabetes_cells"
CACHE = "data/uci/diabetes_split.pkl"


def _cell_csv_path(N, seed):
    return cell_core.cell_csv_path(CELL_DIR, N, seed)


def _failed_path(N, seed):
    return cell_core.failed_path(CELL_DIR, N, seed)


def load_or_build_cache(train_frac, split_seed):
    key = (train_frac, split_seed)
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            c = pickle.load(f)
        if c.get("_key") == key:
            return c["train_full"], c["eval_data"], c["cfg"]
    train_full, eval_data, cfg = load_diabetes(train_frac=train_frac,
                                               seed=split_seed)
    atomic_pickle_dump({"_key": key, "train_full": train_full,
                        "eval_data": eval_data, "cfg": cfg}, CACHE)
    return train_full, eval_data, cfg


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
        train_full, eval_data, cfg = load_or_build_cache(train_frac, split_seed)
        td = subsample_rows(
            train_full, N_, seed=split_seed * 1_000_003 + seed_ * 9973 + N_,
        )
        return (td, eval_data, int(cfg["T"]), int(cfg["D"]),
                float(cfg["TAU"]), cfg["B"])

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
    p.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    rows = run_one_cell(N=args.N, seed=args.seed, steps=args.steps,
                        split_seed=args.split_seed, force=args.force)
    cell_core.print_cell_rows(rows, CELL_DIR, args.N, args.seed)

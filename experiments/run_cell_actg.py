"""
Single (N, seed) cell on ACTG Study 175 (real 4-arm randomized HIV trial —
see `experiments/data_actg.py`). Y_pot is unknown (real data), so oracle
metrics are NaN; IPW is unbiased by randomization.

Writes `results/actg_cells/cell_N{N}_seed{seed}.csv`.
"""

from experiments import cell_core
from experiments.common import subsample_rows
from experiments.data_actg import load_actg


CELL_DIR = "results/actg_cells"


def _cell_csv_path(N, seed):
    return cell_core.cell_csv_path(CELL_DIR, N, seed)


def _failed_path(N, seed):
    return cell_core.failed_path(CELL_DIR, N, seed)


def run_one_cell(
    N, seed,
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
        train_full, eval_data, cfg = load_actg(seed=split_seed)
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

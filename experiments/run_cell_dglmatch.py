"""Single (N, seed) cell on the ported Dual-Guided Learning matching DGP
(see `experiments/data_dglmatch.py` for what is theirs and what we add).

Ground truth is available: the generator emits every location's outcome
probability, so deployed value is scored against known potential outcomes.

The PtO suite includes `mlp`, the capacity-matched per-arm estimator (same
trunk shape, steps and lr as the end-to-end score model), so nothing here can
be attributed to function-class mismatch.

Writes `results/dglmatch_cells/cell_N{N}_seed{seed}.csv`.
"""

from experiments import cell_core
from experiments.common import S2_METHODS
from experiments.data_dglmatch import generate_dglmatch

CELL_DIR = "results/dglmatch_cells"
CELL_DIR_TIGHT = "results/dglmatch_tight_cells"


def cell_dir_for(outside):
    return CELL_DIR if outside else CELL_DIR_TIGHT


def run_one_cell(
    N, seed,
    outside=True,
    steps=800,
    lr=5e-3,
    f_tau=0.03,
    cap_buffer=None,
    alt_inner_freq=5,
    N_sim=1000,
    lambda_people=1.0,
    max_time_mult=1.5,
    force=False,
):
    # With their tight capacities (sum b = 1) any sub-capacity buffer makes the
    # deployment LP infeasible, so the buffer is only available when the
    # outside arm supplies slack.
    if cap_buffer is None:
        cap_buffer = 0.92 if outside else 1.0

    def prepare(N_, seed_):
        td, eval_data, cfg = generate_dglmatch(N_, seed_, outside=outside)
        return (td, eval_data, int(cfg["T"]), int(cfg["D"]),
                float(cfg["TAU"]), cfg["B"])

    return cell_core.run_cell_generic(
        cell_dir_for(outside), prepare, N, seed,
        steps=steps, lr=lr, f_tau=f_tau, cap_buffer=cap_buffer,
        alt_inner_freq=alt_inner_freq, N_sim=N_sim,
        lambda_people=lambda_people, max_time_mult=max_time_mult,
        force=force,
        s2_methods=list(S2_METHODS) + ["mlp"],
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--tight", action="store_true",
                   help="their exact 3-arm structure, sum b = 1, no buffer")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    rows = run_one_cell(N=args.N, seed=args.seed, steps=args.steps,
                        outside=not args.tight, force=args.force)
    cell_core.print_cell_rows(rows, cell_dir_for(not args.tight),
                              args.N, args.seed)

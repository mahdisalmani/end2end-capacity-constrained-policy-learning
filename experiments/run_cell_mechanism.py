"""
Single (N, seed) cell on the mechanism DGP (dose-matching on a shared
severity score under a prognosis-nuisance dial — see
`experiments/data_mechanism.py`). Ground truth available (synthetic).

The S2 suite here includes `mlp`, the capacity-matched per-arm estimator
(same trunk shape, steps and lr as the end-to-end score model), so the
comparison cannot be attributed to function-class mismatch.

Writes `results/mechanism_cells/amp{amp:g}/cell_N{N}_seed{seed}.csv`.
"""

from experiments import cell_core
from experiments.common import S2_METHODS
from experiments.data_mechanism import AMP_DEFAULT, generate_mechanism


def cell_dir_for(amp):
    return f"results/mechanism_cells/amp{amp:g}"


def run_one_cell(
    N, seed,
    amp=AMP_DEFAULT,
    steps=800,
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
        td, eval_data, cfg = generate_mechanism(N_, seed_, amp=amp)
        return (td, eval_data, int(cfg["T"]), int(cfg["D"]),
                float(cfg["TAU"]), cfg["B"])

    return cell_core.run_cell_generic(
        cell_dir_for(amp), prepare, N, seed,
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
    p.add_argument("--amp", type=float, default=AMP_DEFAULT)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    rows = run_one_cell(N=args.N, seed=args.seed, amp=args.amp,
                        steps=args.steps, force=args.force)
    cell_core.print_cell_rows(rows, cell_dir_for(args.amp),
                              args.N, args.seed)

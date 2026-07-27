"""Buffer-sweep ablation: can a tighter deployment buffer substitute for
decision-aware training?

For one (dataset, seed) at deployment scale, train F and G once and fit
PtO-mlp and PtO-tree once, then deploy EVERY method at a grid of capacity
buffers (the LP's caps scaled by `buffer` on the scarce arms) and push
each deployment through the queue at the TRUE caps. No retraining across
buffers: the buffer is purely a deployment-time choice, which is exactly
the point of the ablation.

Writes results/buffer_cells/{dataset}_N{N}_seed{seed}.csv.

Usage:
    PYTHONPATH=. python3 scripts/buffer_sweep.py --dataset mechanism --seed 0
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (                     # noqa: E402
    arms_and_assigner_from_model,
    eval_arms_row,
    make_s2_assigner,
    simulate_one_method,
)
from experiments.prepare_registry import DEPLOY_N, get_prepare  # noqa: E402
from src.s2_dual import fit_outcome_models, get_mhat_matrix, solve_dual_lp  # noqa: E402
from src.train import train_GF                        # noqa: E402

BUFFERS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.96, 1.00]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    N = DEPLOY_N[args.dataset]
    prepare, _, steps, sim_kwargs = get_prepare(args.dataset, N, args.seed)
    td, ev, T, D, TAU, B = prepare(N, args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mF, _, _ = train_GF(kind="F", train_data=td, D=D, T=T, tau=0.03, b=B,
                        steps=steps, lr=5e-3, log_every=steps, seed=args.seed)
    mG, _, _ = train_GF(kind="Gs", train_data=td, D=D, T=T, tau=TAU, b=B,
                        steps=steps, lr=5e-3, log_every=steps, seed=args.seed)
    pto = {}
    for meth in ("mlp", "tree"):
        pto[meth] = fit_outcome_models(
            X_train=td["X"], T_train=td["T"], Y_train=td["Y"],
            T=T, method=meth, E_train=td["E"], mlp_steps=steps)

    rows = []
    for buf in BUFFERS:
        B_e2e = np.asarray(B, dtype=float).copy()
        B_e2e[1:] = B_e2e[1:] * buf          # buffer the scarce arms only:
        # shrinking b_0 as well can push sum(b) below 1, making the dual
        # LP genuinely unbounded on tight two-arm instances.
        for name, model in (("F", mF), ("G", mG)):
            arms_tr, arms_ev, asg = arms_and_assigner_from_model(
                model, td, ev, B_e2e, 1.0)
            r = simulate_one_method(asg, name, ev, T, B,
                                    sim_kwargs["N_sim"],
                                    sim_kwargs["lambda_people"],
                                    sim_kwargs["max_time_mult"],
                                    sim_seed=args.seed)
            r.update(eval_arms_row(name, arms_tr, arms_ev, td, ev))
            r.update(buffer=buf, N=N, seed=args.seed)
            rows.append(r)
        for meth, models in pto.items():
            B_buf = np.asarray(B, dtype=float).copy()
            B_buf[1:] = B_buf[1:] * buf
            Mh_tr = get_mhat_matrix(models, td["X"], T)
            mu_hat, _, _, _ = solve_dual_lp(Mh_tr, B_buf, verbose=False)
            arms_tr = (Mh_tr - mu_hat[None, :]).argmax(axis=1)
            Mh_ev = get_mhat_matrix(models, ev["X"], T)
            arms_ev = (Mh_ev - mu_hat[None, :]).argmax(axis=1)
            asg = make_s2_assigner(models, mu_hat, ev, T)
            name = f"PtO-{meth}"
            r = simulate_one_method(asg, name, ev, T, B,
                                    sim_kwargs["N_sim"],
                                    sim_kwargs["lambda_people"],
                                    sim_kwargs["max_time_mult"],
                                    sim_seed=args.seed)
            r.update(eval_arms_row(name, arms_tr, arms_ev, td, ev))
            r.update(buffer=buf, N=N, seed=args.seed)
            rows.append(r)

    os.makedirs("results/buffer_cells", exist_ok=True)
    out = f"results/buffer_cells/{args.dataset}_N{N}_seed{args.seed}.csv"
    tmp = out + f".tmp{os.getpid()}"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, out)
    print(f"[buffer] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

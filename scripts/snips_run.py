"""SNIPS ablation: train F with the self-normalised IPW outer objective
(one-line change to the outer loss) at deployment scale, deploy through
the standard pipeline, and record the same row schema as the sweeps so
the result pairs per-seed with the existing F rows.

Writes results/snips_cells/{dataset}_seed{seed}.csv.

Usage:
    PYTHONPATH=. python3 scripts/snips_run.py --dataset adultsemi --seed 0
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
    simulate_one_method,
)
from experiments.prepare_registry import DEPLOY_N, get_prepare  # noqa: E402
from src.train import train_GF                        # noqa: E402


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

    model, _, _ = train_GF(kind="F", train_data=td, D=D, T=T, tau=0.03, b=B,
                           steps=steps, lr=5e-3, log_every=steps,
                           seed=args.seed, outer="snips")
    arms_tr, arms_ev, asg = arms_and_assigner_from_model(model, td, ev, B, 0.92)
    r = simulate_one_method(asg, "F-SNIPS", ev, T, B,
                            sim_kwargs["N_sim"], sim_kwargs["lambda_people"],
                            sim_kwargs["max_time_mult"], sim_seed=args.seed)
    r.update(eval_arms_row("F-SNIPS", arms_tr, arms_ev, td, ev))
    r.update(N=N, seed=args.seed)

    os.makedirs("results/snips_cells", exist_ok=True)
    out = f"results/snips_cells/{args.dataset}_seed{args.seed}.csv"
    tmp = out + f".tmp{os.getpid()}"
    pd.DataFrame([r]).to_csv(tmp, index=False)
    os.replace(tmp, out)
    print(f"[snips] {args.dataset} seed={args.seed}: "
          f"ipw_val={r['ipw_val']:.4f} alloc_1={r.get('alloc_1', float('nan')):.3f}")


if __name__ == "__main__":
    main()

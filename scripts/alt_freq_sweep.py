"""Is the implicit gradient's advantage just a matter of how often the duals
are refreshed?

Alt holds the prices fixed for `inner_freq` ascent steps, then re-solves them,
and never differentiates through them. F re-solves them at every step AND
differentiates through them. So the two differ in two ways at once: how stale
the prices are, and whether the gradient carries the term dmu/dtheta.

This sweep separates those. Refresh frequency is swept down to 1, where Alt
re-solves the prices at EVERY outer step -- the prices are then exactly as
fresh as F's, the compute saving is gone, and the only remaining difference is
the missing gradient term. If Alt closes the gap as freq -> 1, the advantage
was staleness and their shortcut just needs tuning. If a gap survives at
freq = 1, it is the implicit gradient itself.

Everything else is held fixed: same data, same seed, same initialisation, same
step budget, same deployment buffer, same queueing streams.

Writes results/altfreq_cells/{dataset}_N{N}_seed{seed}.csv

Usage:
    PYTHONPATH=. python3 scripts/alt_freq_sweep.py --dataset dglmatch --seed 0
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
from src.train_alt import train_alt                   # noqa: E402

FREQS = [1, 2, 5, 10, 20, 50]
F_TAU = 0.03
CAP_BUFFER = 0.92


def deploy(name, model, td, ev, T, B, sim_kwargs, seed, extra):
    arms_tr, arms_ev, asg = arms_and_assigner_from_model(
        model, td, ev, B, CAP_BUFFER)
    r = simulate_one_method(asg, name, ev, T, B,
                            sim_kwargs["N_sim"], sim_kwargs["lambda_people"],
                            sim_kwargs["max_time_mult"], sim_seed=seed)
    r.update(eval_arms_row(name, arms_tr, arms_ev, td, ev))
    r.update(extra)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dglmatch")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    N = DEPLOY_N[args.dataset]
    prepare, _, steps, sim_kwargs = get_prepare(args.dataset, N, args.seed)
    td, ev, T, D, TAU, B = prepare(N, args.seed)

    rows = []

    # reference: full implicit gradient, prices re-solved every step
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    mF, _, _ = train_GF(kind="F", train_data=td, D=D, T=T, tau=F_TAU, b=B,
                        steps=steps, lr=5e-3, log_every=steps, seed=args.seed)
    rows.append(deploy("F", mF, td, ev, T, B, sim_kwargs, args.seed,
                       dict(method_family="implicit", inner_freq=1,
                            dual_solves=steps, N=N, seed=args.seed)))

    # the shortcut, at every refresh frequency
    for freq in FREQS:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        mA, _, _ = train_alt(train_data=td, D=D, T=T, tau=F_TAU, b=B,
                             outer_steps=steps, inner_freq=freq, lr=5e-3,
                             log_every=steps, seed=args.seed)
        rows.append(deploy(f"Alt-f{freq}", mA, td, ev, T, B, sim_kwargs,
                           args.seed,
                           dict(method_family="shortcut", inner_freq=freq,
                                dual_solves=int(np.ceil(steps / freq)),
                                N=N, seed=args.seed)))

    os.makedirs("results/altfreq_cells", exist_ok=True)
    out = f"results/altfreq_cells/{args.dataset}_N{N}_seed{args.seed}.csv"
    tmp = out + f".tmp{os.getpid()}"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, out)
    print(f"[altfreq] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

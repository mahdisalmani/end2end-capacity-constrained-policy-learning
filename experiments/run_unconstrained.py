"""The missing baseline: an UNCONSTRAINED off-policy IPW maximiser
("IPW-unc"), trained with prices frozen at zero and deployed as a pure
argmax, then pushed through the same queue as every other method.

Implementation note that makes this exact rather than approximate: we run
the alternating trainer with every capacity set to 1. The inner gradient
is b_t - mean_i sigma_{t,i} >= 0 whenever b_t = 1, so the inner minimiser
is mu* = 0 at every refresh: the training loop provably reduces to plain
IPW ascent on the softmax policy. Deployment uses the same LP layer with
unit caps and no buffer, which reduces to argmax_t m_t(x). The queue
simulation, of course, runs at the TRUE caps.

Appends a row `IPW-unc` to the existing cell CSV (idempotent).

Usage:
    python3 -m experiments.run_unconstrained --dataset mechanism --N 32000 --seed 0
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from experiments import cell_core
from experiments.common import (
    arms_and_assigner_from_model,
    eval_arms_row,
    simulate_one_method,
)
from experiments.prepare_registry import get_prepare
from src.train_alt import train_alt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    prepare, cell_dir, steps, sim_kwargs = get_prepare(
        args.dataset, args.N, args.seed)
    out_path = cell_core.cell_csv_path(cell_dir, args.N, args.seed)
    if not os.path.exists(out_path):
        print(f"[unc] no cell at {out_path}; skipping")
        return
    df = pd.read_csv(out_path)
    if "IPW-unc" in set(df["method"]):
        print("[unc] already present")
        return

    td, ev, T, D, TAU, B = prepare(args.N, args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    b_ones = np.ones(T)
    model, _, _ = train_alt(
        train_data=td, D=D, T=T, tau=0.03, b=b_ones,
        outer_steps=steps, inner_freq=5, lr=5e-3,
        log_every=max(1, steps), seed=args.seed,
    )
    arms_tr, arms_ev, assigner = arms_and_assigner_from_model(
        model, td, ev, b_ones, 1.0)

    srow = simulate_one_method(assigner, "IPW-unc", ev, T, B,
                               sim_kwargs["N_sim"],
                               sim_kwargs["lambda_people"],
                               sim_kwargs["max_time_mult"],
                               sim_seed=args.seed)
    srow.update(eval_arms_row("IPW-unc", arms_tr, arms_ev, td, ev))
    srow["N"] = int(args.N)
    srow["seed"] = int(args.seed)

    out = pd.concat([df, pd.DataFrame([srow])], ignore_index=True)
    tmp = out_path + f".tmp{os.getpid()}"
    out.to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    print(f"[unc] {args.dataset} N={args.N} seed={args.seed}: "
          f"alloc={[round(srow.get(f'alloc_{t}', 0), 3) for t in range(T)]} "
          f"u={srow['frac_unserved']:.3f} W={srow['mean_wait_all']:.1f}")


if __name__ == "__main__":
    main()

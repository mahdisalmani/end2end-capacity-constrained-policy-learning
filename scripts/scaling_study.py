"""Wall-clock scaling of the four training paths in |T| (number of treatments).

The Final Report notes that reviewer feedback focused on "the scalability of
the implementation with respect to the number of treatments". This measures it
directly: for each T, time a fixed number of training steps for

    G   — convex dual through a CVXPYLayer (diffcp implicit differentiation)
    Gs  — same objective, scipy L-BFGS-B forward + IFT backward
    F   — non-convex objective, scipy L-BFGS-B forward + IFT backward
    Alt — block-coordinate, inner solve every `inner_freq` steps, no IFT

and report seconds per outer step. The IFT paths materialize a |T|x|T| Hessian
row by row and solve one dense (|T|+|A|) system per step, so their cost grows
mildly in T; the CVXPYLayer path additionally re-derives a cone program whose
size grows with N*T.

Usage:
    python scripts/scaling_study.py --out results/scaling.json
    python scripts/scaling_study.py --t-values 2 4 6 8 --steps 10
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import generate_data
from src.inner_G import initialize_G_layer
from src.train import train_GF
from src.train_alt import train_alt


def capacities(T):
    """Control unconstrained; the T-1 scarce arms share a 1.5 budget so the
    problem stays feasible and comparably tight as T grows."""
    return np.array([1.0] + [1.5 / (T - 1)] * (T - 1))


def time_one(kind, train_data, D, T, tau, b, steps, seed):
    t0 = time.time()
    if kind == "Alt":
        train_alt(train_data=train_data, D=D, T=T, tau=tau, b=b,
                  outer_steps=steps, inner_freq=5, lr=5e-3,
                  log_every=10 ** 9, seed=seed)
    else:
        train_GF(kind=kind, train_data=train_data, D=D, T=T, tau=tau, b=b,
                 steps=steps, lr=5e-3, log_every=10 ** 9, seed=seed)
    return (time.time() - t0) / steps


def main():
    p = argparse.ArgumentParser(description="Scaling in |T| of the four paths.")
    p.add_argument("--t-values", type=int, nargs="+",
                   default=[2, 4, 6, 8, 10, 15, 20])
    p.add_argument("--N", type=int, default=400)
    p.add_argument("--D", type=int, default=20)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--methods", type=str, nargs="+",
                   default=["G", "Gs", "F", "Alt"])
    p.add_argument("--out", type=str, default="results/scaling.json")
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)

    out = {"T": [], "N": args.N, "D": args.D, "steps": args.steps,
           "tau": args.tau, "threads": args.threads,
           "per_step": {m: [] for m in args.methods}}

    for T in args.t_values:
        b = capacities(T)
        train_data = generate_data(
            N=args.N, seed=args.seed, d=args.D, T=T, sigma_y=0.05,
            propensity_strength=0.7, outcome_strength=2.0,
            treatment_effect_strength=6.0, clip_propensity=0.02,
        )
        out["T"].append(T)
        for m in args.methods:
            if m == "G":
                initialize_G_layer(N=args.N, T=T, tau=args.tau, b=b)
            try:
                s = time_one(m, train_data, args.D, T, args.tau, b,
                             args.steps, args.seed)
            except Exception as e:  # noqa: BLE001
                print(f"[scaling] T={T} {m} FAILED: {type(e).__name__}: {e}")
                s = None
            out["per_step"][m].append(s)
            print(f"[scaling] T={T:3d}  {m:3s}  {s if s is None else f'{s:.4f}'} s/step",
                  flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[scaling] wrote {args.out}")


if __name__ == "__main__":
    main()

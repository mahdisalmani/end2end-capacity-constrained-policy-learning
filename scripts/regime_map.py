"""Regime map: WHEN does end-to-end training beat two-stage?

One config = (te_nonlinearity, cap_scale, seed) on the Adult semi-synthetic
problem at fixed N. The two dials span the axes the week's experiments
suggested matter:

  te_nonlinearity (lam)  0 -> the outcome surface is linear and S2-linear /
                         lasso are correctly specified; 1 -> effects live in
                         Gaussian bumps invisible to linear models.
  cap_scale              scales every scarce cap; small -> constraints tight
                         and binding, large -> effectively unconstrained.

Hypothesis (stated before running): end-to-end wins in the tight+nonlinear
corner and ties or loses in the slack or linear corners — which would also
retro-explain LaLonde (slack corner) and Criteo (2 arms, mild surface).
Whatever the map shows is what gets reported.

Scored on ORACLE value of the deployed argmax policy on held-out data
(ground truth is known here — the point of semi-synthetic), plus the raw
policy's capacity violation.

Usage (one cell; the SLURM driver sweeps the grid):
    python scripts/regime_map.py --lam 0.75 --cap-scale 0.5 --seed 0
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train_GF
from src.train_alt import train_alt
from experiments.common import (
    arms_and_assigner_from_model,
    s2_arms_and_assigner,
    subsample_rows,
)
from experiments.data_adult_semi import load_adult_semi

OUT_DIR = "results/regime_cells"
N_TRAIN = 2000
STEPS = 400
F_TAU = 0.03
CAP_BUFFER = 0.92
S2_SET = ("linear", "lasso", "tree")


def oracle_value(arms, eval_data):
    return float(eval_data["Y_pot"][np.arange(len(arms)), arms].mean())


def violation(arms, B):
    alloc = np.bincount(arms, minlength=len(B)) / len(arms)
    return float(np.maximum(alloc - np.asarray(B), 0.0).max())


def main():
    p = argparse.ArgumentParser(description="One regime-map cell.")
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--cap-scale", type=float, required=True, dest="cap_scale")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--sigma", type=float, default=0.5,
                   help="outcome noise sd (SNR dial: effects have scale ~1-4)")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    sig_tag = "" if args.sigma == 0.5 else f"_sig{args.sigma:g}"
    out = os.path.join(
        OUT_DIR,
        f"cell_lam{args.lam:g}_cap{args.cap_scale:g}{sig_tag}_s{args.seed}.csv")
    if os.path.exists(out) and not args.force:
        print(f"[regime] cached: {out}")
        return

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

    train_full, eval_data, cfg = load_adult_semi(
        te_nonlinearity=args.lam, cap_scale=args.cap_scale,
        sigma_y=args.sigma, seed=0)
    T, D, TAU, B = (int(cfg["T"]), int(cfg["D"]),
                    float(cfg["TAU"]), cfg["B"])
    td = subsample_rows(train_full, N_TRAIN, seed=9973 * args.seed + 17)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = []

    def record(method, arms_eval):
        rows.append({
            "lam": args.lam, "cap_scale": args.cap_scale,
            "sigma": args.sigma, "seed": args.seed,
            "method": method,
            "oracle_val": oracle_value(arms_eval, eval_data),
            "violation": violation(arms_eval, B),
        })

    model_F, _, _ = train_GF("F", td, D=D, T=T, tau=F_TAU, b=B,
                             steps=args.steps, lr=5e-3,
                             log_every=10 ** 9, seed=args.seed)
    _, a, _ = arms_and_assigner_from_model(model_F, td, eval_data, B, CAP_BUFFER)
    record("F", a)

    model_G, _, _ = train_GF("Gs", td, D=D, T=T, tau=TAU, b=B,
                             steps=args.steps, lr=5e-3,
                             log_every=10 ** 9, seed=args.seed)
    _, a, _ = arms_and_assigner_from_model(model_G, td, eval_data, B, CAP_BUFFER)
    record("Gs", a)

    model_A, _, _ = train_alt(td, D=D, T=T, tau=F_TAU, b=B,
                              outer_steps=args.steps, inner_freq=5, lr=5e-3,
                              log_every=10 ** 9, seed=args.seed)
    _, a, _ = arms_and_assigner_from_model(model_A, td, eval_data, B, CAP_BUFFER)
    record("Alt", a)

    for m in S2_SET:
        _, a, _ = s2_arms_and_assigner(td, eval_data, T, B, m)
        record(f"S2-{m}", a)

    # references
    rng = np.random.default_rng(args.seed)
    record("random", rng.integers(T, size=len(eval_data["T"])))
    record("oracle_argmax_no_cap", eval_data["Y_pot"].argmax(1))

    import pandas as pd
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[regime] wrote {out}")
    for r in rows:
        print(f"  {r['method']:22s} oracle={r['oracle_val']:.3f} "
              f"viol={r['violation']:.3f}")


if __name__ == "__main__":
    main()

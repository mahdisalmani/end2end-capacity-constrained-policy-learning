"""
Generic (N, seed) cell worker shared by all queue-sim experiments.

A "cell" trains the full method suite — random, treat_all, F, Alt, Gs and
the S2 baselines — on one training set, computes train/val IPW values for
each method's deterministic policy, runs one queue simulation per method,
and writes the rows to `{cell_dir}/cell_N{N}_seed{seed}.csv`.

The dataset-specific wrappers (`run_cell_synth`, `run_cell_criteo`,
`run_cell_lalonde`, `run_cell_nonnested`) supply only a `prepare(N, seed)`
callable returning `(train_data, eval_data, T, D, TAU, B)` plus their own
defaults; everything else lives here, once.

Cells are resumable (an existing CSV is returned without re-running) and
failure-isolated (a traceback lands in `cell_N{N}_seed{seed}.FAILED` and
the cell returns []).
"""

import os
import time
import traceback

import numpy as np
import pandas as pd
import torch

from src.train import train_GF
from src.train_alt import train_alt

from experiments.common import (
    S2_METHODS,
    arms_and_assigner_from_model,
    eval_arms_row,
    make_random_assigner,
    make_treat_all_assigner,
    s2_arms_and_assigner,
    simulate_one_method,
)


def cell_csv_path(cell_dir, N, seed):
    return os.path.join(cell_dir, f"cell_N{N}_seed{seed}.csv")


def failed_path(cell_dir, N, seed):
    return os.path.join(cell_dir, f"cell_N{N}_seed{seed}.FAILED")


def run_cell_generic(
    cell_dir,
    prepare,
    N, seed,
    steps=500,
    lr=5e-3,
    f_tau=0.03,
    cap_buffer=0.92,
    alt_inner_freq=5,
    N_sim=1000,
    lambda_people=1.0,
    max_time_mult=1.5,
    force=False,
    s2_methods=None,
):
    """Run one (N, seed) cell. Returns list of row-dicts; also writes CSV.

    prepare(N, seed) -> (train_data, eval_data, T, D, TAU, B). It runs
    BEFORE the global torch/numpy seeding so cached loads don't perturb
    the method RNG stream.
    """
    os.makedirs(cell_dir, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    out_path = cell_csv_path(cell_dir, N, seed)
    if os.path.exists(out_path) and not force:
        return pd.read_csv(out_path).to_dict(orient="records")

    failed = failed_path(cell_dir, N, seed)
    if os.path.exists(failed):
        os.remove(failed)

    if s2_methods is None:
        s2_methods = list(S2_METHODS)

    try:
        td, eval_data, T, D, TAU, B = prepare(N, seed)

        n_train = len(td["T"])
        n_eval = len(eval_data["T"])

        torch.manual_seed(seed)
        np.random.seed(seed)

        rows = []

        def add(method, arms_train, arms_eval, assigner):
            srow = simulate_one_method(
                assigner, method, eval_data, T, B, N_sim,
                lambda_people, max_time_mult, sim_seed=seed,
            )
            srow.update(eval_arms_row(method, arms_train, arms_eval,
                                      td, eval_data))
            rows.append(srow)

        # baselines
        rng_r = np.random.default_rng(seed * 7919 + 1)
        add("random",
            rng_r.integers(T, size=n_train),
            rng_r.integers(T, size=n_eval),
            make_random_assigner(T))

        add("treat_all",
            np.ones(n_train, dtype=np.int64),
            np.ones(n_eval, dtype=np.int64),
            make_treat_all_assigner())

        # F (end-to-end, non-convex inner objective)
        model_F, _, _ = train_GF(
            kind="F", train_data=td, D=D, T=T, tau=f_tau, b=B,
            steps=steps, lr=lr, log_every=max(1, steps), seed=seed,
        )
        add("F", *arms_and_assigner_from_model(model_F, td, eval_data,
                                               B, cap_buffer))

        # Alt (block-coordinate, no implicit differentiation)
        model_Alt, _, _ = train_alt(
            train_data=td, D=D, T=T, tau=f_tau, b=B,
            outer_steps=steps, inner_freq=alt_inner_freq, lr=lr,
            log_every=max(1, steps), seed=seed,
        )
        add("Alt", *arms_and_assigner_from_model(model_Alt, td, eval_data,
                                                 B, cap_buffer))

        # Gs (end-to-end, convex dual via scipy + IFT)
        model_Gs, _, _ = train_GF(
            kind="Gs", train_data=td, D=D, T=T, tau=TAU, b=B,
            steps=steps, lr=lr, log_every=max(1, steps), seed=seed,
        )
        add("Gs", *arms_and_assigner_from_model(model_Gs, td, eval_data,
                                                B, cap_buffer))

        # S2 two-stage baselines
        for method in s2_methods:
            add(f"S2-{method}",
                *s2_arms_and_assigner(td, eval_data, T, B, method))

        # Annotate rows with cell coords.
        for r in rows:
            r["N"] = int(N)
            r["seed"] = int(seed)

        df = pd.DataFrame(rows)
        tmp = out_path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, out_path)
        return rows

    except Exception:
        with open(failed, "w") as f:
            f.write(traceback.format_exc())
        return []


def print_cell_rows(rows, cell_dir, N, seed):
    """Human-readable one-cell summary for the __main__ blocks."""
    if rows:
        cols = ["method", "ipw_train", "ipw_val",
                "mean_wait_served", "frac_unserved"]
        print(pd.DataFrame(rows)[cols].to_string(index=False))
    else:
        print("CELL FAILED — see", failed_path(cell_dir, N, seed))

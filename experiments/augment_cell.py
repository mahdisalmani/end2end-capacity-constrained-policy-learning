"""
Add missing S2 baselines to one EXISTING cell, without re-training the
end-to-end methods. Used to extend every sweep with `mlp` — the per-arm
estimator whose trunk shape, steps and lr match the end-to-end score
model — so no comparison can be attributed to function-class mismatch.

The dataset's own `run_one_cell` builds its `prepare` closure (subsample
seed formulas, caches, split seeds); we capture that closure by stubbing
`cell_core.run_cell_generic` for one call rather than duplicating any of
those formulas here. PREPARE_KW pins the non-default arguments the
original SLURM sweeps used (from logs/slurm_*/job.sh), which is what
makes the reconstructed training set identical to the original cell's.

Usage (one cell, matches the SLURM array pattern):
    python3 -m experiments.augment_cell --dataset criteo --N 4000 --seed 3
"""

import argparse
import importlib

from experiments import cell_core

DATASETS = ["adultsemi", "actg", "criteo", "lalonde", "nonnested",
            "diabetes"]

# prepare-affecting kwargs the original sweeps overrode (see job.sh logs).
PREPARE_KW = {
    "adultsemi": {"steps": 800},
    "actg": {"steps": 500},
    "criteo": {"variant": "full", "subsample": 50_000, "steps": 500},
    "lalonde": {"steps": 500},
    "nonnested": {"steps": 1000},
    "diabetes": {"steps": 500},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--methods", nargs="+", default=["mlp"])
    args = ap.parse_args()

    mod = importlib.import_module(f"experiments.run_cell_{args.dataset}")

    captured = {}
    real = cell_core.run_cell_generic

    def capture(cell_dir, prepare, N, seed, **kw):
        captured.update(cell_dir=cell_dir, prepare=prepare,
                        N_sim=kw.get("N_sim", 1000),
                        lambda_people=kw.get("lambda_people", 1.0),
                        max_time_mult=kw.get("max_time_mult", 1.5),
                        steps=kw.get("steps", 500))
        return []

    cell_core.run_cell_generic = capture
    try:
        mod.run_one_cell(N=args.N, seed=args.seed,
                         **PREPARE_KW.get(args.dataset, {}))
    finally:
        cell_core.run_cell_generic = real
    if "prepare" not in captured:
        raise SystemExit("run_one_cell never reached run_cell_generic")

    rows = cell_core.augment_cell_generic(
        captured["cell_dir"], captured["prepare"], args.N, args.seed,
        s2_methods=args.methods, N_sim=captured["N_sim"],
        lambda_people=captured["lambda_people"],
        max_time_mult=captured["max_time_mult"],
        mlp_steps=captured["steps"],
    )
    added = [r["method"] for r in rows]
    print(f"[augment] {args.dataset} N={args.N} seed={args.seed}: "
          f"added {added if added else 'nothing (already present or no cell)'}")


if __name__ == "__main__":
    main()

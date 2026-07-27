"""One place to reconstruct any dataset's cell `prepare` exactly as the
original sweep built it (same subsample formulas, caches, split seeds),
via the same capture-shim used by `augment_cell`. Returns
(prepare, steps, sim_kwargs) so callers train with the sweep's step
budget and simulate with the sweep's queue settings.
"""

import importlib

from experiments import cell_core

# prepare-affecting kwargs the original sweeps overrode (see job.sh logs).
PREPARE_KW = {
    "adultsemi": {"steps": 800},
    "actg": {"steps": 500},
    "criteo": {"variant": "full", "subsample": 50_000, "steps": 500},
    "lalonde": {"steps": 500},
    "nonnested": {"steps": 1000},
    "diabetes": {"steps": 500},
    "mechanism": {"steps": 3000},
}

DEPLOY_N = {"adultsemi": 16000, "actg": 1497, "criteo": 32000,
            "lalonde": 1873, "nonnested": 4000, "diabetes": 48000,
            "mechanism": 32000}


def get_prepare(dataset, N, seed):
    mod = importlib.import_module(f"experiments.run_cell_{dataset}")
    captured = {}
    real = cell_core.run_cell_generic

    def capture(cell_dir, prepare, N_, seed_, **kw):
        captured.update(cell_dir=cell_dir, prepare=prepare,
                        steps=kw.get("steps", 500),
                        N_sim=kw.get("N_sim", 1000),
                        lambda_people=kw.get("lambda_people", 1.0),
                        max_time_mult=kw.get("max_time_mult", 1.5))
        return []

    cell_core.run_cell_generic = capture
    try:
        mod.run_one_cell(N=N, seed=seed, **PREPARE_KW.get(dataset, {}))
    finally:
        cell_core.run_cell_generic = real
    if "prepare" not in captured:
        raise RuntimeError("run_one_cell never reached run_cell_generic")
    sim_kwargs = {k: captured[k]
                  for k in ("N_sim", "lambda_people", "max_time_mult")}
    return captured["prepare"], captured["cell_dir"], captured["steps"], sim_kwargs

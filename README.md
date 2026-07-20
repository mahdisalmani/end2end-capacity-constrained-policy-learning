# End-to-End Policy Optimization for Capacity-Constrained Resource Allocation

Learn treatment-allocation policies from observational data when each
treatment arm has a long-run capacity budget. Instead of the classic
two-stage pipeline (fit outcome models, then price capacities with an LP),
the outcome model here is trained **end-to-end** against an
inverse-propensity-weighted (IPW) estimate of policy value, with the dual
capacity prices re-solved at every training step as an implicit function of
the model — a bilevel program. Companion code for the ISE 619 report
*"End-to-End Policy Optimization for Capacity-Constrained Resource
Allocation"* (see `papers/`).

## Problem

Given a log $(X_i, T_i, Y_i, \hat e_{T_i}(X_i))_{i=1}^N$ over $|\mathcal T|$
treatments with capacities $b_t$ (fraction of the population arm $t$ can
absorb; control unconstrained), maximize the IPW policy value

$$\hat V_{\mathrm{IPW}}(\theta) = \frac{1}{N}\sum_{i=1}^N \pi_{\theta,T_i}(X_i)\,\frac{Y_i}{\hat e_{T_i}(X_i)}$$

over the softmax dual-price policy class

$$\pi_{\theta,t}(x) = \operatorname{softmax}_t\!\big((m_{t,\theta}(x)-\mu_{\theta,t})/\tau\big),$$

where the shadow prices $\mu_\theta \ge 0$ solve an inner
capacity-pricing problem for the current scores $M_\theta = (m_{t,\theta}(X_i))_{i,t}$.

## Methods

| Name | Inner objective | Gradient path | Where |
|------|-----------------|---------------|-------|
| **F**   | $F(\mu)=\frac1N\sum_i \sigma_i(\mu)^\top(M_i-\mu) + b^\top\mu$ (non-convex, literal softmax Lagrangian) | scipy L-BFGS-B forward, implicit-function-theorem backward on the KKT system | `src/inner_F.py` |
| **G**   | $G(\mu)=\frac\tau N\sum_i \log\sum_t e^{(M_{it}-\mu_t)/\tau} + b^\top\mu$ (convex log-sum-exp dual) | CVXPYLayer (diffcp) | `src/inner_G.py` |
| **Gs**  | same $G$ | scipy L-BFGS-B + IFT (no cvxpy in the loop, any $N$) | `src/inner_G.py` |
| **Alt** | same as F | block-coordinate: $\mu$ refreshed every `inner_freq` steps, treated as a constant (no IFT) — the dual-refresh idea of Rodriguez-Diaz et al., arXiv:2511.04909 | `src/train_alt.py` |
| **S2-*** | two-stage baseline (Tang et al. 2024): fit $\hat m_t$ per arm (OLS / lasso / tree / kNN / DR / MLP), solve the sample dual LP for $\hat\mu$, deploy $\arg\max_t \hat m_t(x)-\hat\mu_t$ | none | `src/s2_dual.py` |

The two inner objectives satisfy $F \le G \le F + \tau\log|\mathcal T|$
(they differ by the policy entropy, $G-F = \frac{\tau}{N}\sum_i H(\sigma_i)$)
and coincide as $\tau \to 0$ with the hard LP dual. A property worth knowing
(verified in `tests/test_inner_layers.py`): **G's** stationarity is
$b_t = \frac1N\sum_i \pi_{t,i}$ on binding arms, so its inner optimum is
feasible-in-expectation *exactly*; **F's** stationary point can exceed caps
at finite $\tau$. Deployment therefore either uses a queueing repair layer
or re-prices with a sub-capacity buffer (see below).

## Repository layout

```
src/                      core library
  config.py               experiment constants (N, T=10, D=30, tau, B)
  data.py                 nested synthetic DGP (copula covariates, monotone arms)
  models.py               MLPScore: shared trunk + T-dim head
  policy.py               softmax policy, IPW / DR / oracle value estimators
  inner_common.py         shared L-BFGS-B forward + IFT backward (KKT, active set)
  inner_F.py / inner_G.py the two inner objectives (G also has the CVXPYLayer)
  train.py / train_alt.py bilevel (F/G/Gs) and alternating training loops
  s2_dual.py              two-stage baselines: outcome fits + dual LP + argmax
  evaluation.py           score_policy_pair + method evaluators
  baselines.py            random / oracle-greedy reference policies
  comparison.py           final comparison table

generate_data.py          snapshot train/eval .npz from src.config
main.py                   run all methods on the snapshot -> comparison CSV

experiments/
  common.py               shared harness: assigners, queue simulator, IPW, subsampling
  cell_core.py            generic (N, seed) cell: full method suite + queue sim
  sweep_core.py           generic multiprocessing (N x seed) sweep driver
  run_cell_{synth,criteo,lalonde,nonnested}.py   dataset bindings (thin)
  sweep_{synth,criteo,lalonde,nonnested}.py      sweep CLIs (thin)
  n_sweep_criteo.py / n_sweep_lalonde.py         single-process N-sweeps + plots
  real_queue_experiment.py                       snapshot queue experiment
                                                 (softmax-sampling deployment)
  data_criteo.py / data_lalonde.py / data_nonnested.py   dataset loaders / DGP
  add_s2_mlp_*.py, reeval_nonnested_oracle.py, plot_*.py post-hoc add-ons
  legacy/                 superseded first-generation pipeline (see headers)

scripts/
  tau_study.py            temperature study: how tau trades policy sharpness
                          against capacity feasibility, for F vs G
  scaling_study.py        wall-clock per training step vs |T| for G/Gs/F/Alt

tests/                    numerical checks (IFT vs finite differences, F–G
                          identity, complementary slackness, LP = tau->0 limit,
                          queue conservation, training smoke)
legacy/ipw_policy.py      original T=3 prototype the repo grew out of
papers/                   reference paper (arXiv:2511.04909) + project reports
docs/dgp.tex              LaTeX description of the non-nested DGP
docs/cleanup_report.html  code-review / refactor / verification report
docs/experiments_report.html  interactive results report (open in a browser)
```

## Install & run

```bash
pip install -r requirements.txt

# sanity checks (no data needed, ~1 min)
python -m tests.test_inner_layers
python -m tests.test_policy_and_pipeline

# snapshot pipeline: generate data, run every method, print comparison table
python generate_data.py
python main.py

# queue-deployment experiment on the snapshot
python -m experiments.real_queue_experiment --N-sim 2000 --num-sim-seeds 3

# multi-seed sweeps (results/ CSVs + plots)
python -m experiments.sweep_synth     --seeds 20 --workers 20
python -m experiments.sweep_nonnested --seeds 20 --workers 20 --steps 1500
python -m experiments.sweep_lalonde   --seeds 20 --workers 20   # downloads NBER files
python -m experiments.sweep_criteo    --seeds 20 --workers 20   # downloads Criteo (~300MB)

# standalone studies (write results/tau_study.json, results/scaling.json)
python scripts/tau_study.py
python scripts/scaling_study.py
```

On a SLURM cluster, prefer one job per cell over the local multiprocessing
pool — cells are independent and resumable, so they map onto an array job and
scale past a single node's cores:

```bash
scripts/slurm_sweep.sh criteo "500 1000 2000 4000 8000 16000 32000" 6
squeue -u "$USER"
```

Each array task runs one `(N, seed)` cell and writes its own
`results/<dataset>_cells/cell_N{N}_seed{s}.csv`; aggregate with
`experiments.sweep_core.gather_results` once they land (safe to run while jobs
are still in flight).

The Criteo loader tries the original scikit-uplift S3 bucket first, then the
Hugging Face mirror of `criteo-research-uplift-v2.1` (the S3 bucket began
returning 403 in 2026). The discontinued 10% file is derived locally from the
full one on demand, so `--criteo-variant 10pct` still works — but it is a
different 10% sample than the historical one, so row-level numbers will not
match pre-2026 runs.

Every sweep cell is resumable: finished cells are cached as
`results/*_cells/cell_N{n}_seed{s}.csv` and skipped on re-run (`--force`
overrides). Failures leave a `.FAILED` traceback next to the cell.

## Deployment conventions (important when comparing numbers)

- **Sweep/cell harnesses** deploy every learned model deterministically:
  re-solve the dual LP on the *train* scores with capacities shrunk by
  `--cap-buffer` (default 0.92), then assign
  $\arg\max_t (m_t(x) - \mu_t^{\mathrm{calibrated}})$
  (`experiments.common.arms_and_assigner_from_model`). No eval peeking.
- **`real_queue_experiment.py`** instead *samples* from the softmax policy
  with no buffer — the report's original stochastic deployment. Its wait
  times are not comparable to the sweep numbers.
- The bilevel convention is that $\mu$ is part of the trained policy:
  never re-solve $\mu$ on eval data (the legacy `run_cell.py` "-mu"
  variants did exactly that, and are retired).

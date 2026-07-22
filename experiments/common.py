"""
Shared building blocks for the experiment harnesses.

This module is the single home of code that used to be copy-pasted across
`real_queue_experiment.py`, `n_sweep_*.py`, `run_cell_*.py` and the
`add_s2_mlp_*.py` scripts:

  - assigner factories (random / treat-all / oracle-greedy / S2 / model+LP),
  - the discrete-event queueing simulator and its arrival streams,
  - IPW policy-value helpers,
  - row-subsampling of a dataset dict,
  - the pooled logistic propensity fit used by the real-data loaders.

`real_queue_experiment` and `n_sweep_criteo` re-export these names, so all
historical import paths keep working.

Conventions:
  - a "dataset dict" has keys {X, T, Y, e_T, E, Y_pot, Beta, Alpha};
  - an "assigner" has signature (rng, person_idx_in_eval) -> arm, with all
    trained artifacts precomputed at construction time so the inner
    simulation loop does no torch / sklearn calls.
"""

import os
import time
from collections import deque

import numpy as np
import torch

from src.s2_dual import fit_outcome_models, get_mhat_matrix, solve_dual_lp


S2_METHODS = ["linear", "lasso", "tree", "knn", "dr"]


# === Assigners ===============================================================

def make_random_assigner(T):
    def assign(rng, person_idx):
        return int(rng.integers(T))
    return assign


def make_treat_all_assigner():
    """Always pick the treatment arm (T=1)."""
    def assign(rng, person_idx):
        return 1
    return assign


def make_oracle_greedy_assigner(eval_data):
    a_star = eval_data["Y_pot"].argmax(axis=1)
    def assign(rng, person_idx):
        return int(a_star[person_idx])
    return assign


def make_s2_assigner(outcome_models, mu_hat, eval_data, T):
    """Deterministic argmax over (m_hat(x) - mu_hat)."""
    M_hat = get_mhat_matrix(outcome_models, eval_data["X"], T)
    a_star = (M_hat - mu_hat[None, :]).argmax(axis=1)
    def assign(rng, person_idx):
        return int(a_star[person_idx])
    return assign


def arms_and_assigner_from_model(model, train_data, eval_data, B, cap_buffer):
    """Return (arms_train, arms_eval, assigner) for a trained score model.

    Deterministic deployment used by every sweep/cell harness: re-solve the
    dual LP on the model's M(X_train) with cap vector `cap_buffer * B`
    (default 0.92 -> sub-cap), then deploy `argmax(M - mu_calibrated)` on
    both splits. Calibration uses train data only — no peek at eval.
    """
    B_arr = np.asarray(B, dtype=float)
    B_shrunk = cap_buffer * B_arr

    with torch.no_grad():
        M_train = model(torch.tensor(train_data["X"])).numpy()
    mu_calibrated, _, _, _ = solve_dual_lp(M_train, B_shrunk, verbose=False)
    arms_train = (M_train - mu_calibrated[None, :]).argmax(axis=1)

    with torch.no_grad():
        M_eval = model(torch.tensor(eval_data["X"])).numpy()
    arms_eval = (M_eval - mu_calibrated[None, :]).argmax(axis=1)

    def assigner(rng, person_idx):
        return int(arms_eval[person_idx])

    return arms_train, arms_eval, assigner


def s2_arms_and_assigner(train_data, eval_data, T, B, method):
    """Fit one S2 pipeline (outcome models -> dual LP -> argmax policy).

    Returns (arms_train, arms_eval, assigner).
    """
    outcome_models = fit_outcome_models(
        X_train=train_data["X"],
        T_train=train_data["T"],
        Y_train=train_data["Y"],
        T=T, method=method,
        E_train=train_data["E"],
    )
    M_hat_train = get_mhat_matrix(outcome_models, train_data["X"], T)
    mu_hat, _, _, _ = solve_dual_lp(M_hat_train, B, verbose=False)
    arms_train = (M_hat_train - mu_hat[None, :]).argmax(axis=1)
    M_hat_eval = get_mhat_matrix(outcome_models, eval_data["X"], T)
    arms_eval = (M_hat_eval - mu_hat[None, :]).argmax(axis=1)
    assigner = make_s2_assigner(outcome_models, mu_hat, eval_data, T)
    return arms_train, arms_eval, assigner


# === Arrival streams =========================================================
# One paired stream per sim_seed, shared across all methods so the method
# delta is not contaminated by Poisson noise.

def make_streams(eval_data, N_sim, lambda_people, B, max_time_mult, seed):
    rng = np.random.default_rng(seed)
    N_eval = eval_data["X"].shape[0]
    T = len(B)

    inter = rng.exponential(scale=1.0 / lambda_people, size=N_sim)
    people_t = np.cumsum(inter)
    person_idx = rng.integers(0, N_eval, size=N_sim)

    T_max = float(people_t[-1] * max_time_mult)

    resource_t = []
    for t in range(T):
        rate = float(B[t]) * lambda_people
        if rate <= 0.0:
            resource_t.append(np.empty(0, dtype=np.float64))
            continue
        expected = rate * T_max
        n_init = max(16, int(expected + 8.0 * np.sqrt(expected) + 16.0))
        ts = np.cumsum(rng.exponential(scale=1.0 / rate, size=n_init))
        while ts[-1] < T_max:
            extra = np.cumsum(rng.exponential(scale=1.0 / rate, size=n_init))
            ts = np.concatenate([ts, ts[-1] + extra])
        ts = ts[ts <= T_max]
        resource_t.append(ts)

    return people_t, person_idx, T_max, resource_t


# === Discrete-event simulator ================================================

def simulate(people_t, person_idx, resource_t, assigner, T, T_max,
             eval_data, sim_seed):
    """One queueing simulation. Returns dict of per-person arrays of length N_sim."""
    Y_pot = eval_data["Y_pot"]
    rng = np.random.default_rng(sim_seed * 9_973_337 + 1)
    N_sim = len(people_t)

    inventory = np.zeros(T, dtype=np.int64)
    queues = [deque() for _ in range(T)]
    next_r_idx = np.zeros(T, dtype=np.int64)

    arms = np.zeros(N_sim, dtype=np.int64)
    waits = np.zeros(N_sim, dtype=np.float64)
    served = np.zeros(N_sim, dtype=bool)
    outcomes = np.full(N_sim, np.nan, dtype=np.float64)

    def serve_resources_until(t_now):
        for a in range(T):
            ts = resource_t[a]
            idx = next_r_idx[a]
            n = len(ts)
            while idx < n and ts[idx] <= t_now:
                t_r = ts[idx]
                idx += 1
                if queues[a]:
                    t_arr, k, p_idx = queues[a].popleft()
                    waits[k] = t_r - t_arr
                    served[k] = True
                    arms[k] = a
                    outcomes[k] = Y_pot[p_idx, a]
                else:
                    inventory[a] += 1
            next_r_idx[a] = idx

    for k in range(N_sim):
        t_p = people_t[k]
        serve_resources_until(t_p)

        a = assigner(rng, person_idx[k])
        if inventory[a] > 0:
            inventory[a] -= 1
            arms[k] = a
            waits[k] = 0.0
            served[k] = True
            outcomes[k] = Y_pot[person_idx[k], a]
        else:
            queues[a].append((t_p, k, person_idx[k]))

    serve_resources_until(T_max)

    for a in range(T):
        while queues[a]:
            t_arr, k, p_idx = queues[a].popleft()
            arms[k] = a
            waits[k] = T_max - t_arr
            served[k] = False

    return {
        "arm": arms,
        "person_idx": person_idx,
        "wait": waits,
        "served": served,
        "oracle_outcome": outcomes,
        # Needed by aggregate_one to price unserved arrivals at the control arm.
        "Y_pot": Y_pot,
    }


def aggregate_one(records, method, sim_seed, B, N_sim, sim_wall):
    """Aggregate one simulation run into a flat result row.

    Two outcome columns, deliberately:

      `mean_oracle_outcome_served` averages over SERVED arrivals only. It is
      a conditional-on-service quantity and is subject to survivorship bias —
      a policy that abandons hard cases looks better on it. Kept because it
      answers "what did the people who got a resource receive?".

      `mean_oracle_outcome_all` charges unserved arrivals their control-arm
      (t=0) outcome, i.e. what they would have got with no intervention.
      This is the population quantity and the one to compare methods on,
      because it prices non-service instead of discarding it.
    """
    arms = records["arm"]
    waits = records["wait"]
    served = records["served"]
    outcomes = records["oracle_outcome"]
    person_idx = records["person_idx"]
    Y_pot = records.get("Y_pot")
    T = len(B)

    n_unserved = int((~served).sum())
    served_waits = waits[served]
    served_outcomes = outcomes[served]

    # Population outcome: served arrivals get their realized arm, unserved
    # arrivals fall back to the control arm (no intervention delivered).
    if Y_pot is not None and Y_pot.size and not np.all(np.isnan(Y_pot)):
        all_out = np.where(served, outcomes, Y_pot[person_idx, 0])
        mean_all = float(np.nanmean(all_out)) if all_out.size else float("nan")
    else:
        mean_all = float("nan")   # real data: no counterfactuals

    row = {
        "method": method,
        "sim_seed": sim_seed,
        "N_sim": N_sim,
        "total_wait": float(waits.sum()),
        "mean_wait_all": float(waits.mean()),
        "mean_wait_served": (float(served_waits.mean())
                             if served_waits.size > 0 else float("nan")),
        "mean_oracle_outcome_served": (float(served_outcomes.mean())
                                       if served_outcomes.size > 0 else float("nan")),
        "mean_oracle_outcome_all": mean_all,
        "num_unserved": n_unserved,
        "frac_unserved": n_unserved / N_sim,
        "sim_wall_s": sim_wall,
    }
    counts = np.bincount(arms, minlength=T)
    fracs = counts / N_sim
    for t in range(T):
        row[f"alloc_{t}"] = float(fracs[t])
    return row


def simulate_one_method(assigner, method, eval_data, T, B, N_sim,
                        lambda_people, max_time_mult, sim_seed):
    """Convenience: streams + simulate + aggregate for one (method, seed)."""
    people_t, person_idx, T_max, resource_t = make_streams(
        eval_data, N_sim, lambda_people, B, max_time_mult,
        seed=sim_seed * 7 + 13,
    )
    t0 = time.time()
    recs = simulate(
        people_t, person_idx, resource_t, assigner,
        T=T, T_max=T_max, eval_data=eval_data, sim_seed=sim_seed,
    )
    wall = time.time() - t0
    return aggregate_one(recs, method, sim_seed, B, N_sim, wall)


# === IPW policy value ========================================================

def precompute_arms(assigner, n_eval, rng_seed=0):
    """Apply assigner to every eval index. RNG is shared across all
    calls so stochastic policies (random) get a single coherent draw."""
    rng = np.random.default_rng(rng_seed)
    return np.fromiter(
        (assigner(rng, i) for i in range(n_eval)),
        dtype=np.int64, count=n_eval,
    )


def ipw_policy_value(arms, data):
    """V_IPW = (1 / N) * sum_i  Y_i * 1{arms[i] = T_i} / e_T_i.

    Works for any (arms, data) where the lengths match. Use it for
    both train-side and eval-side IPW values.
    """
    Y = data["Y"]
    T = data["T"]
    e_T = data["e_T"]
    matched = (arms == T).astype(np.float64)
    return float((Y * matched / e_T).mean())


def eval_arms_row(method, arms_train, arms_eval, train_data, eval_data):
    """Standard IPW row for one method's deterministic policy arms.

    When counterfactuals exist (synthetic / semi-synthetic data), also
    records the DIRECT oracle value of the deployed assignment on eval —
    the ground-truth quantity the IPW columns estimate. NaN on real data.
    """
    Y_pot = eval_data.get("Y_pot")
    if Y_pot is not None and Y_pot.size and np.isfinite(Y_pot).all():
        oracle_val = float(Y_pot[np.arange(len(arms_eval)), arms_eval].mean())
    else:
        oracle_val = float("nan")
    return {
        "method": method,
        "ipw_train": ipw_policy_value(arms_train, train_data),
        "ipw_val": ipw_policy_value(arms_eval, eval_data),
        "oracle_val": oracle_val,
    }


# === Dataset-dict utilities ==================================================

def atomic_pickle_dump(payload, path):
    """Write a pickle so concurrent writers cannot clobber each other.

    The temp name carries the pid: when N array jobs start at once with no
    cache present, they all build it, and a shared temp path means one job's
    os.replace() moves the file out from under another's, which then dies with
    FileNotFoundError. With per-process temp names every writer succeeds and
    the last rename wins — safe here because all writers produce identical
    content for the same cache key.
    """
    import pickle

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


PER_ROW_KEYS = {"X", "T", "Y", "e_T", "E", "Y_pot"}


def subsample_rows(train_data, n, seed):
    """Take a permuted subset of n rows from train_data. Beta / Alpha
    are not per-row; pass them through unchanged."""
    rng = np.random.default_rng(seed)
    N_train = len(train_data["T"])
    n = min(n, N_train)
    idx = rng.permutation(N_train)[:n]
    return {
        k: (v[idx] if k in PER_ROW_KEYS else v)
        for k, v in train_data.items()
    }


# === Real-data propensity fit ================================================

def fit_logistic_propensity(X, T, clip=(0.05, 0.95), fit_idx=None):
    """e(x) = P(T=1 | X=x) via logistic regression, clipped for IPW stability.

    `fit_idx` restricts ESTIMATION to a subset of rows (the training split)
    while still predicting for all rows. Fitting on the pooled data and then
    splitting leaks eval-split information into the IPW weights that the same
    eval split is later scored with — the propensity model has seen the
    outcomes' treatment assignments on rows it is asked to weight. Pass the
    training indices to keep the eval-side estimator honest.

    Standardization has the same issue; see `standardize_train_fit` below.
    """
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1.0, max_iter=2000)
    if fit_idx is None:
        lr.fit(X, T)
    else:
        lr.fit(X[fit_idx], T[fit_idx])
    e1 = lr.predict_proba(X)[:, 1]
    return np.clip(e1, clip[0], clip[1])


def fit_multinomial_propensity(X, T, n_arms, clip=0.02, fit_idx=None):
    """e_t(x) = P(T=t | X=x) for multi-arm data via multinomial logistic
    regression. Fit on `fit_idx` (training rows) only, predict for all rows;
    per-class probabilities are clipped below at `clip` and renormalized so
    IPW weights stay bounded. Returns the full (n, n_arms) matrix."""
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1.0, max_iter=3000)
    if fit_idx is None:
        lr.fit(X, T)
    else:
        lr.fit(X[fit_idx], T[fit_idx])
    E = np.zeros((len(T), n_arms))
    E[:, lr.classes_.astype(int)] = lr.predict_proba(X)
    E = np.clip(E, clip, 1.0)
    return E / E.sum(axis=1, keepdims=True)


def standardize_train_fit(X, fit_idx=None):
    """Standardize X with mean/scale estimated on `fit_idx` rows only."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    if fit_idx is None:
        return scaler.fit_transform(X)
    scaler.fit(X[fit_idx])
    return scaler.transform(X)


def split_indices(n, train_frac, seed):
    """Shared permutation split so loaders can fit preprocessing on train only."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_frac * n))
    return perm[:n_train], perm[n_train:], n_train

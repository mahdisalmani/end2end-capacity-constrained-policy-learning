"""Port of the Dual-Guided Learning synthetic matching generator.

Source: `dgl/Benchmarks/SyntheticMatching.py` in
https://github.com/paularodr/Dual-Guided-Learning (MIT), the benchmark of
Rodriguez-Diaz, Bansak and Paulson (arXiv:2511.04909), itself motivated by
the refugee-assignment setting of Bansak and Paulson (2024).

WHY A PORT AND NOT THE BENCHMARK ITSELF
---------------------------------------
Their task cannot be run here unchanged, for two structural reasons:

  1. It is full-information. Every location utility u_ij is observed and used
     as a supervised label. Our outer objective is an IPW estimate, which
     exists only because the outcome of the arm NOT taken is missing.
  2. It is per-instance. Their dataset is 800 independent cohorts of 10, each
     with its own feasible set and its own dual vector (their Algorithm 1
     computes one lambda per instance). Our constraint is a population rate,
     so there is no per-cohort feasible set to inherit.

What ports cleanly is the DATA-GENERATING PROCESS. Their generator emits, for
every individual, the utilities of ALL locations -- exactly a potential-outcome
matrix. So we keep their features, their outcome model, and their capacities,
and place them in our problem: one arrival stream, bandit feedback, a
population-level capacity, and a queue at deployment. Report it as "their DGP,
our problem", never as "their benchmark".

WHAT IS KEPT VERBATIM FROM THEIR CODE
-------------------------------------
  * 10 base features + 5 location features, X ~ N(0, s) with per-feature
    s ~ U(0.1, 5.0), drawn once as part of the DGP.
  * base_form='simple':  y_base = sum_k c_k X_base_k, c_k ~ randint(1, 5),
    then z-scored.
  * locdiff_form='complex': for each location j, coefficients c_j ~
    randint(-5, 5), and
        y_loc_j = sum_k c_jk |X_loc_k| + c_j,last * X_loc_0 * X_loc_1,
    then the whole (n x 3) block z-scored. NOTE the absolute values: the
    location term is EVEN in the location features, so a linear model has zero
    covariance with it. This is the same geometry as our mechanism dataset,
    arrived at independently by them.
  * y = sigmoid(y_base + 1.5 * y_loc + 0.1 * eps), giving u_ij in (0, 1).
  * Realised outcomes are Bernoulli draws of those probabilities, as in their
    `Yobs = torch.bernoulli(self.Ys)`.
  * Capacities [0.2, 0.3, 0.5], already expressed by them as fractions of the
    population.

WHAT WE ADD, AND WHY
--------------------
An unconstrained outside arm t=0 ("not assigned this period"), with capacity
b_0 = 1 and outcome sigmoid(y_base), i.e. their own model with the location
term switched off. Two reasons, both substantive:

  * Their one-of-many constraint forces every individual to be assigned, so
    their capacities sum to exactly 1.0. At sum_t b_t = 1 the smoothed dual G
    is invariant to adding a constant to every price (the log-sum-exp term
    falls by c, the linear term rises by c * sum_t b_t = c), so mu* is unique
    only up to that shift, and any sub-capacity deployment buffer makes the
    hard LP infeasible. The outside arm restores sum_t b_t > 1, the
    non-degeneracy condition of our feasibility proposition, and anchors the
    scale at mu_0 = 0.
  * Our problem class has an outside option by definition (t = 0 is the
    unconstrained no-treatment arm). Adding it is what makes this an instance
    of our problem rather than of theirs.

Set outside=False to reproduce their tight structure exactly (3 arms,
sum b = 1); deployment must then use cap_buffer=1.0.

Logging is uniform over the arms, so propensities are known exactly and the
comparison isolates the constraint structure rather than confounding.
"""

import numpy as np

# --- their defaults, verbatim -------------------------------------------------
N_BASE_FEATURES = 10
N_LOC_FEATURES = 5
N_LOCATIONS = 3
NOISE = 0.1
WEIGHT_LOCDIFF = 1.5
CAPACITIES = [0.2, 0.3, 0.5]
DGP_SEED = 10          # their rand_seed default

D = N_BASE_FEATURES + N_LOC_FEATURES        # 15 covariates
N_EVAL = 10_000


def _dgp_constants():
    """Coefficients and feature scales: fixed across cells, as in their code
    where they are drawn once per dataset under `rand_seed`."""
    rng = np.random.default_rng(DGP_SEED)
    feature_stds = rng.uniform(0.1, 5.0, size=D)
    # torch.randint(1, 5) -> 1..4 ; torch.randint(-5, 5) -> -5..4
    base_coeffs = rng.integers(1, 5, size=N_BASE_FEATURES).astype(float)
    loc_coeffs = rng.integers(-5, 5,
                              size=(N_LOCATIONS, N_LOC_FEATURES + 1)).astype(float)
    return feature_stds, base_coeffs, loc_coeffs


FEATURE_STDS, BASE_COEFFS, LOC_COEFFS = _dgp_constants()


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _raw_components(X):
    """(y_base, y_loc) before standardisation -- their `_generate_baseline_label`
    and `_generate_locdiff_label` with base_form='simple', locdiff_form='complex'."""
    X_base = X[:, :N_BASE_FEATURES]
    X_loc = X[:, N_BASE_FEATURES:]

    y_base = (X_base * BASE_COEFFS).sum(axis=1)

    y_loc = np.empty((X.shape[0], N_LOCATIONS))
    for j in range(N_LOCATIONS):
        c = LOC_COEFFS[j]
        # exp = 1 in their complex branch: |X_loc| ** 1
        y_loc[:, j] = (np.abs(X_loc) * c[:-1]).sum(axis=1) \
            + c[-1] * X_loc[:, 0] * X_loc[:, 1]
    return y_base, y_loc


def _potential_outcomes(X_all, rng, outside):
    """Their label model, standardised over the pooled draw exactly as their
    `_generate_labels` standardises over its dataset. Returns probabilities."""
    y_base, y_loc = _raw_components(X_all)

    y_base = (y_base - y_base.mean()) / y_base.std()
    y_loc = (y_loc - y_loc.mean()) / y_loc.std()

    y_comb = y_base[:, None] + WEIGHT_LOCDIFF * y_loc
    y_comb = y_comb + rng.normal(0.0, NOISE, size=y_comb.shape)
    p_loc = _sigmoid(y_comb)                       # (n, 3) in (0, 1)

    if not outside:
        return p_loc

    # outside arm: their model with the location term switched off
    y0 = y_base + rng.normal(0.0, NOISE, size=y_base.shape)
    return np.concatenate([_sigmoid(y0)[:, None], p_loc], axis=1)


def capacities(outside=True):
    return np.array(([1.0] if outside else []) + list(CAPACITIES), dtype=float)


def _slice(X, T, Y_pot, rng, n_arms):
    n = X.shape[0]
    E = np.full((n, n_arms), 1.0 / n_arms)
    # realised outcome is a Bernoulli draw, as in their `Yobs = bernoulli(Ys)`
    Y = rng.binomial(1, Y_pot[np.arange(n), T]).astype(float)
    return {"X": X, "T": T, "Y": Y, "e_T": E[np.arange(n), T], "E": E,
            "Y_pot": Y_pot,
            "Beta": np.zeros((n_arms, D)), "Alpha": np.zeros((n_arms, D))}


def generate_dglmatch(N_train, seed, n_eval=N_EVAL, outside=True):
    """Return (train_data, eval_data, cfg).

    Train and evaluation individuals are drawn together and standardised
    together, mirroring their generator, which builds one dataset and then
    splits it into train/val/test.
    """
    rng = np.random.default_rng(7_919 * seed + 101)
    n_all = N_train + n_eval

    X_all = rng.normal(loc=0.0, scale=FEATURE_STDS, size=(n_all, D))
    Y_pot_all = _potential_outcomes(X_all, rng, outside)

    n_arms = Y_pot_all.shape[1]
    T_all = rng.integers(0, n_arms, size=n_all)

    tr = slice(0, N_train)
    ev = slice(N_train, n_all)
    train_data = _slice(X_all[tr], T_all[tr], Y_pot_all[tr], rng, n_arms)
    eval_data = _slice(X_all[ev], T_all[ev], Y_pot_all[ev], rng, n_arms)

    cfg = {"N": int(N_train), "T": int(n_arms), "D": D, "TAU": 0.1,
           "B": capacities(outside), "outside": bool(outside)}
    return train_data, eval_data, cfg


def oracle_value(eval_data, outside=True):
    """Ground-truth value of the best capacity-feasible assignment on this
    evaluation draw, by solving the assignment LP on the true probabilities.

    The LP over the simplex with population capacities is solved exactly by
    its dual; we use scipy's linear_sum-style greedy on the Lagrangian, which
    is exact here because the constraint matrix is an interval matrix (each
    person contributes to exactly one arm).
    """
    from scipy.optimize import linprog

    Y = eval_data["Y_pot"]
    n, n_arms = Y.shape
    b = capacities(outside)

    # max sum_i sum_t z_it Y_it  s.t.  sum_t z_it = 1,  mean_i z_it <= b_t
    c = -Y.reshape(-1)
    # equality: one arm per person
    rows, cols, data = [], [], []
    for i in range(n):
        for t in range(n_arms):
            rows.append(i); cols.append(i * n_arms + t); data.append(1.0)
    from scipy.sparse import coo_matrix
    A_eq = coo_matrix((data, (rows, cols)), shape=(n, n * n_arms))
    b_eq = np.ones(n)
    # capacity rows
    rows, cols, data = [], [], []
    for t in range(n_arms):
        for i in range(n):
            rows.append(t); cols.append(i * n_arms + t); data.append(1.0 / n)
    A_ub = coo_matrix((data, (rows, cols)), shape=(n_arms, n * n_arms))
    res = linprog(c, A_ub=A_ub, b_ub=b, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, 1), method="highs")
    if not res.success:
        raise RuntimeError(f"oracle LP failed: {res.message}")
    return float(-res.fun / n)

"""
Semi-synthetic multi-arm allocation problem on REAL census covariates
(UCI Adult, id=2, N=48,842, CC BY 4.0).

Why semi-synthetic: the pure-synthetic DGP draws X from a copula, which
reviewers discount; the real datasets (Criteo, LaLonde) have no
counterfactuals, so oracle policy value cannot be evaluated. Here the
covariates are real — correlated, mixed-type, heavy-tailed (capital-gain) —
while the potential outcomes are constructed, so every method can be scored
against ground truth on held-out data.

Design (all surfaces fixed by MASTER_SEED, independent of the train/eval
split and of per-cell training seeds):

  Arms      T = 8: arm 0 is an uncapped default; arms 1..7 are scarce
            programs with capacity `cap_scale * 0.08` each
            (sum of caps 1.0 + 7*0.08 = 1.56 >= 1, feasible).
  Effects   tau_k(x) = A * [ (1-lam) * linear_k(x) + lam * bump_k(x) ]
            where bump_k is a Gaussian bump on 3 real feature dims centred
            at an actual data row, covering ~15-20% of the population —
            roughly twice each arm's 8% cap, so caps BIND at the optimum
            (verified at build time; the LaLonde lesson).
            `lam` (te_nonlinearity) is the misspecification dial: at lam=0
            a linear outcome model is correctly specified and two-stage
            S2-linear/lasso should win; at lam=1 the effect is invisible
            to linear models.
  Logging   confounded: assignment propensities are a softmax over scores
            aligned with the bumps (strength `conf_strength`), clipped at
            0.02 — so naive averaging is biased and IPW matters. The TRUE
            logging propensities are recorded (standard for semi-synthetic
            benchmarks) and used by the estimators.
  Outcome   Y^0 = m0(x) + eps,  Y^k = m0(x) + tau_k(x) + eps_k,
            m0 linear-plus-mild-curvature, sigma_y = 0.5, A = 4.

Returns the project's standard (train_data, eval_data, cfg) with real
Y_pot on both splits; 70/30 split; loader output cached to
data/uci/adult_semi_<key>.pkl.
"""

import os
import pickle

import numpy as np
import pandas as pd

from experiments.common import atomic_pickle_dump, split_indices

RAW_CSV = "data/uci/adult.csv"
CACHE_DIR = "data/uci"
MASTER_SEED = 0
# Bumped whenever the surface-construction code changes, so cached datasets
# built by an older DGP cannot be silently reused.
DGP_VERSION = 3

NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEG = ["workclass", "marital-status", "occupation", "relationship",
         "race", "sex", "native-country"]

T_ARMS = 8
BASE_CAP = 0.08
SIGMA_Y = 0.5
EFFECT_A = 4.0


def _load_X():
    if not os.path.exists(RAW_CSV):
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=2)
        os.makedirs(CACHE_DIR, exist_ok=True)
        pd.concat([ds.data.features, ds.data.targets], axis=1).to_csv(RAW_CSV, index=False)
    df = pd.read_csv(RAW_CSV)
    cols = {}
    for c in NUMERIC:
        cols[c] = df[c].astype(float).values
    for c in CATEG:
        # fillna BEFORE astype: Arrow-backed strings keep NaN through
        # astype(str), so missing entries would otherwise poison X.
        s = df[c].fillna("Missing").astype(str).str.strip()
        # frequency-ordinal: deterministic, keeps D small, no target used
        order = s.value_counts().index
        rank = {v: i for i, v in enumerate(order)}
        cols[c] = s.map(rank).astype(float).values
    X = np.column_stack([cols[c] for c in NUMERIC + CATEG])
    # standardize on the FULL X: this is part of the fixed DGP definition
    # (the surfaces below are functions of these coordinates), not an
    # estimator — the estimators only ever see X itself.
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X, NUMERIC + CATEG


def _farthest_rows(X, k, rng):
    """k spread-out real rows (greedy farthest-point) to centre the bumps."""
    n = len(X)
    idx = [int(rng.integers(n))]
    d = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(k - 1):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(X - X[idx[-1]], axis=1))
    return np.array(idx)


def _build_surfaces(X, lam, conf_strength, rng):
    """Fixed potential-outcome means and logging propensities on X."""
    n, d = X.shape
    K = T_ARMS - 1

    w0 = rng.normal(size=d) / np.sqrt(d)
    m0 = X @ w0 + 0.25 * np.tanh(X[:, 0] * X[:, 4])   # age x hours curvature

    # Bump dims: only features with rich support. On near-binary encoded
    # dims (sex, race) squared distance is discrete, so bump coverage jumps
    # from ~0 to ~1 with no usable middle.
    rich = [j for j in range(d) if len(np.unique(X[:, j])) >= 8]

    # Candidate centres come from DENSE regions: farthest-point sampling on
    # the raw rows picks outliers (capital-gain = 99999 style), around which
    # no bump width yields a mid-sized group. Trim to rows within 2 sigma on
    # the rich dims, then spread candidates by farthest-point within that
    # pool, and accept the first candidate whose best coverage lands in a
    # usable band.
    pool = np.where(np.abs(X[:, rich]).max(1) <= 2.0)[0]
    cand = pool[_farthest_rows(X[pool], min(K * 12, len(pool)), rng)]

    TARGET_COV, COV_BAND = 0.18, (0.12, 0.30)
    sig_grid = np.geomspace(0.15, 6.0, 60)
    tau = np.zeros((n, K))
    bump_cov = []
    next_cand = 0
    for k in range(K):
        S = rng.choice(rich, size=3, replace=False)
        best = None
        for _ in range(12):
            c = X[cand[next_cand % len(cand)]]
            next_cand += 1
            d2 = ((X[:, S] - c[S]) ** 2).sum(1)
            covs = np.array([float((np.exp(-d2 / (2 * s ** 2)) > 0.5).mean())
                             for s in sig_grid])
            i = int(np.argmin(np.abs(covs - TARGET_COV)))
            if best is None or (abs(covs[i] - TARGET_COV)
                                < abs(best[0] - TARGET_COV)):
                best = (float(covs[i]), float(sig_grid[i]), d2)
            if COV_BAND[0] <= covs[i] <= COV_BAND[1]:
                break
        cov, sig, d2 = best
        bump = np.exp(-d2 / (2 * sig ** 2))
        wk = rng.normal(size=d) / np.sqrt(d)
        lin = X @ wk
        lin = np.abs((lin - lin.mean()) / (lin.std() + 1e-9))   # >=0, mean ~0.8:
        # never degenerate (the positive-part form can collapse to ~0 mean and
        # blow up the rescale into overflow -> NaN propensities)
        lin *= bump.mean() / lin.mean()       # match average effect mass
        tau[:, k] = EFFECT_A * ((1 - lam) * lin + lam * bump)
        bump_cov.append(cov)

    M = np.column_stack([m0, m0[:, None] + tau])       # (n, T) true means

    # confounded logging: scores follow the (nonlinear) effect structure
    sc = np.zeros((n, T_ARMS))
    sc[:, 1:] = conf_strength * (tau / (np.abs(tau).mean() + 1e-9))
    sc -= sc.max(1, keepdims=True)
    E = np.exp(sc); E /= E.sum(1, keepdims=True)
    E = np.clip(E, 0.02, 1.0); E /= E.sum(1, keepdims=True)
    assert np.isfinite(M).all() and np.isfinite(E).all(), "non-finite surface"
    return M, E, bump_cov


def load_adult_semi(
    te_nonlinearity=0.85,
    cap_scale=1.0,
    conf_strength=1.5,
    train_frac=0.7,
    seed=0,
):
    """Build (or load cached) the fixed semi-synthetic dataset and split it.

    `seed` seeds ONLY the train/eval split permutation and the observed
    (T, Y) draw — the outcome surfaces and propensities are fixed by
    MASTER_SEED so every cell sees the same ground truth.
    """
    key = (round(te_nonlinearity, 4), round(cap_scale, 4),
           round(conf_strength, 4), round(train_frac, 4), seed, MASTER_SEED,
           DGP_VERSION)
    cache = os.path.join(CACHE_DIR, f"adult_semi_{abs(hash(key)) % 10**10}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            pay = pickle.load(f)
        if pay.get("_key") == key:
            return pay["train"], pay["eval"], pay["cfg"]

    X, feat_names = _load_X()
    n, d = X.shape
    master = np.random.default_rng(MASTER_SEED)
    M, E, bump_cov = _build_surfaces(X, te_nonlinearity, conf_strength, master)

    draw = np.random.default_rng(10_000 + seed)
    eps = SIGMA_Y * draw.normal(size=(n, T_ARMS))
    Y_pot = M + eps
    T_obs = np.array([draw.choice(T_ARMS, p=E[i]) for i in range(n)])
    Y_obs = Y_pot[np.arange(n), T_obs]
    e_T = E[np.arange(n), T_obs]

    B = np.array([1.0] + [cap_scale * BASE_CAP] * (T_ARMS - 1))
    tr, ev, n_train = split_indices(n, train_frac, seed + 1)

    def _slice(I):
        return {"X": X[I].copy(), "T": T_obs[I].copy(), "Y": Y_obs[I].copy(),
                "e_T": e_T[I].copy(), "E": E[I].copy(), "Y_pot": Y_pot[I].copy(),
                "Beta": np.zeros((T_ARMS, d)), "Alpha": np.zeros((T_ARMS, d))}

    train_data, eval_data = _slice(tr), _slice(ev)
    cfg = {"N": int(n_train), "T": T_ARMS, "D": int(d), "TAU": 0.1, "B": B,
           "te_nonlinearity": te_nonlinearity, "cap_scale": cap_scale,
           "conf_strength": conf_strength, "features": feat_names}

    # ---- binding check (the LaLonde lesson): does the ORACLE want more of
    # each scarce arm than its cap allows?
    demand = np.bincount(M.argmax(1), minlength=T_ARMS) / n
    over = demand[1:] / np.maximum(B[1:], 1e-9)
    print(f"[adult_semi] lam={te_nonlinearity} cap_scale={cap_scale}  "
          f"bump coverage={np.round(bump_cov, 2)}")
    print(f"[adult_semi] oracle argmax demand per scarce arm / cap: "
          f"{np.round(over, 2)}  (>1 means the cap binds)")
    print(f"[adult_semi] N={n} train={n_train} eval={n - n_train}  D={d}  "
          f"T={T_ARMS}  B[1]={B[1]:.3f}  e_T range "
          f"[{e_T.min():.3f}, {e_T.max():.3f}]")

    atomic_pickle_dump({"_key": key, "train": train_data, "eval": eval_data,
                        "cfg": cfg}, cache)
    return train_data, eval_data, cfg

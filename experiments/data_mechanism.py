"""
Mechanism DGP: dose-matching on a shared severity score, under a
prognosis nuisance dial.

The construction isolates TWO structural differences between the pipelines
and turns them into the whole problem:

    (i)  shared representation: the decision-relevant feature (a single
         severity score s(x) = |v . x|) is COMMON to all arms — every one
         of the N training samples informs it. End-to-end learns it in one
         shared trunk; two-stage splits the data into T per-arm
         regressions of ~N/T samples each and must rediscover s(x) in
         every one of them.
    (ii) level-nuisance cancellation: a prognosis term g(x) shared by all
         arms cancels inside the argmax over (m_t(x) - mu_t), so
         end-to-end never has to represent it; two-stage must drive its
         per-arm g-error below the effect size before the effect is
         visible in MSE.

Data model (structural constants fixed across N and seeds — one world):

    X ~ N(0, I_20),  severity s = |v . x|  (v a fixed dense direction)
    responders R = {s > c}, P(R) = 0.30  (the extremes on both sides of
        the v-axis need the intervention; the middle does not)
    Y(0) = g(x) + eta;   Y(1) = g(x) + delta(x) + eta
    delta = DELTA * 1{x in R} - EPS * 1{x not in R}
    (treating a non-responder is actively harmful — EPS)
    g = AMP-scaled smooth multi-index function (the nuisance dial)
    eta ~ N(0, 0.5^2);  logging T ~ Uniform{0, 1}, e = 1/2 known exactly
    cap b_1 = 0.25 < P(R) = 0.30, so capacity binds and WHO is served
        decides the value
    E[delta] = 0.30*DELTA - 0.70*EPS < 0: the arm hurts on average — a
        method that cannot find R does better assigning control, and
        filling the cap on noise is worse than doing nothing.

Why each baseline fails BY CONSTRUCTION (not by tuning):

    S2-linear / S2-lasso / S2-dr(lasso): the responder indicator is
        EVEN in v . x, so cov(x, delta) = 0 exactly — blind at every N.
    S2-tree (depth 5): R is the outside of a slab along a DENSE
        direction; axis-aligned splits cannot express |v . x| at depth 5.
    S2-knn: wide neighborhoods in d = 20 blur the slab boundary.
    S2-mlp (the capacity-matched control): can represent everything,
        but each per-arm net gets ~N/2 samples and must drive its
        g-error below DELTA before the effect is visible in MSE — the
        honest baseline that recovers as AMP -> 0 or N grows large.
    F / Gs / Alt: one trunk, all N samples, g cancels — the network
        capacity goes entirely to the severity score and its threshold.

AMP is the dial. At AMP = 0 the nuisance is absent and the remaining gap
is pure shared-representation efficiency (i); the gap then grows with AMP
through mechanism (ii). Both regimes are reported — including where
two-stage recovers — so the figure demonstrates a mechanism, not a tune.
"""

import numpy as np

# ---- fixed structural constants (one world) --------------------------------
D = 20
T_ARMS = 2
DELTA = 1.0          # benefit for responders
EPS = 0.5            # harm from treating a non-responder
SIGMA_Y = 0.5
B = np.array([1.0, 0.25])
N_EVAL = 4000
AMP_DEFAULT = 2.0

# responder threshold on s = |v.x|: P(s > c) = 2*(1-Phi(c)) = 0.30.
_C = np.array([1.0364, np.inf])

_rng_const = np.random.default_rng(20260721)
_v = _rng_const.normal(size=D)
_v /= np.linalg.norm(_v)
_U = _rng_const.normal(size=(4, D))
_U /= np.linalg.norm(_U, axis=1, keepdims=True)
_PHASE = _rng_const.uniform(0, 2 * np.pi, size=3)


def _nuisance(X, amp):
    """Smooth multi-index prognosis term with sd ~= amp."""
    if amp == 0.0:
        return np.zeros(X.shape[0])
    s1 = np.sin(2.0 * X @ _U[0] + _PHASE[0])
    s2 = np.sin(3.0 * X @ _U[1] + _PHASE[1])
    s3 = np.sin(2.0 * X @ _U[2] + _PHASE[2]) * np.cos(X @ _U[3])
    g = s1 + s2 + s3
    return amp * g / 0.87    # 0.87 = empirical sd of (s1+s2+s3)


def band_of(X):
    """1 where the treatment helps (responders: s > threshold), else 0."""
    s = np.abs(X @ _v)
    return (s > _C[0]).astype(int)


def _potential_outcomes(X, amp, rng):
    N = X.shape[0]
    g = _nuisance(X, amp)
    bands = band_of(X)
    Y_pot = np.empty((N, T_ARMS))
    Y_pot[:, 0] = g + rng.normal(0, SIGMA_Y, N)
    for k in range(1, T_ARMS):
        delta = np.where(bands == k, DELTA, -EPS)
        Y_pot[:, k] = g + delta + rng.normal(0, SIGMA_Y, N)
    return Y_pot


def _slice(X, T, Y_pot):
    N = X.shape[0]
    Y = Y_pot[np.arange(N), T]
    E = np.full((N, T_ARMS), 1.0 / T_ARMS)
    return {"X": X, "T": T, "Y": Y, "e_T": E[np.arange(N), T], "E": E,
            "Y_pot": Y_pot,
            "Beta": np.zeros((T_ARMS, D)), "Alpha": np.zeros((T_ARMS, D))}


def generate_mechanism(N_train, seed, amp=AMP_DEFAULT, n_eval=N_EVAL):
    """Return (train_data, eval_data, cfg). Logging is uniform (e = 1/T,
    known exactly) so the comparison isolates estimation, not confounding."""
    rng = np.random.default_rng(1_000_003 * seed + 17)
    X_tr = rng.normal(size=(N_train, D))
    X_ev = rng.normal(size=(n_eval, D))
    T_tr = rng.integers(0, T_ARMS, size=N_train)
    T_ev = rng.integers(0, T_ARMS, size=n_eval)
    Yp_tr = _potential_outcomes(X_tr, amp, rng)
    Yp_ev = _potential_outcomes(X_ev, amp, rng)
    cfg = {"N": int(N_train), "T": T_ARMS, "D": D, "TAU": 0.1,
           "B": B.copy(), "amp": float(amp)}
    return _slice(X_tr, T_tr, Yp_tr), _slice(X_ev, T_ev, Yp_ev), cfg


def oracle_value(eval_data):
    """Ground-truth value of the best cap-feasible assignment on this
    eval draw: band-matched arms, truncated at the caps."""
    Yp = eval_data["Y_pot"]
    bands = band_of(eval_data["X"])
    N = len(bands)
    arm_of = np.zeros(N, dtype=int)
    for k in range(1, T_ARMS):
        idx = np.flatnonzero(bands == k)[: int(B[k] * N)]
        arm_of[idx] = k
    return float(Yp[np.arange(N), arm_of].mean())

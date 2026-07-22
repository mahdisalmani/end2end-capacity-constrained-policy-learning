"""
Mechanism DGP: a dense decision margin under tight capacity.

The construction turns the paper's thesis into geometry. A single severity
score s(x) = |v . x| (v a fixed dense direction) drives a SMOOTH treatment
effect that crosses zero at a threshold:

    delta(x) = DELTA * tanh((s - c) / W_MARGIN)

so the population is dense exactly where allocation is decided: many people
sit near delta = 0, and WHO crosses the capacity price is a fine
distinction. The cap (0.25) is below the responder mass P(delta > 0) = 0.30,
so the price is strictly positive and the margin matters.

    X ~ N(0, I_20);  Y(0) = g + eta;  Y(1) = g + delta + eta
    eta ~ N(0, 0.5^2);  logging T ~ Uniform{0,1}, e = 1/2 known exactly
    g = AMP-scaled smooth nuisance (default AMP = 0; see the dial note)

Why this separates the pipelines AT DEPLOYMENT, not just in fit quality:
a two-stage method thresholds its NOISY estimate of delta at an LP price
fit on the training split. With a dense margin, estimate noise spills the
deployed allocation past the cap (observed: per-arm MLP deploys ~0.27 on a
0.25 cap even with a 0.92 buffered LP), and the queue converts a persistent
overshoot into waits that grow with the horizon — 10-100x the end-to-end
policy's. End-to-end training sees the price during learning and deploys
with the buffered LP on its own scores, which concentrates its allocation
under the cap; its waits stay near zero.

Per-baseline failure taxonomy (each fails for its own stated reason):

    S2-linear / S2-lasso / S2-dr(lasso): delta is EVEN in v . x, so
        cov(x, delta) = 0 — the fitted arm difference is flat, the LP
        prices the arm out, and the deployed allocation is ~0. Blind at
        every N by geometry.
    S2-tree (depth 5): axis-aligned splits cannot express |v . x| along a
        dense direction; treats half-blind.
    S2-knn: neighborhood averaging attenuates delta toward zero;
        under-allocates scarce capacity and captures a fraction of the
        achievable value.
    S2-mlp (capacity-matched: same trunk, steps, lr as the e2e model):
        competitive in fit, but margin noise makes its deployment
        overshoot the cap — the queue does the rest. A 0.92 cap buffer
        does not repair it (robustness row in the probe).
    F / Gs / Alt: trained against the prices, deploy under cap, waits
        near zero, and the best value among learnable baselines.

Nuisance dial note: AMP > 0 adds a smooth outcome-level term shared by both
arms. Probing showed it is NOT a lever in our favour: regression treats
smooth nuisance as learnable signal while IPW pays for it in gradient
variance, so the capacity-matched control overtakes end-to-end at AMP >= 1.
The default is AMP = 0 and the dial is reported as the honest boundary of
the method's advantage, not hidden.
"""

import numpy as np

# ---- fixed structural constants (one world) --------------------------------
D = 20
T_ARMS = 2
DELTA = 1.0          # benefit for responders
EPS = 0.5            # harm from treating a non-responder
SIGMA_Y = 0.5
W_MARGIN = 0.35      # width of the dense decision margin around the threshold
B = np.array([1.0, 0.25])
N_EVAL = 4000
AMP_DEFAULT = 0.0

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


def true_delta(X):
    """Smooth effect, dense at the decision margin: many people sit near
    delta = 0, so WHO crosses the capacity price is decided by fine
    distinctions — the regime where deploying noisy estimates against a
    price overshoots the caps."""
    s = np.abs(X @ _v)
    return DELTA * np.tanh((s - _C[0]) / W_MARGIN)


def _potential_outcomes(X, amp, rng):
    N = X.shape[0]
    g = _nuisance(X, amp)
    Y_pot = np.empty((N, T_ARMS))
    Y_pot[:, 0] = g + rng.normal(0, SIGMA_Y, N)
    Y_pot[:, 1] = g + true_delta(X) + rng.normal(0, SIGMA_Y, N)
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
    eval draw: treat the top-delta people with delta > 0, up to the cap."""
    Yp = eval_data["Y_pot"]
    d = true_delta(eval_data["X"])
    N = len(d)
    arm_of = np.zeros(N, dtype=int)
    idx = np.argsort(-d)[: int(B[1] * N)]
    arm_of[idx[d[idx] > 0]] = 1
    return float(Yp[np.arange(N), arm_of].mean())

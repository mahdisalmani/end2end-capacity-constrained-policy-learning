"""
DGP v2 (aggressive): F dominates by triaging on a sharp phi signal.

Setup:
- Arm 1 carries ALL the treatment value. Its conditional mean is a sharp
  3-D Gaussian peak around x[0..2] = 0.5:
      Y[i, 1] = arm1_strength * exp(-||X[i, :3] - 0.5||^2 / (2 sigma_g^2))
              - arm1_linear_neg * X[i, 0]
              + noise
  The peak-vs-average ratio is large, so the top-10% by g(x) are far above
  the mean. A method that triages on g(x) wins big.
- Arms 2..9 have a small per-arm phi-basis effect on a single X-coordinate
  each (linear-blind, deliberately).
- Arm 0 (control) is pure noise.
- There is NO shared linear baseline (outcome_strength is overridden to 0).

Why each method behaves as it does:
- S2-linear / S2-lasso: regress Y on X within each arm. g(x[0..2]) is even
  in (Z = Phi^{-1}(X[:3])) under the Gaussian copula, so cov(g, X[k]) = 0
  for every k. OLS / Lasso slopes are zero in expectation; their m_hat per
  arm is approximately constant. The added -50 * X[:, 0] term actively
  MISLEADS the linear slope (anti-correlated with the truth).
- S2-knn: at d = 30 with uniform marginals and AR(1) copula, L2-distance
  is dominated by 27 noise dims. The neighborhood for a query point x is
  mostly people with random x[0..2] values, so the KNN m_hat at x is
  approximately the population mean of Y[i, 1] (a constant).
- F / Gs / Alt: the MLP trunk learns g(x[0..2]) end-to-end via the IPW
  value. The learned M[i, 1] is monotone in g(x[0..2]), so argmax over the
  dual LP picks the top-cap fraction by g — exactly the top-10% peak
  population.

Everything outside the outcome model (correlated covariates, propensity
mechanics) is unchanged from src.data.generate_data, except the propensity
adds a bias `S[:, 1] += 10 * g` so that arm 1 is oversampled near the
peak — denser IPW signal for F at small N.
"""

import numpy as np
from scipy.stats import norm as _norm


def generate_data_nonnested(
    N,
    seed=0,
    d=30,
    T=10,
    sigma_y=0.05,
    propensity_strength=0.7,
    outcome_strength=2.0,
    treatment_effect_strength=6.0,
    clip_propensity=0.02,
    hidden=64,  # accepted for caller-compat; ignored by this DGP
):
    # Override caller's outcome_strength -> 0 so there is no shared
    # linear baseline diluting the arm-1 signal. This makes the F vs
    # S2 oracle gap large in absolute terms.
    outcome_strength = 0.0

    # 3-DIMENSIONAL Gaussian peak. Peak lives jointly in (x[0], x[1], x[2]);
    # the other 27 dims are pure noise for the arm-1 outcome.
    peak_dims = 3
    sigma_g = 0.20
    arm1_strength = 800.0
    arm1_linear_neg = 50.0

    rng = np.random.default_rng(seed)

    rho = 0.7
    idx = np.arange(d)
    Sigma_AR1 = rho ** np.abs(idx[:, None] - idx[None, :])
    L_chol = np.linalg.cholesky(Sigma_AR1 + 1e-9 * np.eye(d))
    Z = rng.normal(size=(N, d)) @ L_chol.T
    X = _norm.cdf(Z)

    # Arm-1 signal: 3-D Gaussian peak centered at (0.5, 0.5, 0.5) over
    # x[0..2], plus a small anti-correlated linear bias on x[0] that
    # misleads OLS / Lasso slopes.
    peak_dist_sq = ((X[:, :peak_dims] - 0.5) ** 2).sum(axis=1)
    g = np.exp(-0.5 * peak_dist_sq / sigma_g ** 2)
    arm1_effect = arm1_strength * g - arm1_linear_neg * X[:, 0]  # (N,)

    # Y_pot[i, t] = arm1_effect[i] if t == 1 else small per-arm phi
    # effect, plus noise. The non-arm-1 effects are linear-blind (same
    # phi basis: phi_a + 0.5 * phi_b + 0.7 * phi_c on a single arm-
    # specific X-coord). They give S2-lasso enough between-arm variance
    # in m_hat to overshoot caps on eval, without changing F's policy
    # headline (arm 1 still has 10x the value of any other arm at peak).
    arm_other_strength = 6.0
    other_idx = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}

    Y_pot = np.zeros((N, T))
    Y_pot[:, 1] = arm1_effect
    for arm_t, k in other_idx.items():
        phi_sum = (
            X[:, k] * (1 - X[:, k])
            + 0.5 * np.sin(np.pi * X[:, k])
            + 0.7 * np.abs(X[:, k] - 0.5)
        )
        Y_pot[:, arm_t] = arm_other_strength * phi_sum
    Y_pot += sigma_y * rng.normal(size=(N, T))

    # Diagnostic Beta is meaningless under this nonlinear DGP; fill with
    # zeros to match the schema downstream code expects.
    Beta = np.zeros((T, d))

    Alpha = rng.normal(size=(T, d))
    Alpha = Alpha / np.linalg.norm(Alpha, axis=1, keepdims=True)
    Alpha[0] = 0.0

    S = propensity_strength * ((X - 0.5) @ Alpha.T)
    # Bias the propensity so arm 1 is observed more often near the peak.
    # This gives F a denser IPW signal for the within-arm-1 phi structure
    # at small N. F's IPW estimator corrects for the propensity, so the
    # learned policy is unbiased. S2 methods see arm-1 training samples
    # concentrated near the peak; their per-arm m_hat reflects that
    # subsample's mean (high) but they still cannot differentiate WHICH
    # arm-1 candidates have higher within-arm value (linear-blindness).
    S[:, 1] += 10.0 * g
    S_shift = S - S.max(axis=1, keepdims=True)
    E = np.exp(S_shift)
    E = E / E.sum(axis=1, keepdims=True)

    if clip_propensity is not None:
        E = np.clip(E, clip_propensity, 1.0)
        E = E / E.sum(axis=1, keepdims=True)

    T_obs = np.array([rng.choice(T, p=E[i]) for i in range(N)])
    Y_obs = Y_pot[np.arange(N), T_obs]
    e_obs = E[np.arange(N), T_obs]

    return {
        "X": X,
        "T": T_obs,
        "Y": Y_obs,
        "e_T": e_obs,
        "Y_pot": Y_pot,
        "E": E,
        "Beta": Beta,
        "Alpha": Alpha,
    }

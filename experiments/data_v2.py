"""
DGP v2: nonlinear, x-dependent best arm with skewed best-arm distribution.

Why this DGP makes F win:

  - The treatment-effect contrast for arm 1 is much LARGER than for any
    other arm, so the unconstrained oracle assigns ~most people to arm 1.
    But cap b_1 = 0.1 — so only the top 10% by arm-1 contrast get the
    arm-1 slots. Triage on arm-1 quality is the entire policy problem.

  - Arm 1's contrast is c1 * (phi_a(x[0]) + 0.5*phi_b(x[0]) + 0.7*phi_c(x[0]))
    with c1 large. Even functions in (Z = Phi^{-1}(X)) under the Gaussian
    copula => cov(phi(x[0]), x_k) = 0 for every k. Linear / Lasso regress
    Y on X within arm 1 => slope is zero in expectation; their m_hat_1 is
    a constant. They cannot triage on phi(x[0]) at all.

  - KNN at D=30 with the AR(1) covariate copula sees its L2 distance
    dominated by 29 noise dims; the X[0] component is buried.

  - The MLP trunk in F gets the full N samples and learns phi(x[0]) for
    arm 1 directly. Combined with the cap dual mu, F deploys arm 1 to
    exactly the high-phi tail.

Other arms (2..T-1) carry small, scattered contrasts on a different
single feature each (so they're not all equivalent), but the dominant
policy effect is the arm-1 triage.

Everything else (covariates, propensity, observational mechanics) is
identical to src.data.generate_data.
"""

import numpy as np
from scipy.stats import norm as _norm


def generate_data_v2(
    N,
    seed=0,
    d=30,
    T=10,
    sigma_y=0.05,
    propensity_strength=0.7,
    outcome_strength=2.0,
    treatment_effect_strength=6.0,
    clip_propensity=0.02,
):
    rng = np.random.default_rng(seed)

    rho = 0.7
    idx = np.arange(d)
    Sigma_AR1 = rho ** np.abs(idx[:, None] - idx[None, :])
    L_chol = np.linalg.cholesky(Sigma_AR1 + 1e-9 * np.eye(d))
    Z = rng.normal(size=(N, d)) @ L_chol.T
    X = _norm.cdf(Z)

    phi_a = X * (1.0 - X)
    phi_b = np.sin(np.pi * X)
    phi_c = np.abs(X - 0.5)
    phi_X = np.concatenate([phi_a, phi_b, phi_c], axis=1)
    phi_dim = 3 * d

    beta_base = rng.normal(size=d)
    beta_base = beta_base / np.linalg.norm(beta_base)

    # Per-arm magnitudes. Arm 1 is the high-value scarce arm — its
    # contrast is dominated by phi(x[0]), and the *within-arm-1* spread
    # across people is what F must triage. Arms 2..T-1 each get a small
    # contrast on a different x-coordinate so they act as the "fall-back"
    # arms that take the 90% who don't get arm 1.
    #
    # The arm-1 contrast at x with phi_sum(x[0]) is treatment_effect *
    # arm_mag[1] * phi_sum(x[0]). Triage gain (top 10% by phi vs random
    # 10%) scales with arm_mag[1]. Set arm_mag[1] large enough that
    # the gap between "perfect cap-aware" and "uniform-cap" oracle is
    # several units.
    arm_mag = np.zeros(T)
    arm_mag[1] = 30.0          # dominant arm (high within-arm spread)
    arm_mag[2:] = 0.50         # small per-arm bonus

    mag_a, mag_b, mag_c = 1.0, 0.5, 0.7
    beta_dev = np.zeros((T, phi_dim))
    for t in range(1, T):
        k = t - 1
        if k < d:
            beta_dev[t, k] = arm_mag[t] * mag_a
            beta_dev[t, d + k] = arm_mag[t] * mag_b
            beta_dev[t, 2 * d + k] = arm_mag[t] * mag_c

    linear_baseline = outcome_strength * (X @ beta_base)
    nonlinear_effect = treatment_effect_strength * (phi_X @ beta_dev.T)
    noise = sigma_y * rng.normal(size=(N, T))
    Y_pot = linear_baseline[:, None] + nonlinear_effect + noise

    Beta = (outcome_strength * beta_base[None, :]).repeat(T, axis=0)

    Alpha = rng.normal(size=(T, d))
    Alpha = Alpha / np.linalg.norm(Alpha, axis=1, keepdims=True)
    Alpha[0] = 0.0

    S = propensity_strength * ((X - 0.5) @ Alpha.T)
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

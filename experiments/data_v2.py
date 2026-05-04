"""
DGP v2 (aggressive): F dominates by triaging on a sharp phi signal.

Setup:
  - Arm 1 carries ALL the treatment value. Its conditional mean is a
    sharp Gaussian peak around x[0] = 0.5:
        Y[i, 1] = arm1_strength * exp(-((X[i, 0] - 0.5) / sigma_g)^2 / 2) + noise
    The peak vs. average ratio is large, so the top-10% by phi(x[0]) are
    far above the mean. A method that triages on phi(x[0]) wins big.
  - Arms 2..T-1 have zero treatment effect (Y = noise only).
  - Arm 0 (control) also has zero effect.
  - There is NO linear baseline (outcome_strength is overridden to 0).

Why each method behaves as it does:
  - S2-linear / S2-lasso: regress Y on X within each arm. The Gaussian
    peak g(x[0]) is even in (Z = Phi^{-1}(X[0])) under the Gaussian
    copula, so cov(g, X[k]) = 0 for every k. OLS / Lasso slopes are
    zero in expectation; their m_hat per arm is approximately constant
    (just the per-arm mean). The LP equilibrates the constants and the
    deployed policy assigns *random* 10% of the population to arm 1.
    Random 10% of people are mostly far from x[0] = 0.5, so they get
    near-zero Y[i, 1]. Their oracle outcome is roughly
    (population mean of g) * arm1_strength.

  - S2-knn: at D = 30 with uniform marginals and AR(1) copula, the
    L2-distance is dominated by 29 noise dims. The neighborhood for
    a query point x is mostly people with random x[0] values, so the
    KNN m_hat at x is approximately the population mean of Y[i, 1]
    (a constant). Same fate as linear / lasso.

  - F: the MLP trunk learns g(x[0]) end-to-end via the IPW value. The
    learned M[i, 1] is monotone in g(x[0]), so argmax over the dual
    LP picks the top-cap fraction by g — exactly the top-10% peak
    population. F's oracle outcome on arm 1 is ~peak * arm1_strength.

Numerical example (10k eval samples, sigma_g = 0.07, arm1_strength = 200):
  - oracle_no_cap (everyone on arm 1):     ~ population mean of g * 200
  - random over arms 1..9 (S2 ceiling):    ~ that mean / 9
  - cap-aware perfect (F ceiling):          ~ peak * 0.1 ~ 20
  Gap is several units of oracle outcome, plus F's deterministic-LP
  deployment matches caps tightly so wait time also stays low.

Everything outside the outcome model (correlated covariates, propensity
mechanics) is unchanged from src.data.generate_data.
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
    # Override caller's outcome_strength -> 0 so there is no shared
    # linear baseline diluting the arm-1 signal. This makes the F vs
    # S2 oracle gap large in absolute terms.
    outcome_strength = 0.0

    # Sharpness of the arm-1 phi peak (sigma in normalized space).
    # Smaller -> sharper -> bigger top-vs-mean gap, but harder to learn
    # at small N because fewer training samples land in the peak.
    sigma_g = 0.20
    arm1_strength = 500.0
    # Anti-correlated linear bias inside the arm-1 outcome. Within-arm
    # OLS / Lasso slope is dominated by this term (the gaussian peak is
    # linear-blind), so S2-linear / S2-lasso end up routing low-x[0]
    # people to arm 1 -- exactly the people who are FAR from the peak,
    # so their oracle outcome on arm 1 collapses. Magnitude tuned
    # much smaller than arm1_strength so the actual best arm at
    # x[0]=0.5 is still arm 1.
    arm1_linear_neg = 50.0

    rng = np.random.default_rng(seed)

    rho = 0.7
    idx = np.arange(d)
    Sigma_AR1 = rho ** np.abs(idx[:, None] - idx[None, :])
    L_chol = np.linalg.cholesky(Sigma_AR1 + 1e-9 * np.eye(d))
    Z = rng.normal(size=(N, d)) @ L_chol.T
    X = _norm.cdf(Z)

    # Arm-1 signal: sharp Gaussian peak in x[0] around 0.5, plus an
    # anti-correlated linear term that misleads OLS / Lasso slopes.
    g = np.exp(-0.5 * ((X[:, 0] - 0.5) / sigma_g) ** 2)
    arm1_effect = arm1_strength * g - arm1_linear_neg * X[:, 0]   # (N,)

    # Y_pot[i, t] = arm1_effect[i] if t == 1 else small per-arm phi
    # effect, plus noise. The non-arm-1 effects are linear-blind (same
    # phi basis: phi_a + 0.5 * phi_b + 0.7 * phi_c on a single
    # arm-specific X-coord). They give S2-lasso enough between-arm
    # variance in m_hat to overshoot caps on eval (the v1 pathology),
    # without changing F's policy headline (arm 1 still has 10x the
    # value of any other arm at its peak).
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
    # that are concentrated near the peak; their per-arm m_hat reflects
    # that subsample's mean (which is high) but they still cannot
    # differentiate WHICH arm-1 candidates have higher within-arm value
    # because of linear-blindness.
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

"""
Inner layer for the convex Lagrangian dual G(mu).

    G(mu) = (tau / N) * sum_i log sum_t exp((m_{t,i} - mu_t) / tau) + b . mu

G is the entropy-smoothed dual of the capacity-constrained allocation LP and
relates to F (see `inner_F`) by G(mu) = F(mu) + (tau/N) sum_i H(sigma_i(mu)),
so F <= G <= F + tau*log(T).

Two gradient paths are provided:
  - "G"  : CVXPYLayer (diffcp implicit differentiation). Exact but slow, and
           the compiled layer is size-locked to the training (N, T).
  - "Gs" : scipy L-BFGS-B forward + IFT backward via `inner_common` — same
           machinery as F, no cvxpy in the loop, works at any N.

Because G's stationarity condition is  b_t = mean_i sigma_{t,i}  on inactive
coordinates, the allocation induced by mu*_G is feasible-in-expectation by
construction: binding arms sit exactly at capacity.
"""

import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from .inner_common import ImplicitMu, solve_inner_lbfgsb

# L-BFGS-B iteration cap used by the scipy paths.
G_MAXITER = 500


def build_mu_layer_G(N, T, tau, b):
    M_param = cp.Parameter((N, T))
    mu_var = cp.Variable(T, nonneg=True)

    ones_col = np.ones((N, 1))
    U = M_param - ones_col @ cp.reshape(mu_var, (1, T), order="F")

    obj = (tau / N) * cp.sum(cp.log_sum_exp(U / tau, axis=1)) + b @ mu_var
    prob = cp.Problem(cp.Minimize(obj))

    assert prob.is_dpp(), "G is not DPP — cvxpylayers will refuse"

    return CvxpyLayer(prob, parameters=[M_param], variables=[mu_var])


# Module-level cache of the CVXPYLayer parametrized for the training (N, T).
MU_LAYER_G = None


def initialize_G_layer(N, T, tau, b):
    """Build and cache the CVXPYLayer for the given training (N, T, tau, b)."""
    global MU_LAYER_G
    MU_LAYER_G = build_mu_layer_G(N, T, tau, b)
    print("[G] CVXPYLayer initialized.")


def mu_of_M_G(M):
    """M: (N, T) torch -> mu*: (T,) torch. Gradients flow via diffcp."""
    if MU_LAYER_G is None:
        raise RuntimeError("Call initialize_G_layer(N, T, tau, b) first.")

    mu_star, = MU_LAYER_G(
        M,
        solver_args={
            "solve_method": "SCS",
            "eps": 1e-4,
            "acceleration_lookback": 10,
            "acceleration_interval": 10,
            "max_iters": 500,
        },
    )
    return mu_star


def _G_torch(mu, M, b, tau):
    U = (M - mu.unsqueeze(0)) / tau
    return tau * torch.logsumexp(U, dim=1).mean() + (mu * b).sum()


def _solve_G_inner(M_t, b_t, tau, mu_init=None, tol=1e-10, maxiter=G_MAXITER):
    """argmin_{mu >= 0} G(mu; M) via scipy L-BFGS-B. Returns (mu_star, info)."""
    return solve_inner_lbfgsb(_G_torch, M_t, b_t, tau,
                              mu_init=mu_init, tol=tol, maxiter=maxiter)


def solve_G_scipy(M_t, b, tau, mu_init=None, tol=1e-10, maxiter=G_MAXITER):
    """Forward-only G solve at arbitrary N. No gradients."""
    mu_star, _ = _solve_G_inner(M_t, b, tau, mu_init=mu_init,
                                tol=tol, maxiter=maxiter)
    return mu_star


# Warm-start cache for the scipy + IFT path ("Gs").
_G_STATE = {"mu_warm": None, "last_info": None}


def reset_G_state():
    _G_STATE["mu_warm"] = None
    _G_STATE["last_info"] = None


def mu_of_M_G_scipy(M, b, tau):
    """M: (N, T) -> mu*: (T,). Gradients via IFT on G(mu). No cvxpy."""
    return ImplicitMu.apply(
        M,
        b if torch.is_tensor(b) else torch.tensor(b),
        float(tau),
        _G_torch,
        _G_STATE,
        G_MAXITER,
    )

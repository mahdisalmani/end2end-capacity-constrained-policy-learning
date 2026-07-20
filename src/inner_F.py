"""
Inner layer for the literal non-convex F(mu).

    F(mu) = (1/N) sum_i sum_t sigma_{t,i}(mu) * (m_{t,i} - mu_t) + b . mu

where sigma_i(mu) = softmax((M_i - mu) / tau). Non-convex in mu because the
softmax weights depend on mu. Forward/backward are the shared L-BFGS-B +
implicit-function-theorem machinery in `inner_common`.

Note: unlike G (see `inner_G`), the stationary point of F does NOT satisfy
exact complementary slackness for the capacity constraints — at finite tau
the induced allocation mean_i sigma_{t,i} can differ from b_t on arms with
mu_t > 0 (it approaches b_t only as tau -> 0).
"""

import torch

from .inner_common import ImplicitMu, solve_inner_lbfgsb

# L-BFGS-B iteration cap used by the training-time layer.
F_MAXITER = 200


def _F_torch(mu, M, b, tau):
    """F(mu; M). mu: (T,), M: (N, T). Returns a scalar torch tensor."""
    U = (M - mu.unsqueeze(0)) / tau
    sigma = torch.softmax(U, dim=1)
    V = M - mu.unsqueeze(0)
    return (sigma * V).sum(dim=1).mean() + (mu * b).sum()


def _solve_F_inner(M_t, b_t, tau, mu_init=None, tol=1e-10, maxiter=F_MAXITER):
    """argmin_{mu >= 0} F(mu; M) via scipy L-BFGS-B. Returns (mu_star, info)."""
    return solve_inner_lbfgsb(_F_torch, M_t, b_t, tau,
                              mu_init=mu_init, tol=tol, maxiter=maxiter)


# Module-level state: warm start across consecutive forward calls and the most
# recent solver info for diagnostics.
_F_STATE = {"mu_warm": None, "last_info": None}


def reset_F_state():
    _F_STATE["mu_warm"] = None
    _F_STATE["last_info"] = None


def mu_of_M_F(M, b, tau):
    """M: (N, T) -> mu*: (T,). Gradients via IFT on F."""
    return ImplicitMu.apply(
        M,
        b if torch.is_tensor(b) else torch.tensor(b),
        float(tau),
        _F_torch,
        _F_STATE,
        F_MAXITER,
    )

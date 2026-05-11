"""
Inner layer for the convex Lagrangian dual G(mu).

    G(mu) = (tau / N) * sum_i log sum_t exp((m_{t,i} - mu_t) / tau) + b . mu

Solved as a CVXPYLayer for the training-time gradient path, and via scipy
L-BFGS-B for forward-only evaluation at arbitrary N (used by μ-on-eval
variants where the cached CVXPYLayer is size-locked to training N).
"""

import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from scipy.optimize import minimize


def build_mu_layer_G(N, T, tau, b):
    M_param = cp.Parameter((N, T))
    mu_var = cp.Variable(T, nonneg=True)

    ones_col = np.ones((N, 1))
    U = M_param - ones_col @ cp.reshape(mu_var, (1, T))

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


def _solve_G_inner(M_t, b_t, tau, mu_init=None, tol=1e-10, maxiter=500):
    """argmin_{mu >= 0} G(mu; M) via scipy L-BFGS-B. Returns (mu_star, info).

    Same shape as `_solve_F_inner`; used by the scipy+IFT implementation
    of mu_G and by the forward-only `solve_G_scipy` wrapper below.
    """
    T_ = M_t.shape[1]

    mu0 = np.zeros(T_) if mu_init is None else mu_init.detach().cpu().numpy()
    mu0 = np.maximum(mu0, 0.0)

    M_det = M_t.detach()
    b_det = b_t.detach() if torch.is_tensor(b_t) else torch.tensor(b_t)

    def fg(mu_np):
        with torch.enable_grad():
            mu = torch.tensor(mu_np, dtype=M_det.dtype, requires_grad=True)
            Gv = _G_torch(mu, M_det, b_det, float(tau))
            g, = torch.autograd.grad(Gv, mu)
        return float(Gv.item()), g.detach().numpy().astype(np.float64)

    res = minimize(
        fg, mu0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * T_,
        options={"ftol": tol, "gtol": tol, "maxiter": maxiter},
    )

    mu_star = torch.tensor(np.maximum(res.x, 0.0), dtype=M_t.dtype)
    info = {
        "fun": res.fun,
        "nit": res.nit,
        "gnorm": float(np.max(np.abs(res.jac))),
        "msg": res.message,
    }
    return mu_star, info


def solve_G_scipy(M_t, b, tau, mu_init=None, tol=1e-10, maxiter=500):
    """Forward-only G solve at arbitrary N. No gradients."""
    b_t = b if torch.is_tensor(b) else torch.tensor(b)
    mu_star, _ = _solve_G_inner(M_t, b_t, tau, mu_init=mu_init,
                                tol=tol, maxiter=maxiter)
    return mu_star


# === scipy + IFT implementation of mu_G ====================================
# Mirrors src/inner_F.py's ImplicitMuF: forward via scipy L-BFGS-B with
# warm-starts across consecutive training steps; backward via the implicit
# function theorem on the KKT system. Active set (mu_t = 0) is handled by
# augmenting J with multiplier rows/cols, exactly as in F.
#
# Provides the same speed profile as F's path -- no cvxpy / SCS / diffcp,
# no N-locked CVXPYLayer to rebuild.

_G_STATE = {"mu_warm": None, "last_info": None}


def reset_G_state():
    _G_STATE["mu_warm"] = None
    _G_STATE["last_info"] = None


class ImplicitMuG(torch.autograd.Function):

    @staticmethod
    def forward(ctx, M, b, tau):
        mu_init = _G_STATE["mu_warm"]
        mu_star, info = _solve_G_inner(M, b, float(tau), mu_init=mu_init)

        _G_STATE["mu_warm"] = mu_star.detach().clone()
        _G_STATE["last_info"] = info

        ctx.save_for_backward(M.detach(), mu_star.detach(), b.detach())
        ctx.tau = float(tau)
        return mu_star

    @staticmethod
    def backward(ctx, grad_mu):
        M, mu_star, b = ctx.saved_tensors
        tau = ctx.tau

        T_ = mu_star.numel()
        eps = 1e-7

        active = mu_star < eps
        A_idx = active.nonzero(as_tuple=True)[0]
        n_A = int(active.sum())
        n_sys = T_ + n_A

        M_d = M.clone().detach().requires_grad_(True)
        mu_d = mu_star.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            Gv = _G_torch(mu_d, M_d, b, tau)
            g_mu, = torch.autograd.grad(Gv, mu_d, create_graph=True)

            H = torch.zeros(T_, T_, dtype=M.dtype)
            for k in range(T_):
                row, = torch.autograd.grad(g_mu[k], mu_d, retain_graph=True)
                H[k] = row

            J = torch.zeros(n_sys, n_sys, dtype=M.dtype)
            J[:T_, :T_] = H + 1e-6 * torch.eye(T_, dtype=M.dtype)

            for j, a in enumerate(A_idx):
                a = a.item()
                J[a, T_ + j] = -1.0
                J[T_ + j, a] = 1.0

            dFp_dM = torch.zeros(n_sys, M_d.numel(), dtype=M.dtype)
            for k in range(T_):
                gk, = torch.autograd.grad(g_mu[k], M_d, retain_graph=True)
                dFp_dM[k] = gk.reshape(-1)

            du_dM = torch.linalg.solve(J, -dFp_dM)
            grad_M = (du_dM[:T_].t() @ grad_mu).reshape(M.shape)

        return grad_M, None, None


def mu_of_M_G_scipy(M, b, tau):
    """M: (N, T) -> mu*: (T,). Gradients via IFT on G(mu). No cvxpy."""
    return ImplicitMuG.apply(
        M,
        b if torch.is_tensor(b) else torch.tensor(b),
        float(tau),
    )

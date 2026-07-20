"""
Shared machinery for the inner shadow-price layers.

Both inner objectives (the non-convex F in `inner_F.py` and the convex dual
G in `inner_G.py`) are minimized over mu >= 0 the same way:

  forward : scipy L-BFGS-B with box bounds and warm starts, calling a small
            torch graph for (value, gradient);
  backward: implicit-function theorem on the KKT system of
            argmin_{mu >= 0} obj(mu; M), with the active set {t : mu_t = 0}
            handled by augmenting the Hessian with multiplier rows/columns
            (the saddle-point system of Final Report, Appendix B).

This module holds the single implementation of both passes; `inner_F` and
`inner_G` supply only their objective functions and warm-start caches.
"""

import numpy as np
import torch
from scipy.optimize import minimize

# Active-set threshold: mu_t below this counts as an active bound (mu_t = 0).
ACTIVE_EPS = 1e-7
# Ridge added to the Hessian block of the KKT matrix for numerical stability.
KKT_RIDGE = 1e-6


def _as_detached_tensor(b, like):
    if torch.is_tensor(b):
        return b.detach()
    return torch.tensor(b, dtype=like.dtype)


def solve_inner_lbfgsb(objective, M_t, b_t, tau, mu_init=None,
                       tol=1e-10, maxiter=200):
    """argmin_{mu >= 0} objective(mu; M) via scipy L-BFGS-B.

    `objective(mu, M, b, tau)` must be a torch scalar function. M and b are
    detached here so no autograd tape from the caller leaks into the solve;
    gradients w.r.t. mu are produced on a small local graph.

    Returns (mu_star, info) with mu_star a detached torch tensor and info a
    dict of solver diagnostics.
    """
    T_ = M_t.shape[1]

    mu0 = np.zeros(T_) if mu_init is None else mu_init.detach().cpu().numpy()
    mu0 = np.maximum(mu0, 0.0)

    M_det = M_t.detach()
    b_det = _as_detached_tensor(b_t, M_t)

    def fg(mu_np):
        with torch.enable_grad():
            mu = torch.tensor(mu_np, dtype=M_det.dtype, requires_grad=True)
            val = objective(mu, M_det, b_det, float(tau))
            g, = torch.autograd.grad(val, mu)
        return float(val.item()), g.detach().numpy().astype(np.float64)

    res = minimize(
        fg,
        mu0,
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


def ift_vjp(objective, M, mu_star, b, tau, grad_mu):
    """VJP (d mu*/dM)^T grad_mu via the IFT on the augmented KKT system.

    With A = {t : mu*_t = 0} the active set and H = grad^2_mu obj, solve

        [ H + ridge*I   -E_A^T ] [ d mu*/dM     ]     [ grad^2_{mu,M} obj ]
        [ E_A            0     ] [ d lambda_A/dM]  = -[ 0                 ],

    then contract the first |T| rows with grad_mu. Active coordinates get
    d mu*_t/dM = 0 (enforced by the bottom block), matching strict
    complementarity.
    """
    T_ = mu_star.numel()

    active = mu_star < ACTIVE_EPS
    A_idx = active.nonzero(as_tuple=True)[0]
    n_A = int(active.sum())
    n_sys = T_ + n_A

    M_d = M.clone().detach().requires_grad_(True)
    mu_d = mu_star.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        val = objective(mu_d, M_d, b, tau)
        g_mu, = torch.autograd.grad(val, mu_d, create_graph=True)

        H = torch.zeros(T_, T_, dtype=M.dtype)
        for k in range(T_):
            row, = torch.autograd.grad(g_mu[k], mu_d, retain_graph=True)
            H[k] = row

        J = torch.zeros(n_sys, n_sys, dtype=M.dtype)
        J[:T_, :T_] = H + KKT_RIDGE * torch.eye(T_, dtype=M.dtype)

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

    return grad_M


class ImplicitMu(torch.autograd.Function):
    """Generic mu*(M) layer: L-BFGS-B forward, IFT backward.

    apply(M, b, tau, objective, state, maxiter) -> mu* of shape (T,).

    `objective` is the inner objective function, `state` a per-objective dict
    {"mu_warm": ..., "last_info": ...} used to warm-start consecutive solves,
    `maxiter` the L-BFGS-B iteration cap. Only M receives a gradient.
    """

    @staticmethod
    def forward(ctx, M, b, tau, objective, state, maxiter):
        mu_star, info = solve_inner_lbfgsb(
            objective, M, b, float(tau),
            mu_init=state["mu_warm"], maxiter=maxiter,
        )

        state["mu_warm"] = mu_star.detach().clone()
        state["last_info"] = info

        ctx.save_for_backward(M.detach(), mu_star.detach(),
                              _as_detached_tensor(b, M))
        ctx.tau = float(tau)
        ctx.objective = objective
        return mu_star

    @staticmethod
    def backward(ctx, grad_mu):
        M, mu_star, b = ctx.saved_tensors
        grad_M = ift_vjp(ctx.objective, M, mu_star, b, ctx.tau, grad_mu)
        return grad_M, None, None, None, None, None

"""Numerical checks for the inner shadow-price layers (src/inner_*).

Run directly (no pytest needed):
    python -m tests.test_inner_layers
or via pytest:
    pytest tests/
"""

import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from src.inner_common import ift_vjp  # noqa: E402
from src.inner_F import _F_torch, _solve_F_inner, mu_of_M_F, reset_F_state  # noqa: E402
from src.inner_G import (  # noqa: E402
    _G_torch,
    _solve_G_inner,
    build_mu_layer_G,
    mu_of_M_G_scipy,
    reset_G_state,
)
from src.policy import softmax_policy  # noqa: E402


N, T = 25, 4
TAU = 0.5
B = torch.tensor([0.6, 0.25, 0.2, 0.15])


def _test_matrix():
    rng = np.random.default_rng(0)
    M = torch.tensor(rng.normal(size=(N, T)))
    M[:, 1] += 1.0   # arm 1 attractive -> its cap binds (mu_1 > 0)
    M[:, 3] -= 1.0   # arm 3 unattractive -> mu_3 hits the bound (active set)
    return M


def _solve(objective, M, tau=TAU):
    fn = _solve_F_inner if objective == "F" else _solve_G_inner
    mu, _ = fn(M, B, tau, mu_init=None, tol=1e-12, maxiter=2000)
    return mu


def _fd_vjp(objective, M, g, tau=TAU, h=1e-5):
    out = torch.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Mp = M.clone(); Mp[i, j] += h
            Mm = M.clone(); Mm[i, j] -= h
            out[i, j] = (
                (_solve(objective, Mp, tau) - _solve(objective, Mm, tau))
                / (2 * h) * g
            ).sum()
    return out


def _rel_err(a, b):
    scale = max(a.abs().max().item(), b.abs().max().item(), 1e-12)
    return (a - b).abs().max().item() / scale


def test_fg_entropy_identity():
    """G(mu) - F(mu) == (tau/N) sum_i H(sigma_i)  (Final Report, Prop. 1)."""
    M = _test_matrix()
    rng = np.random.default_rng(1)
    mu = torch.tensor(rng.uniform(0, 0.5, size=T))
    Fv = _F_torch(mu, M, B, TAU)
    Gv = _G_torch(mu, M, B, TAU)
    sigma = torch.softmax((M - mu) / TAU, dim=1)
    H = -(sigma * torch.log(sigma)).sum(dim=1).mean()
    assert abs((Gv - Fv) - TAU * H) < 1e-12
    # and the bound F <= G <= F + tau log T
    assert Fv <= Gv <= Fv + TAU * np.log(T) + 1e-12


def test_implicit_mu_F_backward_matches_finite_differences():
    M = _test_matrix()
    g = torch.tensor(np.random.default_rng(2).normal(size=T))
    reset_F_state()
    M_leaf = M.clone().requires_grad_(True)
    mu = mu_of_M_F(M_leaf, B, TAU)
    vjp, = torch.autograd.grad((mu * g).sum(), M_leaf)
    vjp_fd = _fd_vjp("F", M, g)
    err = _rel_err(vjp, vjp_fd)
    assert err < 2e-3, f"F IFT vs FD rel err {err:.2e}"


def test_implicit_mu_G_backward_matches_finite_differences():
    M = _test_matrix()
    g = torch.tensor(np.random.default_rng(3).normal(size=T))
    reset_G_state()
    M_leaf = M.clone().requires_grad_(True)
    mu = mu_of_M_G_scipy(M_leaf, B, TAU)
    vjp, = torch.autograd.grad((mu * g).sum(), M_leaf)
    vjp_fd = _fd_vjp("G", M, g)
    err = _rel_err(vjp, vjp_fd)
    assert err < 2e-3, f"G IFT vs FD rel err {err:.2e}"


def test_ift_vjp_zero_on_active_set():
    """Active coordinates (mu_t = 0) must have d mu_t / dM = 0."""
    M = _test_matrix()
    mu_star = _solve("G", M)
    active = (mu_star < 1e-7).nonzero(as_tuple=True)[0]
    assert len(active) > 0, "test setup: some arm should be at the bound"
    a = int(active[0])
    g = torch.zeros(T); g[a] = 1.0   # upstream gradient only on the active coord
    vjp = ift_vjp(_G_torch, M, mu_star, B, TAU, g)
    assert vjp.abs().max() < 1e-8


def test_G_complementary_slackness():
    """At mu*_G, binding arms sit exactly at capacity; slack arms below.

    This is G's stationarity b_t = mean_i sigma_{t,i} on inactive coords —
    the allocation of the G-policy is feasible-in-expectation by
    construction. (F has NO such property at finite tau.)
    """
    M = _test_matrix()
    mu = _solve("G", M)
    pi = softmax_policy(M, mu, TAU)
    alloc = pi.mean(dim=0)
    for t in range(T):
        if mu[t] > 1e-6:
            assert abs(alloc[t] - B[t]) < 1e-6, (t, alloc[t], B[t])
        else:
            assert alloc[t] <= B[t] + 1e-6


def test_cvxpylayer_matches_scipy():
    try:
        import cvxpylayers  # noqa: F401
    except ImportError:
        print("  (cvxpylayers not installed — skipping)")
        return
    M = _test_matrix()
    mu_scipy = _solve("G", M)
    layer = build_mu_layer_G(N, T, TAU, B.numpy())
    M_leaf = M.clone().requires_grad_(True)
    mu_cvx, = layer(M_leaf, solver_args={"solve_method": "SCS", "eps": 1e-10,
                                         "max_iters": 100000})
    assert _rel_err(mu_cvx.detach(), mu_scipy) < 1e-6
    g = torch.tensor(np.random.default_rng(4).normal(size=T))
    vjp_cvx, = torch.autograd.grad((mu_cvx * g).sum(), M_leaf)
    reset_G_state()
    M_leaf2 = M.clone().requires_grad_(True)
    mu2 = mu_of_M_G_scipy(M_leaf2, B, TAU)
    vjp_ift, = torch.autograd.grad((mu2 * g).sum(), M_leaf2)
    assert _rel_err(vjp_cvx, vjp_ift) < 5e-3


def test_dual_lp_is_small_tau_limit_of_G():
    """The sample dual LP (S2's price problem) is the tau -> 0 limit of G."""
    from src.s2_dual import solve_dual_lp
    M = _test_matrix()
    mu_lp, _, status, _ = solve_dual_lp(M.numpy(), B.numpy())
    assert status == "optimal"
    mu_g = _solve("G", M, tau=1e-3)
    assert float((torch.tensor(mu_lp) - mu_g).abs().max()) < 5e-2


def _run_all():
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        print(f"[test] {name} ...", flush=True)
        fn()
        print(f"[test] {name} OK")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    sys.exit(_run_all())

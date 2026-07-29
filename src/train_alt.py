"""
Alternating block-coordinate training for the capacity-constrained policy:
the alternating variant Alt of Sec. 3.1 of the paper.

Targets the same Lagrangian as F (`src/inner_F.py`), but instead of
backpropagating through the inner argmin via the implicit function theorem,
it alternates:

  1. Hold mu fixed (treated as a constant), take `inner_freq` gradient
     steps on theta against the outer IPW objective.
  2. Hold theta fixed, refit mu by solving the inner problem with scipy
     L-BFGS-B (same routine F uses, `_solve_F_inner`).

mu is used as a value, never a function of theta — no IFT, no diffcp. This
is the dual-refresh idea of Rodriguez-Diaz et al. (arXiv:2511.04909) applied
to the IPW policy objective: the chain-rule term (dV/dmu)(dmu/dtheta) is
dropped, so refreshes must be frequent enough for mu to track theta.

Per the project's bilevel-eval convention: `mu_final` returned here is part
of the trained policy. Callers must NOT re-solve mu on eval data.
"""

import time

import torch

from .models import MLPScore
from .policy import softmax_policy, ipw_value
from .inner_F import _solve_F_inner
from .train import prepare_torch_training_data, log_training_step


def train_alt(
    train_data,
    D,
    T,
    tau,
    b,
    outer_steps=500,
    inner_freq=5,
    lr=5e-3,
    log_every=20,
    seed=0,
):
    torch.manual_seed(seed)

    X_t, T_t, Y_t, e_T_t = prepare_torch_training_data(train_data)
    N_local = X_t.shape[0]

    model = MLPScore(d_in=D, hidden=64, T=T)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    b_t = b if torch.is_tensor(b) else torch.tensor(b)

    mu = torch.zeros(T, dtype=X_t.dtype)
    mu_warm = None

    history = []
    t0 = time.time()

    for step in range(outer_steps):
        if step % inner_freq == 0:
            with torch.no_grad():
                M_full = model(X_t)
            mu, _ = _solve_F_inner(M_full, b_t, float(tau), mu_init=mu_warm)
            mu = mu.detach()
            mu_warm = mu.clone()

        M = model(X_t)
        pi = softmax_policy(M, mu.detach(), tau)
        V = ipw_value(pi, T_t, Y_t, e_T_t)
        loss = -V

        assert torch.allclose(
            pi.sum(dim=1),
            torch.ones(N_local),
        ), "policy rows !~ 1"

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == outer_steps - 1:
            log_training_step("Alt", step, V, mu, pi, b_t, model, history)

    wall = time.time() - t0
    print(f"[train-Alt] done in {wall:.1f}s")

    with torch.no_grad():
        M_final = model(X_t)
    mu_final, _ = _solve_F_inner(M_final, b_t, float(tau), mu_init=mu_warm)
    mu_final = mu_final.detach()

    return model, mu_final, history

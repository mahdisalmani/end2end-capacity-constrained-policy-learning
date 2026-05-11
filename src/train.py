"""Training loop for the G/F bilevel pipelines."""

import time

import torch

from .models import MLPScore
from .policy import softmax_policy, ipw_value
from .inner_G import mu_of_M_G, mu_of_M_G_scipy, reset_G_state
from .inner_F import mu_of_M_F, reset_F_state


def prepare_torch_training_data(train_data):
    X_t = torch.tensor(train_data["X"])
    T_t = torch.tensor(train_data["T"], dtype=torch.long)
    Y_t = torch.tensor(train_data["Y"])
    e_T_t = torch.tensor(train_data["e_T"])
    return X_t, T_t, Y_t, e_T_t


def make_mu_layer(kind, b, tau):
    """Return a callable M -> mu* for the chosen inner formulation."""
    if kind == "G":
        return mu_of_M_G

    if kind == "Gs":
        reset_G_state()
        b_t = b if torch.is_tensor(b) else torch.tensor(b)
        return lambda M: mu_of_M_G_scipy(M, b_t, tau)

    if kind == "F":
        reset_F_state()
        b_t = b if torch.is_tensor(b) else torch.tensor(b)
        return lambda M: mu_of_M_F(M, b_t, tau)

    raise ValueError(kind)


def train_GF(
    kind,
    train_data,
    D,
    T,
    tau,
    b,
    steps=200,
    lr=5e-3,
    log_every=20,
    seed=0,
):
    torch.manual_seed(seed)

    X_t, T_t, Y_t, e_T_t = prepare_torch_training_data(train_data)
    N_local = X_t.shape[0]

    model = MLPScore(d_in=D, hidden=64, T=T)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    mu_layer = make_mu_layer(kind, b, tau)
    b_t = torch.tensor(b)

    history = []
    t0 = time.time()

    for step in range(steps):
        M = model(X_t)
        mu_star = mu_layer(M)
        pi = softmax_policy(M, mu_star, tau)

        V = ipw_value(pi, T_t, Y_t, e_T_t)
        loss = -V

        assert torch.allclose(
            pi.sum(dim=1),
            torch.ones(N_local),
        ), "policy rows !~ 1"

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                pi_mean = pi.mean(0).detach()
                cap_viol = (pi_mean - b_t).clamp(min=0)

            grad_norm = float(
                sum(
                    p.grad.pow(2).sum()
                    for p in model.parameters()
                    if p.grad is not None
                ).sqrt()
            )

            history.append({
                "step": step,
                "V": float(V.item()),
                "mu": mu_star.detach().numpy().copy(),
                "pi_mean": pi_mean.numpy().copy(),
                "cap_viol_sup": float(cap_viol.max().item()),
                "grad_norm": grad_norm,
            })

            print(
                f"[train-{kind}] step={step:4d}  "
                f"V={V.item(): .4f}  "
                f"mu={mu_star.detach().numpy().round(3)}  "
                f"pi_mean={pi_mean.numpy().round(3)}  "
                f"cap_viol_sup={cap_viol.max().item():.3e}  "
                f"|grad|={grad_norm:.2e}"
            )

    wall = time.time() - t0
    print(f"[train-{kind}] done in {wall:.1f}s")

    # Final mu at the post-update theta, frozen for downstream eval.
    with torch.no_grad():
        M_final = model(X_t)
    mu_final = mu_layer(M_final).detach()

    return model, mu_final, history

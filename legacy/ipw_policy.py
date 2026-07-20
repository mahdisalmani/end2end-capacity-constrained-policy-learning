"""
[LEGACY — original T=3 prototype this repo grew out of; superseded by src/ + main.py. Kept for the README's historical walkthrough.]

Capacity-constrained IPW policy learning with two inner layers, side-by-side.

Pipeline:

    outer:  max_theta   V_IPW(theta) = (1/N) sum_i  pi_{theta, T_i}(X_i) * Y_i / e_{T_i}(X_i)

    inner:  mu_theta = argmin_{mu >= 0}  <objective> depending on m_{t,theta}(X_i)

We implement TWO inner objectives:

    G(mu) = (1/N) sum_i  tau * log sum_t exp((m_{t,i} - mu_t)/tau)  +  sum_t mu_t * b_t
        -> convex in mu, DPP-compliant, solved by a CVXPYLayer (diffcp backward).

    F(mu) = (1/N) sum_i sum_t  sigma_{t,i}(mu) * (m_{t,i} - mu_t)   +  sum_t mu_t * b_t
        -> non-convex in mu (user-literal form), solved by L-BFGS-B inside a
           torch.autograd.Function with an implicit-function-theorem backward.

This file is structured as sequential blocks you can paste into notebook cells.
"""

# === Block 1: imports + config ===============================================
# Math:
#   T = 3 treatments, tau is the policy temperature, b are per-treatment
#   capacity budgets (average policy mass allowed per treatment). Feasibility
#   requires sum_t b_t >= 1 since every row's softmax sums to 1.
#   With b = (1.0, 0.4, 0.4): b_0 is trivially slack, b_1 and b_2 may bind.
import time
import numpy as np
import torch
from torch import nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from scipy.optimize import minimize

torch.set_default_dtype(torch.float64)          # cvxpylayers expects float64
SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)

N = 500
T = 3
TAU = 1.0
B = np.array([1.0, 0.4, 0.4])                   # capacities
assert B.sum() >= 1.0, "capacities must sum to >= 1 for feasibility"
DEVICE = "cpu"                                  # cvxpylayers runs on CPU


# === Block 2: data generation ================================================
# DGP recap:
#   X1, X2 ~ N(0,1) i.i.d.; three potential outcomes
#     Y^t = 0.5 X1 + X2 + (2*1[t=1]-1)*0.25*X1 + (2*1[t=2]-1)*0.25*X2
#   Propensities via softmax over scores (s_0 = 0, s_1 = 0.5 X1 - 0.5 X2,
#     s_2 = -0.5 X1 + 0.5 X2); observed (X, T, Y, e_T) only.
def generate_data(N, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 2))
    X1, X2 = X[:, 0], X[:, 1]
    Y_pot = np.stack([
        0.5*X1 + X2 - 0.25*X1 - 0.25*X2,        # t=0
        0.5*X1 + X2 + 0.25*X1 - 0.25*X2,        # t=1
        0.5*X1 + X2 - 0.25*X1 + 0.25*X2,        # t=2
    ], axis=1)                                  # (N, 3)
    S = np.stack([np.zeros(N), 0.5*X1 - 0.5*X2, -0.5*X1 + 0.5*X2], axis=1)
    E = np.exp(S); E = E / E.sum(axis=1, keepdims=True)    # (N, 3) propensities
    T_obs = np.array([rng.choice(3, p=E[i]) for i in range(N)])
    Y_obs = Y_pot[np.arange(N), T_obs]
    e_obs = E[np.arange(N), T_obs]
    return {"X": X, "T": T_obs, "Y": Y_obs, "e_T": e_obs,
            "Y_pot": Y_pot, "E": E}             # Y_pot and E returned for EVAL only


data = generate_data(N, seed=SEED)
print("[Block 2] data shapes:",
      {k: v.shape for k, v in data.items() if hasattr(v, 'shape')})
print("[Block 2] empirical T proportions:",
      np.bincount(data["T"], minlength=3) / N)
print("[Block 2] mean e_T:",
      [round(data["e_T"][data["T"] == t].mean(), 3) for t in range(3)])


# === Block 3: preprocessing (to torch) =======================================
# Training uses ONLY observed (X, T, Y, e_T).  Y_pot and E are reserved
# strictly for oracle/diagnostic evaluation in Block 9.
X_t   = torch.tensor(data["X"])                 # (N, 2)
T_t   = torch.tensor(data["T"], dtype=torch.long)   # (N,)
Y_t   = torch.tensor(data["Y"])                 # (N,)
e_T_t = torch.tensor(data["e_T"])               # (N,)
print("[Block 3] X", X_t.shape, "T", T_t.shape, "Y", Y_t.shape, "e_T", e_T_t.shape)


# === Block 4: score model m_{t,theta}(x) =====================================
# Shared-trunk MLP with a T-dim head.  Output head zero-init so that at step 0
# all scores equal 0, softmax is uniform, and mu=0 is a trivial warm start.
class MLPScore(nn.Module):
    def __init__(self, d_in=2, hidden=16, T=3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head = nn.Linear(hidden, T)
        nn.init.zeros_(self.head.weight)        # zero-init output
        nn.init.zeros_(self.head.bias)
        self.double()                           # float64 everywhere

    def forward(self, X):                       # X: (N, 2)
        return self.head(self.trunk(X))         # M: (N, T)


# === Block 5a: CVXPYLayer for the convex dual G ==============================
# Build the parametric program
#     min_{mu >= 0}  (1/N) * tau * sum_i log_sum_exp((m_{.,i} - mu)/tau)
#                   + b . mu
# with M (shape (N,T)) as a cp.Parameter.  log_sum_exp is DPP-supported and
# M enters affinely, which keeps the whole problem DPP-compliant.
def build_mu_layer_G(N, T, tau, b):
    M_param = cp.Parameter((N, T))
    mu_var  = cp.Variable(T, nonneg=True)
    ones_col = np.ones((N, 1))
    U = M_param - ones_col @ cp.reshape(mu_var, (1, T))     # (N, T)
    obj = (tau / N) * cp.sum(cp.log_sum_exp(U / tau, axis=1)) + b @ mu_var
    prob = cp.Problem(cp.Minimize(obj))
    assert prob.is_dpp(), "G is not DPP — cvxpylayers will refuse"
    return CvxpyLayer(prob, parameters=[M_param], variables=[mu_var])


MU_LAYER_G = build_mu_layer_G(N, T, TAU, B)


def mu_of_M_G(M):
    """M: (N,T) torch -> mu*: (T,) torch; grads flow M <- mu* via diffcp.

    NOTE: this wraps the size-locked CvxpyLayer built for the training N.
    For evaluation at a different N, use solve_G_scipy (no gradients needed).
    """
    mu_star, = MU_LAYER_G(M, solver_args={"solve_method": "SCS", "eps": 1e-9})
    return mu_star


def _G_torch(mu, M, b, tau):
    """G(mu; M). Convex: tau * mean_i logsumexp((m_{.,i}-mu)/tau) + b.mu."""
    U = (M - mu.unsqueeze(0)) / tau
    return tau * torch.logsumexp(U, dim=1).mean() + (mu * b).sum()


def solve_G_scipy(M_t, b_t=None, tau=None, mu_init=None, tol=1e-10, maxiter=500):
    """Forward-only G solve for arbitrary N (used in eval, no gradients)."""
    if b_t is None: b_t = torch.tensor(B)
    if tau is None: tau = TAU
    T_ = M_t.shape[1]
    mu0 = np.zeros(T_) if mu_init is None else mu_init.detach().cpu().numpy()
    mu0 = np.maximum(mu0, 0.0)
    M_det, b_det = M_t.detach(), b_t.detach()

    def fg(mu_np):
        with torch.enable_grad():
            mu = torch.tensor(mu_np, dtype=M_det.dtype, requires_grad=True)
            Gv = _G_torch(mu, M_det, b_det, float(tau))
            g, = torch.autograd.grad(Gv, mu)
        return float(Gv.item()), g.detach().numpy().astype(np.float64)

    res = minimize(fg, mu0, jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None)] * T_,
                   options={"ftol": tol, "gtol": tol, "maxiter": maxiter})
    return torch.tensor(np.maximum(res.x, 0.0), dtype=M_t.dtype)


# === Block 5b: implicit-diff layer for the literal non-convex F ==============
# F(mu) = (1/N) sum_i sum_t sigma_{t,i}(mu) * (m_{t,i} - mu_t) + b . mu
# Inner solve: L-BFGS-B with bounds mu >= 0, warm-started from previous mu*.
# Backward: implicit-function theorem on  nabla_mu F(mu*, M) = 0 restricted to
#   the inactive set {t : mu_t* > 0}.  Active components have dmu*/dM = 0.
def _F_torch(mu, M, b, tau):
    """F(mu; M). mu: (T,), M: (N,T). Returns scalar torch tensor."""
    U = (M - mu.unsqueeze(0)) / tau                 # (N, T)
    sigma = torch.softmax(U, dim=1)                 # (N, T)
    V = M - mu.unsqueeze(0)                         # (N, T)
    return (sigma * V).sum(dim=1).mean() + (mu * b).sum()


def _solve_F_inner(M_t, b_t, tau, mu_init=None, tol=1e-10, maxiter=200):
    """Solve argmin_{mu>=0} F(mu; M) via L-BFGS-B.

    The inner solve is purely numerical: we use scipy with analytic gradients
    from a small, local torch graph.  We detach M and b so no autograd tape
    from the outer call leaks in — the custom backward in ImplicitMuF is the
    only gradient path from mu_star back to M.
    """
    T_ = M_t.shape[1]
    mu0 = np.zeros(T_) if mu_init is None else mu_init.detach().cpu().numpy()
    mu0 = np.maximum(mu0, 0.0)
    M_det = M_t.detach()
    b_det = b_t.detach()

    def fg(mu_np):
        with torch.enable_grad():                       # override outer no_grad if any
            mu = torch.tensor(mu_np, dtype=M_det.dtype, requires_grad=True)
            Fv = _F_torch(mu, M_det, b_det, tau)
            g, = torch.autograd.grad(Fv, mu)
        return float(Fv.item()), g.detach().numpy().astype(np.float64)

    res = minimize(fg, mu0, jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None)] * T_,
                   options={"ftol": tol, "gtol": tol, "maxiter": maxiter})
    mu_star = torch.tensor(np.maximum(res.x, 0.0), dtype=M_t.dtype)
    return mu_star, {"fun": res.fun, "nit": res.nit,
                     "gnorm": float(np.max(np.abs(res.jac))), "msg": res.message}


# Module-level warm-start cache — mu* at step k seeds the solve at step k+1.
_F_STATE = {"mu_warm": None, "last_info": None}


class ImplicitMuF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, b, tau):                        # tau is a Python float
        mu_init = _F_STATE["mu_warm"]
        mu_star, info = _solve_F_inner(M, b, float(tau), mu_init=mu_init)
        _F_STATE["mu_warm"] = mu_star.detach().clone()
        _F_STATE["last_info"] = info
        ctx.save_for_backward(M.detach(), mu_star.detach(), b.detach())
        ctx.tau = float(tau)
        return mu_star

    @staticmethod
    def backward(ctx, grad_mu):
        # IFT on the stationarity map of F, restricted to inactive coords.
        M, mu_star, b = ctx.saved_tensors
        tau = ctx.tau
        T_ = mu_star.numel()
        eps = 1e-7
        active   = mu_star < eps
        inactive = ~active

        M_d  = M.clone().detach().requires_grad_(True)
        mu_d = mu_star.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            Fv = _F_torch(mu_d, M_d, b, tau)
            g_mu, = torch.autograd.grad(Fv, mu_d, create_graph=True)        # (T,)

            H = torch.zeros(T_, T_, dtype=M.dtype)
            for k in range(T_):
                row, = torch.autograd.grad(g_mu[k], mu_d, retain_graph=True)
                H[k] = row

            lam = torch.zeros(T_, dtype=M.dtype)
            if bool(inactive.any()):
                idx = inactive.nonzero(as_tuple=True)[0]
                Hii = H[idx][:, idx] + 1e-6 * torch.eye(len(idx), dtype=M.dtype)
                gi = grad_mu[idx]
                try:
                    lam[idx] = torch.linalg.solve(Hii, gi)
                except Exception:
                    lam[idx] = torch.linalg.lstsq(Hii, gi).solution

            scalar = (g_mu * lam.detach()).sum()
            grad_M, = torch.autograd.grad(scalar, M_d)
        return -grad_M, None, None


def mu_of_M_F(M):
    """M: (N,T) -> mu*: (T,); grads flow M <- mu* via IFT on ∇F=0."""
    return ImplicitMuF.apply(M, torch.tensor(B), float(TAU))


def reset_F_state():
    _F_STATE["mu_warm"] = None
    _F_STATE["last_info"] = None


# === Block 5c: unified factory ===============================================
def make_mu_layer(kind):
    if kind == "G":
        return mu_of_M_G
    if kind == "F":
        reset_F_state()                         # fresh warm start per training run
        return mu_of_M_F
    raise ValueError(kind)


# === Block 6: softmax policy from (M, mu) ====================================
# pi_{t,i} = softmax_t( (m_{t,i} - mu_t) / tau )
def softmax_policy(M, mu, tau):
    return torch.softmax((M - mu.unsqueeze(0)) / tau, dim=1)


# === Block 7: IPW objective ==================================================
# V_IPW(theta) = (1/N) sum_i  pi_{theta, T_i}(X_i) * Y_i / e_{T_i}(X_i)
def ipw_value(pi, T_obs, Y, e_T):
    pi_t = pi.gather(1, T_obs.unsqueeze(1)).squeeze(1)        # (N,)
    return (pi_t * Y / e_T).mean()


# === Block 8: training loop (run twice: G and F) =============================
def train(kind, steps=200, lr=5e-3, log_every=20, seed=0):
    torch.manual_seed(seed)
    model = MLPScore(d_in=2, hidden=16, T=T)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mu_layer = make_mu_layer(kind)
    b_t = torch.tensor(B)

    history = []
    t0 = time.time()
    for step in range(steps):
        M = model(X_t)                                        # (N, T)
        mu_star = mu_layer(M)                                 # (T,)
        pi = softmax_policy(M, mu_star, TAU)                  # (N, T)
        V = ipw_value(pi, T_t, Y_t, e_T_t)
        loss = -V

        # policy sums-to-1 diagnostic (asserted every step; cheap)
        assert torch.allclose(pi.sum(dim=1), torch.ones(N)), "policy rows !~ 1"

        opt.zero_grad(); loss.backward(); opt.step()

        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                pi_mean = pi.mean(0).detach()
                cap_viol = (pi_mean - b_t).clamp(min=0)
            grad_norm = float(sum(p.grad.pow(2).sum() for p in model.parameters()
                                  if p.grad is not None).sqrt())
            history.append({
                "step": step,
                "V": float(V.item()),
                "mu": mu_star.detach().numpy().copy(),
                "pi_mean": pi_mean.numpy().copy(),
                "cap_viol_sup": float(cap_viol.max().item()),
                "grad_norm": grad_norm,
            })
            print(f"[train-{kind}] step={step:4d}  V={V.item(): .4f}  "
                  f"mu={mu_star.detach().numpy().round(3)}  "
                  f"pi_mean={pi_mean.numpy().round(3)}  "
                  f"cap_viol_sup={cap_viol.max().item():.3e}  "
                  f"|grad|={grad_norm:.2e}")
    wall = time.time() - t0
    print(f"[train-{kind}] done in {wall:.1f}s")
    return model, history


# === Block 9: evaluation / diagnostics =======================================
def oracle_value(pi_np, Y_pot_np):
    """True mean policy value using the (held-out) counterfactuals."""
    return float((pi_np * Y_pot_np).sum(axis=1).mean())


def evaluate(model, eval_data, tag):
    """Compute oracle V on held-out evaluation data, plus realized allocation.

    On eval we always use the convex G-solve (scipy) regardless of which layer
    trained the model, so the eval allocation / shadow prices are comparable.
    """
    with torch.no_grad():
        X_e = torch.tensor(eval_data["X"])
        M_e = model(X_e)
    mu = solve_G_scipy(M_e)                     # size-flexible G-solve
    with torch.no_grad():
        pi_e = softmax_policy(M_e, mu, TAU)
    pi_np = pi_e.numpy()
    V_orc = oracle_value(pi_np, eval_data["Y_pot"])
    alloc = pi_np.mean(axis=0)
    return {"tag": tag, "V_oracle": V_orc, "alloc": alloc,
            "mu_on_eval": mu.numpy()}


if __name__ == "__main__":
    # Train both pipelines.
    print("\n========= TRAIN G (CVXPYLayer, convex dual) =========")
    model_G, hist_G = train("G", steps=200, lr=5e-3, log_every=20, seed=1)

    print("\n========= TRAIN F (implicit diff, non-convex literal) =========")
    model_F, hist_F = train("F", steps=200, lr=5e-3, log_every=20, seed=1)

    # Evaluate on a fresh held-out sample.
    eval_data = generate_data(N=2000, seed=10_000)
    eG = evaluate(model_G, eval_data, "G")
    eF = evaluate(model_F, eval_data, "F")

    # Baselines.
    pi_rand = np.full((2000, T), 1.0 / T)
    V_rand = oracle_value(pi_rand, eval_data["Y_pot"])

    with torch.no_grad():
        M_eval = model_G(torch.tensor(eval_data["X"])).numpy()
    pi_greedy = np.zeros_like(M_eval)
    pi_greedy[np.arange(len(M_eval)), M_eval.argmax(axis=1)] = 1.0
    V_greedy = oracle_value(pi_greedy, eval_data["Y_pot"])

    print("\n========= EVALUATION =========")
    print(f"random policy           V_oracle = {V_rand: .4f}")
    print(f"greedy-no-cap (from G)  V_oracle = {V_greedy: .4f}  "
          f"alloc = {pi_greedy.mean(0).round(3)}")
    print(f"G (CVXPYLayer)          V_oracle = {eG['V_oracle']: .4f}  "
          f"alloc = {eG['alloc'].round(3)}  mu_eval = {eG['mu_on_eval'].round(3)}")
    print(f"F (implicit diff)       V_oracle = {eF['V_oracle']: .4f}  "
          f"alloc = {eF['alloc'].round(3)}  mu_eval = {eF['mu_on_eval'].round(3)}")

    # Final sanity checks.
    print("\n========= SANITY =========")
    with torch.no_grad():
        M_train = model_G(X_t)
    mu_train = solve_G_scipy(M_train)
    with torch.no_grad():
        pi_G_train = softmax_policy(M_train, mu_train, TAU)
        print("policy sums to 1 (G train):",
              torch.allclose(pi_G_train.sum(dim=1), torch.ones(N)))
    print("propensities sum to 1 per row:",
          np.allclose(data["E"].sum(axis=1), 1.0))
    print("propensities in (0,1):",
          bool(((data["E"] > 0) & (data["E"] < 1)).all()))
    print("capacities b =", B, " ; sum(b) =", B.sum())

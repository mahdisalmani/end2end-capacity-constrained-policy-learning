"""Two tau studies for the report.

Study 1 (inner-only): for fixed random score matrices, solve the inner
problem under F and under G across a tau grid; record the worst relative
capacity excess of the induced softmax allocation on arms with mu > 0,
and the F-G objective gap vs the tau*log(T) bound. Averaged over seeds.

Study 2 (end-to-end): train Gs and F on one synthetic dataset per tau
(short runs), then report eval-side oracle value and capacity violation of
the SOFT policy and of the argmax deployment. Shows how tau trades
sharpness vs feasibility through training.

Writes JSON to results/tau_study.json.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)

from src.inner_F import _solve_F_inner, _F_torch
from src.inner_G import _solve_G_inner, _G_torch
from src.policy import softmax_policy
from src.data import generate_data
from src.train import train_GF
from src.evaluation import onehot_from_scores
from src.policy import oracle_value

TAUS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
T = 6
N = 400
B = np.array([0.5, 0.2, 0.15, 0.15, 0.1, 0.1])
B_t = torch.tensor(B)

out = {"taus": TAUS, "T": T, "N": N, "B": B.tolist()}

# ---- Study 1: inner-only feasibility --------------------------------------
excess_F, excess_G, gap_frac = [], [], []
for tau in TAUS:
    eF, eG, gf = [], [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        M = torch.tensor(rng.normal(size=(N, T)))
        M[:, 1] += 1.2      # make scarce arms attractive so caps bind
        M[:, 2] += 0.8
        muF, _ = _solve_F_inner(M, B_t, tau, tol=1e-12, maxiter=3000)
        muG, _ = _solve_G_inner(M, B_t, tau, tol=1e-12, maxiter=3000)
        for mu, acc in ((muF, eF), (muG, eG)):
            alloc = softmax_policy(M, mu, tau).mean(0).numpy()
            binding = mu.numpy() > 1e-6
            rel_ex = np.where(binding, (alloc - B) / B, -np.inf)
            acc.append(float(rel_ex.max()) if binding.any() else 0.0)
        Fv = _F_torch(muF, M, B_t, tau).item()
        Gv = _G_torch(muG, M, B_t, tau).item()
        gf.append((Gv - Fv) / (tau * np.log(T)))   # fraction of the bound
    excess_F.append(float(np.mean(eF)))
    excess_G.append(float(np.mean(eG)))
    gap_frac.append(float(np.mean(gf)))
    print(f"tau={tau:5.2f}  max_rel_excess F={excess_F[-1]:+.4f}  "
          f"G={excess_G[-1]:+.4f}  (G*-F*)/(tau logT)={gap_frac[-1]:.3f}")

out["inner"] = {"excess_F": excess_F, "excess_G": excess_G,
                "gap_frac": gap_frac, "seeds": 8}

# ---- Study 2: end-to-end tau effect ---------------------------------------
DGP = dict(d=12, T=T, sigma_y=0.05, propensity_strength=0.7,
           outcome_strength=2.0, treatment_effect_strength=6.0,
           clip_propensity=0.02)
train_data = generate_data(300, seed=0, **DGP)
eval_data = generate_data(4000, seed=10_000, **DGP)
X_t = torch.tensor(train_data["X"])
X_e = torch.tensor(eval_data["X"])
Y_pot = eval_data["Y_pot"]

# Train-side violation isolates the OBJECTIVE bias (mu was solved on exactly
# these rows, so G's stationarity should hold to solver tolerance); eval-side
# violation adds finite-sample estimation error on top. Reporting both is what
# separates "the surrogate is biased" from "300 rows is a small sample".
e2e = {"tau": [], "kind": [], "V_soft": [], "V_argmax": [],
       "viol_soft": [], "viol_argmax": [], "viol_soft_train": []}
for tau in TAUS:
    for kind in ("F", "Gs"):
        model, mu, _ = train_GF(kind, train_data, D=12, T=T, tau=tau,
                                b=B, steps=120, lr=5e-3, log_every=1000,
                                seed=1)
        with torch.no_grad():
            M_e = model(X_e)
            M_t = model(X_t)
        pi_soft = softmax_policy(M_e, mu, tau).numpy()
        pi_hard = onehot_from_scores(M_e.numpy() - mu.numpy()[None, :])
        pi_soft_tr = softmax_policy(M_t, mu, tau).numpy()
        viol_tr = float(np.maximum(pi_soft_tr.mean(0) - B, 0).max())
        for tag, pi in (("soft", pi_soft), ("argmax", pi_hard)):
            alloc = pi.mean(0)
            viol = float(np.maximum(alloc - B, 0).max())
            V = oracle_value(pi, Y_pot)
            if tag == "soft":
                v_soft, viol_soft = V, viol
            else:
                v_hard, viol_hard = V, viol
        e2e["tau"].append(tau); e2e["kind"].append(kind)
        e2e["V_soft"].append(v_soft); e2e["V_argmax"].append(v_hard)
        e2e["viol_soft"].append(viol_soft); e2e["viol_argmax"].append(viol_hard)
        e2e["viol_soft_train"].append(viol_tr)
        print(f"tau={tau:5.2f} {kind:2s}  V_soft={v_soft:7.3f} "
              f"viol_train={viol_tr:.5f} viol_eval={viol_soft:.4f}  "
              f"V_argmax={v_hard:7.3f}")

out["e2e"] = e2e
os.makedirs("results", exist_ok=True)
with open("results/tau_study.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote results/tau_study.json")

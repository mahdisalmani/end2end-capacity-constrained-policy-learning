"""Evaluation utilities for soft and one-hot policies.

Every evaluator scores a (train-policy, eval-policy) pair with the same
metrics and returns the same result-row schema, built by `score_policy_pair`:

    V_IPW_train / V_IPW_eval : IPW value on each split
    V_DR_eval                : doubly-robust value on eval (shared m_hat)
    V_oracle                 : counterfactual value on eval (synthetic only)
    alloc / cap_viol_sup / cap_ok : realized eval allocation vs capacities b

Metric conventions follow Sec. 5.1 of the paper.
"""

import numpy as np
import torch

from .policy import softmax_policy, oracle_value, ipw_value_np, dr_value_np


def onehot_from_scores(scores):
    """One-hot argmax policy from an (N, T) score matrix (ties -> lowest t)."""
    pi = np.zeros_like(scores)
    pi[np.arange(len(scores)), scores.argmax(axis=1)] = 1.0
    return pi


def score_policy_pair(pi_train, pi_eval, train_data, eval_data, m_hat_eval,
                      b, tag, mu=None, extra=None):
    """Score a policy on both splits and return the standard result row."""
    b = np.asarray(b)
    T = b.shape[0]

    alloc = pi_eval.mean(axis=0)
    cap_viol_sup = float(np.maximum(alloc - b, 0.0).max())

    row = {
        "tag": tag,
        "V_IPW_train": ipw_value_np(
            pi_train, train_data["T"], train_data["Y"], train_data["e_T"]
        ),
        "V_IPW_eval": ipw_value_np(
            pi_eval, eval_data["T"], eval_data["Y"], eval_data["e_T"]
        ),
        "V_DR_eval": dr_value_np(
            pi_eval, eval_data["T"], eval_data["Y"], eval_data["e_T"], m_hat_eval
        ),
        "V_oracle": oracle_value(pi_eval, eval_data["Y_pot"]),
        "alloc": alloc,
        "cap_viol_sup": cap_viol_sup,
        "cap_ok": bool(np.all(alloc <= b + 1e-3)),
        "method": tag,
        "mu": np.full(T, np.nan) if mu is None else np.asarray(mu),
        "lp_status": "NA",
        "lp_time": np.nan,
        "total_time": np.nan,
    }
    if extra:
        row.update(extra)
    return row


def evaluate_GF_model(model, mu_train, train_data, eval_data, m_hat_eval,
                      b, tau, T, tag, policy="softmax"):
    """
    Apply the trained policy (theta, mu_train) to both splits.
    mu_train is frozen — it is part of the policy, not re-solved on eval.

    `policy` selects the deployment rule:
      - "softmax": pi(t|x) = softmax_t((M[t,x] - mu_t) / tau)   — F's natural form
      - "argmax":  pi(t|x) = onehot(argmax_t (M[t,x] - mu_t))   — G's LP-consistent form
    """
    if not torch.is_tensor(mu_train):
        mu_train = torch.tensor(mu_train)
    mu_np = mu_train.detach().cpu().numpy()

    with torch.no_grad():
        X_e = torch.tensor(eval_data["X"])
        X_t = torch.tensor(train_data["X"])

        if policy == "softmax":
            pi_e = softmax_policy(model(X_e), mu_train, tau).numpy()
            pi_t = softmax_policy(model(X_t), mu_train, tau).numpy()
        elif policy == "argmax":
            pi_e = onehot_from_scores(model(X_e).numpy() - mu_np[None, :])
            pi_t = onehot_from_scores(model(X_t).numpy() - mu_np[None, :])
        else:
            raise ValueError(f"Unknown policy: {policy}")

    return score_policy_pair(pi_t, pi_e, train_data, eval_data, m_hat_eval,
                             b, tag, mu=mu_np)


def evaluate_greedy_no_cap_from_model(model, train_data, eval_data, m_hat_eval,
                                      b, T, tag="greedy_no_cap_from_G"):
    """Greedy argmax over trained scores, ignoring capacity."""
    with torch.no_grad():
        pi_e = onehot_from_scores(model(torch.tensor(eval_data["X"])).numpy())
        pi_t = onehot_from_scores(model(torch.tensor(train_data["X"])).numpy())

    return score_policy_pair(pi_t, pi_e, train_data, eval_data, m_hat_eval,
                             b, tag)


def evaluate_policy(pi_onehot_eval, assignments_eval,
                    pi_onehot_train, assignments_train,
                    train_data, eval_data, m_hat_eval, b, tag):
    """Evaluate a one-hot policy on both splits (used by S2 pipeline)."""
    row = score_policy_pair(pi_onehot_train, pi_onehot_eval,
                            train_data, eval_data, m_hat_eval, b, tag)
    # The S2 pipeline overwrites method/mu/lp_* itself; drop the defaults so
    # the returned dict matches the historical schema of this function.
    for k in ("method", "mu", "lp_status", "lp_time", "total_time"):
        row.pop(k)

    print(
        f"[eval] {tag:12s}  "
        f"V_IPW_train={row['V_IPW_train']: .4f}  "
        f"V_IPW_eval={row['V_IPW_eval']: .4f}  "
        f"V_DR_eval={row['V_DR_eval']: .4f}  "
        f"V_oracle={row['V_oracle']: .4f}  "
        f"capviol={row['cap_viol_sup']:.3e}  "
        f"cap_ok={row['cap_ok']}"
    )
    return row

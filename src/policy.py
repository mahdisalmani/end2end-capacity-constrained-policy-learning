"""Softmax policy + IPW / DR / oracle value functionals."""

import numpy as np
import torch


def softmax_policy(M, mu, tau):
    """pi_{t,i}(x) = softmax_t((m_{t,i} - mu_t) / tau)."""
    return torch.softmax((M - mu.unsqueeze(0)) / tau, dim=1)


def ipw_value(pi, T_obs, Y, e_T):
    """V_IPW(theta) = (1/N) sum_i pi_{T_i,i} * Y_i / e_{T_i}(X_i)."""
    pi_t = pi.gather(1, T_obs.unsqueeze(1)).squeeze(1)
    return (pi_t * Y / e_T).mean()


def ipw_value_np(pi, T_obs, Y, e_T):
    """Numpy version of ipw_value. pi may be soft (rows sum to 1) or one-hot."""
    pi_t = pi[np.arange(len(T_obs)), T_obs]
    return float((pi_t * Y / e_T).mean())


def dr_value_np(pi, T_obs, Y, e_T, m_hat):
    """
    Doubly-robust / AIPW estimator of V(pi).

        V_DR = mean_i sum_t pi_{t,i} * m_hat_{t,i}                     (direct)
             + mean_i pi_{T_i,i} / e_{T_i}(X_i) * (Y_i - m_hat_{T_i,i})  (residual IPW)

    m_hat is an (N, T) matrix of predicted conditional means
    m_hat_{t,i} = E_hat[Y | X_i, T = t].
    """
    N = len(T_obs)
    direct = (pi * m_hat).sum(axis=1).mean()
    pi_t = pi[np.arange(N), T_obs]
    m_t = m_hat[np.arange(N), T_obs]
    correction = (pi_t / e_T * (Y - m_t)).mean()
    return float(direct + correction)


def oracle_value(pi_np, Y_pot_np):
    """Oracle value of a policy given counterfactual outcomes.

    Works for soft policies (rows on the simplex) and one-hot policies alike:
    V = mean_i sum_t pi_{t,i} * Y^t_i.
    """
    return float((pi_np * Y_pot_np).sum(axis=1).mean())


# Backward-compatible aliases (soft and one-hot cases share one formula).
oracle_value_soft = oracle_value
oracle_value_onehot = oracle_value

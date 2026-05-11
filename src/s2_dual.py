"""
S2: sample-based dual-price method.

Fit per-treatment outcome models m_hat_t(x), then solve the sample-based
dual LP for shadow prices mu_hat, then deploy a deterministic policy
    a(x) = argmax_t [m_hat_t(x) - mu_hat_t]
on the eval split.
"""

import time
import warnings

import numpy as np
import cvxpy as cp
import torch
from torch import nn

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .evaluation import evaluate_policy


class _MLPRegressor:
    """Per-arm scalar MLP regressor with the same trunk shape as MLPScore
    (Linear(d_in,64)-Tanh-Linear(64,64)-Tanh-Linear(64,64)-Tanh-Linear(64,1)),
    trained with Adam + MSE. Inputs are standardized; sklearn-style fit/predict
    interface so it slots into the existing `models[t].predict(X)` loop."""

    def __init__(self, hidden=64, steps=500, lr=5e-3, weight_decay=0.0, seed=0):
        self.hidden = int(hidden)
        self.steps = int(steps)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self._scaler = StandardScaler()
        self._net = None
        self._dtype = torch.float64

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        Xs = self._scaler.fit_transform(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        d_in = Xs.shape[1]
        h = self.hidden

        self._net = nn.Sequential(
            nn.Linear(d_in, h), nn.Tanh(),
            nn.Linear(h, h),    nn.Tanh(),
            nn.Linear(h, h),    nn.Tanh(),
            nn.Linear(h, 1),
        ).to(self._dtype)
        for m in self._net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        X_t = torch.tensor(Xs, dtype=self._dtype)
        y_t = torch.tensor(y,  dtype=self._dtype).unsqueeze(1)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        for _ in range(self.steps):
            opt.zero_grad()
            pred = self._net(X_t)
            loss = ((pred - y_t) ** 2).mean()
            loss.backward()
            opt.step()
        return self

    def predict(self, X):
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float64))
        with torch.no_grad():
            out = self._net(torch.tensor(Xs, dtype=self._dtype)).cpu().numpy()
        return out.reshape(-1)


def fit_outcome_models(X_train, T_train, Y_train, T, method, E_train=None):
    """
    Fit per-treatment conditional mean outcome models.

    Methods:
        linear : OLS per treatment
        lasso  : LassoCV per treatment
        tree   : DecisionTreeRegressor per treatment
        knn    : KNeighborsRegressor per treatment
        dr     : doubly-robust pseudo-outcome + Lasso smoothing
    """
    method = method.lower()
    models = []

    if method == "dr":
        if E_train is None:
            raise ValueError("DR method requires E_train, the full propensity matrix.")

        first_stage = []

        for t in range(T):
            mask = T_train == t

            if mask.sum() < 2:
                warnings.warn(
                    f"[DR] treatment {t} has <2 samples; using zero first-stage model."
                )
                first_stage.append(None)
                continue

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lasso", LassoCV(cv=min(5, mask.sum()), max_iter=5000)),
            ])
            pipe.fit(X_train[mask], Y_train[mask])
            first_stage.append(pipe)

        N_train = X_train.shape[0]
        phi = np.zeros((N_train, T))
        E_clip = np.clip(E_train, 1e-6, None)

        for t in range(T):
            if first_stage[t] is None:
                m_hat = np.zeros(N_train)
            else:
                m_hat = first_stage[t].predict(X_train)

            indicator = (T_train == t).astype(float)
            residual = Y_train - m_hat
            phi[:, t] = m_hat + indicator / E_clip[:, t] * residual

        for t in range(T):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lasso", LassoCV(cv=5, max_iter=5000)),
            ])
            pipe.fit(X_train, phi[:, t])
            models.append(pipe)

        return models

    for t in range(T):
        mask = T_train == t

        if mask.sum() < 2:
            warnings.warn(
                f"[{method}] treatment {t} has <2 samples; using zero predictor."
            )
            models.append(None)
            continue

        Xt = X_train[mask]
        Yt = Y_train[mask]

        if method == "linear":
            reg = LinearRegression()
        elif method == "lasso":
            reg = Pipeline([
                ("scaler", StandardScaler()),
                ("lasso", LassoCV(cv=min(5, mask.sum()), max_iter=5000)),
            ])
        elif method == "tree":
            reg = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=0)
        elif method == "knn":
            k = min(10, max(1, mask.sum() - 1))
            reg = KNeighborsRegressor(n_neighbors=k)
        elif method == "mlp":
            reg = _MLPRegressor(hidden=64, steps=500, lr=5e-3, seed=int(t))
        else:
            raise ValueError(f"Unknown method: {method}")

        reg.fit(Xt, Yt)
        models.append(reg)

    return models


def get_mhat_matrix(models, X, T):
    """M_hat[i, t] = m_hat_t(X_i)."""
    N_eval = X.shape[0]
    M_hat = np.zeros((N_eval, T))

    for t in range(T):
        if models[t] is None:
            M_hat[:, t] = 0.0
        else:
            M_hat[:, t] = models[t].predict(X)

    return M_hat


def solve_dual_lp(M_hat, b, verbose=False):
    """
    Sample-based dual LP:

        min_{mu >= 0, z}  (1/N) sum_i z_i + b^T mu
        s.t.              z_i >= M_hat[i, t] - mu_t   for all i, t
    """
    N_local, T_local = M_hat.shape
    assert len(b) == T_local, (
        f"b has length {len(b)}, but M_hat has {T_local} treatments."
    )

    mu = cp.Variable(T_local, nonneg=True)
    z = cp.Variable(N_local)

    constraints = [z >= M_hat[:, t] - mu[t] for t in range(T_local)]

    objective = cp.Minimize((1.0 / N_local) * cp.sum(z) + np.asarray(b) @ mu)
    problem = cp.Problem(objective, constraints)

    t0 = time.time()
    try:
        problem.solve(verbose=verbose)
    except Exception as e:
        warnings.warn(f"Default solver failed with error: {e}. Retrying with SCS.")
        problem.solve(solver=cp.SCS, verbose=verbose)
    solve_time = time.time() - t0

    if mu.value is None:
        raise RuntimeError(f"LP did not return a solution. Status: {problem.status}")

    mu_hat = np.asarray(mu.value).reshape(-1)
    z_hat = np.asarray(z.value).reshape(-1)

    return mu_hat, z_hat, problem.status, solve_time


def recover_policy(M_hat, mu_hat):
    """Deterministic policy a(x) = argmax_t [M_hat[i, t] - mu_hat_t]."""
    adjusted = M_hat - mu_hat[None, :]
    assignments = adjusted.argmax(axis=1)

    N_local, T_local = M_hat.shape
    pi_onehot = np.zeros_like(M_hat)
    pi_onehot[np.arange(N_local), assignments] = 1.0

    return assignments, pi_onehot


def run_dual_method_with_eval_mu(method_name, train_data, eval_data, m_hat_eval,
                                 T, b, verbose_lp=False):
    """
    Run an S2 method and return BOTH the vanilla result (μ fit on train) and
    the μ-variant result (μ fit on eval), sharing the same fitted outcome
    models.

    Returns a list [vanilla_result, mu_result], in that order.
    """
    method_name = method_name.lower()
    tag_vanilla = f"S2-{method_name}"
    tag_mu = f"S2-{method_name}-mu"

    t0 = time.time()

    models = fit_outcome_models(
        X_train=train_data["X"],
        T_train=train_data["T"],
        Y_train=train_data["Y"],
        T=T, method=method_name,
        E_train=train_data["E"],
    )

    M_hat_train = get_mhat_matrix(models, train_data["X"], T)
    M_hat_eval_method = get_mhat_matrix(models, eval_data["X"], T)

    # ----- vanilla: μ fit on train -----
    mu_train, _, status_t, lp_time_t = solve_dual_lp(
        M_hat_train, b, verbose=verbose_lp,
    )
    a_eval_v, pi_eval_v = recover_policy(M_hat_eval_method, mu_train)
    a_train_v, pi_train_v = recover_policy(M_hat_train, mu_train)
    res_v = evaluate_policy(
        pi_onehot_eval=pi_eval_v, assignments_eval=a_eval_v,
        pi_onehot_train=pi_train_v, assignments_train=a_train_v,
        train_data=train_data, eval_data=eval_data,
        m_hat_eval=m_hat_eval, b=b, tag=tag_vanilla,
    )
    res_v.update({
        "method": tag_vanilla,
        "mu": mu_train,
        "lp_status": status_t,
        "lp_time": lp_time_t,
    })

    # ----- μ-variant: μ fit on eval -----
    mu_eval, _, status_e, lp_time_e = solve_dual_lp(
        M_hat_eval_method, b, verbose=verbose_lp,
    )
    a_eval_m, pi_eval_m = recover_policy(M_hat_eval_method, mu_eval)
    a_train_m, pi_train_m = recover_policy(M_hat_train, mu_eval)
    res_m = evaluate_policy(
        pi_onehot_eval=pi_eval_m, assignments_eval=a_eval_m,
        pi_onehot_train=pi_train_m, assignments_train=a_train_m,
        train_data=train_data, eval_data=eval_data,
        m_hat_eval=m_hat_eval, b=b, tag=tag_mu,
    )
    res_m.update({
        "method": tag_mu,
        "mu": mu_eval,
        "lp_status": status_e,
        "lp_time": lp_time_e,
    })

    total_time = time.time() - t0
    res_v["total_time"] = total_time
    res_m["total_time"] = total_time

    return [res_v, res_m]


def run_dual_method(method_name, train_data, eval_data, m_hat_eval,
                    T, b, verbose_lp=False):
    """End-to-end S2 pipeline for one outcome-estimation method."""
    method_name = method_name.lower()
    tag = f"S2-{method_name}"

    t0 = time.time()

    models = fit_outcome_models(
        X_train=train_data["X"],
        T_train=train_data["T"],
        Y_train=train_data["Y"],
        T=T,
        method=method_name,
        E_train=train_data["E"],
    )

    M_hat_train = get_mhat_matrix(models, train_data["X"], T)
    mu_hat, z_hat, status, lp_time = solve_dual_lp(M_hat_train, b, verbose=verbose_lp)
    M_hat_eval = get_mhat_matrix(models, eval_data["X"], T)

    assignments_eval, pi_onehot_eval = recover_policy(M_hat_eval, mu_hat)
    assignments_train, pi_onehot_train = recover_policy(M_hat_train, mu_hat)

    total_time = time.time() - t0

    result = evaluate_policy(
        pi_onehot_eval=pi_onehot_eval,
        assignments_eval=assignments_eval,
        pi_onehot_train=pi_onehot_train,
        assignments_train=assignments_train,
        train_data=train_data,
        eval_data=eval_data,
        m_hat_eval=m_hat_eval,
        b=b,
        tag=tag,
    )
    result.update({
        "method": tag,
        "mu": mu_hat,
        "lp_status": status,
        "lp_time": lp_time,
        "total_time": total_time,
    })

    print(
        f"       LP status={status}, "
        f"LP time={lp_time:.3f}s, "
        f"total time={total_time:.3f}s"
    )
    print(f"       mu={np.array2string(mu_hat, precision=3)}")
    print(f"       alloc={np.array2string(result['alloc'], precision=3)}")

    return result

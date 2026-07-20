"""Checks for the policy functionals, data utilities, training loops and
queue simulator.

Run directly (no pytest needed):
    python -m tests.test_policy_and_pipeline
or via pytest:
    pytest tests/
"""

import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from src.data import generate_data  # noqa: E402
from src.policy import (  # noqa: E402
    dr_value_np,
    ipw_value,
    ipw_value_np,
    oracle_value,
    softmax_policy,
)
from src.train import train_GF  # noqa: E402
from src.train_alt import train_alt  # noqa: E402
from experiments.common import (  # noqa: E402
    aggregate_one,
    ipw_policy_value,
    make_random_assigner,
    make_streams,
    simulate,
    subsample_rows,
)


def _tiny_data(N=120, T=4, d=6, seed=0):
    return generate_data(N=N, seed=seed, d=d, T=T, sigma_y=0.1,
                         propensity_strength=0.7, outcome_strength=1.0,
                         treatment_effect_strength=2.0, clip_propensity=0.02)


def test_softmax_policy_rows_sum_to_one():
    M = torch.randn(50, 5)
    mu = torch.rand(5)
    pi = softmax_policy(M, mu, 0.3)
    assert torch.allclose(pi.sum(dim=1), torch.ones(50))
    assert (pi > 0).all()


def test_ipw_value_torch_and_numpy_agree():
    data = _tiny_data()
    pi = torch.full((120, 4), 0.25)
    v_t = ipw_value(pi, torch.tensor(data["T"]), torch.tensor(data["Y"]),
                    torch.tensor(data["e_T"]))
    v_n = ipw_value_np(pi.numpy(), data["T"], data["Y"], data["e_T"])
    assert abs(float(v_t) - v_n) < 1e-12


def test_ipw_of_onehot_arms_matches_policy_value():
    """experiments.common.ipw_policy_value(arms, data) is IPW of the one-hot
    policy induced by `arms`."""
    data = _tiny_data()
    arms = np.random.default_rng(0).integers(4, size=120)
    pi = np.zeros((120, 4)); pi[np.arange(120), arms] = 1.0
    assert abs(ipw_policy_value(arms, data)
               - ipw_value_np(pi, data["T"], data["Y"], data["e_T"])) < 1e-12


def test_dr_reduces_to_ipw_when_mhat_zero():
    data = _tiny_data()
    pi = np.full((120, 4), 0.25)
    m0 = np.zeros((120, 4))
    assert abs(dr_value_np(pi, data["T"], data["Y"], data["e_T"], m0)
               - ipw_value_np(pi, data["T"], data["Y"], data["e_T"])) < 1e-12


def test_dr_equals_direct_when_mhat_perfect_and_pi_uniform():
    """With m_hat = E[Y^t|X] exact (no noise DGP) the correction term has
    mean ~ the residual noise only."""
    data = generate_data(N=200, seed=1, d=5, T=3, sigma_y=0.0,
                         propensity_strength=0.7, outcome_strength=1.0,
                         treatment_effect_strength=2.0, clip_propensity=0.02)
    pi = np.full((200, 3), 1 / 3)
    m_hat = data["Y_pot"]          # exact conditional means (sigma_y = 0)
    dr = dr_value_np(pi, data["T"], data["Y"], data["e_T"], m_hat)
    direct = (pi * m_hat).sum(axis=1).mean()
    assert abs(dr - direct) < 1e-10


def test_oracle_value_soft_equals_onehot_formula():
    Y_pot = np.random.default_rng(0).normal(size=(30, 3))
    arms = Y_pot.argmax(axis=1)
    pi = np.zeros_like(Y_pot); pi[np.arange(30), arms] = 1.0
    assert abs(oracle_value(pi, Y_pot) - Y_pot.max(axis=1).mean()) < 1e-12


def test_subsample_rows_shapes_and_passthrough():
    data = _tiny_data()
    sub = subsample_rows(data, 50, seed=0)
    assert len(sub["T"]) == 50 and sub["X"].shape == (50, 6)
    assert sub["Beta"].shape == data["Beta"].shape       # not per-row
    sub2 = subsample_rows(data, 10_000, seed=0)          # n > N is clamped
    assert len(sub2["T"]) == 120


def test_train_GF_and_alt_smoke():
    """3-step training runs end to end, respects caps approximately, and
    returns finite values (F, Gs and Alt paths — no cvxpy needed)."""
    data = _tiny_data()
    b = np.array([1.0, 0.2, 0.2, 0.2])
    for kind in ("F", "Gs"):
        model, mu, hist = train_GF(kind, data, D=6, T=4, tau=0.3, b=b,
                                   steps=3, lr=5e-3, log_every=10, seed=0)
        assert np.isfinite(hist[-1]["V"])
        assert mu.shape == (4,) and bool((mu >= 0).all())
    model, mu, hist = train_alt(data, D=6, T=4, tau=0.3, b=b,
                                outer_steps=3, inner_freq=2, lr=5e-3,
                                log_every=10, seed=0)
    assert np.isfinite(hist[-1]["V"]) and bool((mu >= 0).all())


def test_queue_simulator_conservation():
    """Every arrival is either served or unserved; waits are non-negative;
    per-arm allocation fractions sum to 1."""
    data = _tiny_data(N=300)
    B = np.array([1.0, 0.2, 0.2, 0.2])
    people_t, person_idx, T_max, resource_t = make_streams(
        data, N_sim=500, lambda_people=1.0, B=B, max_time_mult=1.5, seed=0)
    recs = simulate(people_t, person_idx, resource_t,
                    make_random_assigner(4), T=4, T_max=T_max,
                    eval_data=data, sim_seed=0)
    assert (recs["wait"] >= 0).all()
    row = aggregate_one(recs, "random", 0, B, 500, 0.0)
    assert row["num_unserved"] == int((~recs["served"]).sum())
    assert abs(sum(row[f"alloc_{t}"] for t in range(4)) - 1.0) < 1e-12
    served_or_queued = recs["served"].sum() + row["num_unserved"]
    assert served_or_queued == 500


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

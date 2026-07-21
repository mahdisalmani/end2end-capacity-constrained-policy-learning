"""
ACTG Study 175 loader (UCI id=890, N=2,139, CC BY 4.0) — a REAL randomized
clinical trial with FOUR treatment arms:

    0  zidovudine (ZDV) monotherapy          — the era's default of care
    1  ZDV + didanosine (ddI)
    2  ZDV + zalcitabine
    3  ddI monotherapy

Why this dataset: it is the strongest real-world fit for capacity-constrained
policy learning in the UCI catalog. (i) Multi-arm and RANDOMIZED — the
assignment mechanism is known, so IPW is unbiased by design (arm shares are
estimated on the training split only and are constant in x). (ii) The
combination arms beat monotherapy for nearly everyone (trial-average CD4 at
20 weeks: 336 vs 372-403), so a capacity cap on the newer regimens MUST bind
— the allocation question is genuinely "who gets the scarce regimen", with
known effect heterogeneity by antiretroviral history. (iii) Scarcity of the
newer regimens is historically real (early-1990s HIV care, and still the
operative constraint in resource-limited programs).

Outcome  Y = CD4 count at 20±5 weeks / 100  (cd420; complete for all rows;
         higher is better; scaled so IPW gradients are well-conditioned).
Features 15 BASELINE covariates only (demographics, Karnofsky score,
         antiretroviral history, baseline CD4/CD8). Post-treatment columns
         (time, cid, offtrt, cd420, cd820) are excluded from X.
Caps     b = (1.0, 0.30, 0.30, 0.30): monotherapy unconstrained as the
         default; each newer regimen can absorb 30% of the population.

Standardization is fit on the training split only and applied to eval,
matching the project's leak-fixed convention.
"""

import os

import numpy as np
import pandas as pd

from experiments.common import split_indices, standardize_train_fit

RAW_CSV = "data/uci/actg175.csv"
CACHE_DIR = "data/uci"

FEATURES = ["age", "wtkg", "hemo", "homo", "drugs", "karnof", "oprior",
            "z30", "preanti", "race", "gender", "str2", "strat", "symptom",
            "cd40", "cd80"]
Y_SCALE = 100.0
T_ARMS = 4


def load_actg(train_frac=0.7, seed=0):
    if not os.path.exists(RAW_CSV):
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=890)
        os.makedirs(CACHE_DIR, exist_ok=True)
        pd.concat([ds.data.features, ds.data.targets], axis=1).to_csv(RAW_CSV, index=False)
    df = pd.read_csv(RAW_CSV)

    feats = [c for c in FEATURES if df[c].nunique() > 1]   # drop constants
    X_raw = df[feats].values.astype(np.float64)
    T = df["trt"].values.astype(np.int64)
    Y = df["cd420"].values.astype(np.float64) / Y_SCALE

    N_total = len(T)
    tr, ev, n_train = split_indices(N_total, train_frac, seed)

    X = standardize_train_fit(X_raw, fit_idx=tr)

    # Randomized trial: propensities are the arm shares, constant in x —
    # estimated on the training split only (they are ~0.25 each by design).
    shares = np.bincount(T[tr], minlength=T_ARMS) / len(tr)
    E = np.tile(shares, (N_total, 1))
    e_T = E[np.arange(N_total), T]

    def _slice(I):
        return {"X": X[I].copy(), "T": T[I].copy(), "Y": Y[I].copy(),
                "e_T": e_T[I].copy(), "E": E[I].copy(),
                "Y_pot": np.full((len(I), T_ARMS), np.nan),
                "Beta": np.zeros((T_ARMS, X.shape[1])),
                "Alpha": np.zeros((T_ARMS, X.shape[1]))}

    train_data, eval_data = _slice(tr), _slice(ev)
    cfg = {"N": int(n_train), "T": T_ARMS, "D": int(X.shape[1]), "TAU": 0.1,
           "B": np.array([1.0, 0.30, 0.30, 0.30])}

    print(f"[actg] N={N_total} train={n_train} eval={N_total - n_train}  "
          f"D={cfg['D']}  arm shares={np.round(shares, 3)}  "
          f"mean Y by arm={np.round([Y[T == a].mean() for a in range(T_ARMS)], 2)}")
    return train_data, eval_data, cfg

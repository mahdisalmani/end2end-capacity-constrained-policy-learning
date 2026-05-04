"""
LaLonde NSW + PSID-1 observational dataset loader.

Downloads the canonical Dehejia-Wahba text files from NBER:
    nswre74_treated.txt  (185 randomized NSW treated)
    psid_controls.txt    (2490 PSID-1 observational controls)

Combined: N = 2675, T in {0, 1}, 8 covariates. Outcome is `re78`
(real annual earnings 1978, dollars).

Returns the same dict schema the rest of the pipeline expects:
    {X, T, Y, e_T, E, Y_pot, Beta, Alpha}
plus a cfg dict with N, T, D, TAU, B. Y_pot is filled with NaN
because counterfactual outcomes are not observed in real data;
downstream code that depends on Y_pot (e.g. the queueing simulator's
oracle outcome metric) will produce NaN there, by design.

Y values are scaled to thousands of dollars so the IPW gradient
stays in a numerically friendly range during F's MLP training.
"""

import os
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


DEHEJIA_URLS = {
    "nswre74_treated.txt": "https://users.nber.org/~rdehejia/data/nswre74_treated.txt",
    "psid_controls.txt":   "https://users.nber.org/~rdehejia/data/psid_controls.txt",
}

DEFAULT_CACHE = "data/lalonde"

COLS = [
    "treat", "age", "education", "black", "hispanic", "married",
    "nodegree", "re74", "re75", "re78",
]
FEATURE_COLS = [
    "age", "education", "black", "hispanic", "married",
    "nodegree", "re74", "re75",
]


def _download_if_needed(cache_dir=DEFAULT_CACHE):
    os.makedirs(cache_dir, exist_ok=True)
    for fn, url in DEHEJIA_URLS.items():
        path = os.path.join(cache_dir, fn)
        if not os.path.exists(path):
            print(f"[lalonde] downloading {url}")
            urlretrieve(url, path)


def _load_raw(cache_dir=DEFAULT_CACHE):
    treated = np.loadtxt(os.path.join(cache_dir, "nswre74_treated.txt"))
    controls = np.loadtxt(os.path.join(cache_dir, "psid_controls.txt"))
    arr = np.vstack([treated, controls])
    df = pd.DataFrame(arr, columns=COLS)
    df["treat"] = df["treat"].astype(int)
    return df


def _fit_propensity(X, T, clip=(0.05, 0.95)):
    """e(x) = P(T = 1 | X = x), via logistic regression on full data,
    clipped to keep IPW weights bounded."""
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(X, T)
    e1 = lr.predict_proba(X)[:, 1]
    return np.clip(e1, clip[0], clip[1])


def load_lalonde(
    cache_dir=DEFAULT_CACHE,
    train_frac=0.7,
    seed=0,
    y_scale=1000.0,
):
    """
    Load LaLonde NSW + PSID-1 and split into train / eval.

    Returns
    -------
    train_data : dict   (n_train rows of X, T, Y, e_T, E, Y_pot, Beta, Alpha)
    eval_data  : dict   (n_eval rows, same schema)
    cfg        : dict   {N, T, D, TAU, B}
    """
    _download_if_needed(cache_dir)
    df = _load_raw(cache_dir)

    # Standardise covariates (zero mean, unit variance).
    X_raw = df[FEATURE_COLS].values.astype(np.float64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    T = df["treat"].values.astype(np.int64)
    Y_raw = df["re78"].values.astype(np.float64)
    Y = Y_raw / y_scale

    # Propensity fit on full data, clipped.
    e1 = _fit_propensity(X, T)
    E = np.stack([1.0 - e1, e1], axis=1)               # shape (N, 2)
    e_T = E[np.arange(len(T)), T]                      # observed-arm prob

    N_total = len(T)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_total)
    n_train = int(round(train_frac * N_total))
    train_idx, eval_idx = perm[:n_train], perm[n_train:]

    def _slice(I):
        return {
            "X":     X[I].copy(),
            "T":     T[I].copy(),
            "Y":     Y[I].copy(),
            "e_T":   e_T[I].copy(),
            "E":     E[I].copy(),
            # No counterfactuals on real data.
            "Y_pot": np.full((len(I), 2), np.nan, dtype=np.float64),
            "Beta":  np.zeros((2, X.shape[1])),
            "Alpha": np.zeros((2, X.shape[1])),
        }

    train_data = _slice(train_idx)
    eval_data = _slice(eval_idx)

    cfg = {
        "N":   int(n_train),
        "T":   2,
        "D":   int(X.shape[1]),
        "TAU": 0.1,
        "B":   np.array([1.0, 0.30], dtype=np.float64),
    }

    print(
        f"[lalonde] N_total={N_total}  train={n_train}  eval={N_total - n_train}  "
        f"D={cfg['D']}  T={cfg['T']}  B={cfg['B']}  y_scale={y_scale}"
    )
    return train_data, eval_data, cfg

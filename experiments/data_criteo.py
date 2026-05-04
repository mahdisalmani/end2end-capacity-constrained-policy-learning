"""
Criteo Uplift Modeling Dataset (10% sample) loader.

Downloads the 10% sample (~32MB compressed, ~1.4M rows) from the
public S3 bucket used by the `scikit-uplift` package:
    https://criteo-bucket.s3.eu-central-1.amazonaws.com/criteo10.csv.gz

Schema (16 columns):
    f0..f11    : 12 anonymised continuous features
    treatment  : binary (T = 1 -> ad-targeting on)
    conversion : binary outcome (very sparse, ~0.3%)
    visit      : binary outcome (~4.7% — used as Y)
    exposure   : binary (ad actually shown)

The original experiment is a randomised incrementality test, so the
"true" propensity is constant ≈ 0.85. We still fit a logistic-regression
e(x) on the 12 features and clip to [0.05, 0.95] for IPW stability;
this lets the same pipeline downstream pretend the data is observational.

Y is `visit` (kept as 0/1 — IPW gradients are well-scaled at this
range, no rescaling needed).

Returns the same dict schema the rest of the pipeline expects:
    {X, T, Y, e_T, E, Y_pot, Beta, Alpha}
plus a cfg dict. Y_pot is filled with NaN — counterfactuals are
unknown for real data.
"""

import os
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


CRITEO_URL = "https://criteo-bucket.s3.eu-central-1.amazonaws.com/criteo10.csv.gz"
DEFAULT_CACHE = "data/criteo"
RAW_CSV = "criteo10.csv.gz"

FEATURE_COLS = [f"f{k}" for k in range(12)]
TARGET_COL = "visit"
TREATMENT_COL = "treatment"


def _download_if_needed(cache_dir=DEFAULT_CACHE):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, RAW_CSV)
    if not os.path.exists(path):
        print(f"[criteo] downloading {CRITEO_URL}")
        urlretrieve(CRITEO_URL, path)
    return path


def _fit_propensity(X, T, clip=(0.05, 0.95)):
    """e(x) = P(T=1 | X=x) via logistic regression on full data, clipped."""
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(X, T)
    e1 = lr.predict_proba(X)[:, 1]
    return np.clip(e1, clip[0], clip[1])


def load_criteo(
    cache_dir=DEFAULT_CACHE,
    train_frac=0.7,
    seed=0,
    subsample=200_000,
    target_col=TARGET_COL,
):
    """Load Criteo Uplift (10% sample), subsample, fit propensity, split.

    Parameters
    ----------
    subsample : int or None
        If int, randomly subsample this many rows from the 1.4M total
        before fitting / splitting. Smaller subsample = faster
        propensity fit and faster experiments. None = use all rows.
    target_col : str
        Either 'visit' (default, ~4.7% positive) or 'conversion'
        (~0.3% positive — too sparse for stable IPW).

    Returns
    -------
    train_data, eval_data : dict
        Same schema as the LaLonde / synthetic loaders. Y is the
        chosen target column (binary 0/1).
    cfg : dict
        {N, T, D, TAU, B}.
    """
    csv_path = _download_if_needed(cache_dir)
    df = pd.read_csv(csv_path, compression="gzip")

    if subsample is not None and subsample < len(df):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=subsample, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    X_raw = df[FEATURE_COLS].values.astype(np.float64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    T = df[TREATMENT_COL].values.astype(np.int64)
    Y = df[target_col].values.astype(np.float64)

    e1 = _fit_propensity(X, T)
    E = np.stack([1.0 - e1, e1], axis=1)
    e_T = E[np.arange(len(T)), T]

    N_total = len(T)
    rng = np.random.default_rng(seed + 1)
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
        "B":   np.array([1.0, 0.50], dtype=np.float64),
    }

    print(
        f"[criteo] subsample={N_total}  train={n_train}  eval={N_total - n_train}  "
        f"D={cfg['D']}  T={cfg['T']}  B={cfg['B']}  target={target_col}  "
        f"P(Y=1)={Y.mean():.4f}  P(T=1)={T.mean():.4f}"
    )
    return train_data, eval_data, cfg

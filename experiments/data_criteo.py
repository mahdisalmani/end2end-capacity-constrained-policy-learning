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
The propensity model and the feature standardization are both fit on the
TRAINING split only and then applied to eval, so no eval-split information
enters the weights the eval split is scored with.

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
from experiments.common import (
    fit_logistic_propensity,
    split_indices,
    standardize_train_fit,
)


# Sources tried in order. The original scikit-uplift S3 bucket started
# returning 403 (checked 2026-07-19); the Hugging Face mirror hosts the
# official criteo-research-uplift-v2.1 file (~300MB gz, ~14M rows).
CRITEO_URLS = {
    "full": [
        "https://criteo-bucket.s3.eu-central-1.amazonaws.com/criteo.csv.gz",
        "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
    ],
    "10pct": [
        "https://criteo-bucket.s3.eu-central-1.amazonaws.com/criteo10.csv.gz",
    ],
}
RAW_CSV_NAMES = {
    "full":  "criteo.csv.gz",
    "10pct": "criteo10.csv.gz",
}
DEFAULT_CACHE = "data/criteo"
# Seed for deriving the 10pct sample locally when its original source is gone.
TENPCT_DERIVE_SEED = 0

FEATURE_COLS = [f"f{k}" for k in range(12)]
TARGET_COL = "visit"
TREATMENT_COL = "treatment"


def _try_urls(urls, path):
    for url in urls:
        try:
            print(f"[criteo] downloading {url} -> {path}")
            urlretrieve(url, path)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[criteo] source failed ({type(e).__name__}: {e}); trying next")
            if os.path.exists(path):
                os.remove(path)
    return False


def _derive_10pct_from_full(cache_dir, path_10pct):
    """The original criteo10.csv.gz is no longer hosted anywhere; derive a
    deterministic 10% row sample (seed TENPCT_DERIVE_SEED) from the full
    file. NOTE: this is a different sample than scikit-uplift's historical
    one, so numbers differ from pre-2026 runs at the row level."""
    full_path = _download_if_needed(cache_dir, variant="full")
    print(f"[criteo] deriving 10pct sample from {full_path} "
          f"(seed={TENPCT_DERIVE_SEED}) — one-time, a few minutes")
    df = pd.read_csv(full_path, compression="gzip")
    rng = np.random.default_rng(TENPCT_DERIVE_SEED)
    idx = rng.choice(len(df), size=len(df) // 10, replace=False)
    df.iloc[np.sort(idx)].to_csv(path_10pct, index=False, compression="gzip")
    print(f"[criteo] wrote {path_10pct} ({len(idx)} rows)")


def _download_if_needed(cache_dir=DEFAULT_CACHE, variant="full"):
    """Ensure the raw csv.gz for `variant` exists locally; return its path."""
    if variant not in CRITEO_URLS:
        raise ValueError(f"Unknown variant {variant!r}; expected 'full' or '10pct'.")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, RAW_CSV_NAMES[variant])
    if os.path.exists(path):
        return path
    if not _try_urls(CRITEO_URLS[variant], path):
        if variant == "10pct":
            _derive_10pct_from_full(cache_dir, path)
        else:
            raise RuntimeError(
                "All Criteo sources failed. Download "
                "criteo-research-uplift-v2.1.csv.gz manually and place it at "
                f"{path}."
            )
    return path


# Pooled logistic propensity fit (canonical impl: experiments.common).
_fit_propensity = fit_logistic_propensity


def load_criteo(
    cache_dir=DEFAULT_CACHE,
    train_frac=0.7,
    seed=0,
    subsample=200_000,
    target_col=TARGET_COL,
    variant="full",
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
    csv_path = _download_if_needed(cache_dir, variant=variant)
    print(f"[criteo] reading {csv_path} ({variant} variant)")
    df = pd.read_csv(csv_path, compression="gzip")

    if subsample is not None and subsample < len(df):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=subsample, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    X_raw = df[FEATURE_COLS].values.astype(np.float64)
    T = df[TREATMENT_COL].values.astype(np.int64)
    Y = df[target_col].values.astype(np.float64)

    # Split FIRST, then fit standardization and the propensity model on the
    # training rows only: both are estimated objects, and fitting them on the
    # pooled data leaks eval-split information into the IPW weights that the
    # eval split is then scored with.
    N_total = len(T)
    train_idx, eval_idx, n_train = split_indices(N_total, train_frac, seed + 1)

    X = standardize_train_fit(X_raw, fit_idx=train_idx)
    e1 = _fit_propensity(X, T, fit_idx=train_idx)
    E = np.stack([1.0 - e1, e1], axis=1)
    e_T = E[np.arange(len(T)), T]

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

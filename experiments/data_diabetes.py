"""
Diabetes 130-US Hospitals loader (UCI id=296) — the suite's LARGE real
observational dataset (~100k encounters, 1999-2008, 130 hospitals),
addressing the sample-size question ACTG 175 cannot (an RCT is small by
nature; this is not).

Treatment (three arms, following Strack et al. 2014, who analysed exactly
this intervention on this data):

    0  HbA1c not measured during the encounter        (the historical default)
    1  HbA1c measured, no medication change
    2  HbA1c measured AND diabetes medication changed

Outcome  Y = 1{not readmitted within 30 days}  (higher is better).
Capacity Testing / care-management capacity is the genuine constraint:
         historically only ~18% of encounters got an HbA1c test. Caps
         b = (1.0, 0.25, 0.25) — if testing-and-acting helps (Strack et
         al.'s finding), a learned policy pushes both testing arms to
         their caps, so the constraint binds from above.

Identification caveat (stated, not hidden): assignment is observational and
the "treatment" occurs during the encounter, so ignorability given the
admission-level covariates below is an assumption — weaker than ACTG's
randomization. The two datasets are complements: ACTG buys identification,
this buys scale.

Preprocessing follows the standard protocol for this dataset: one encounter
per patient (the first), encounters ending in death or hospice discharge
removed (readmission undefined), admission-level covariates only (nothing
downstream of the treatment): demographics, admission type/source, service
utilisation history, counts of procedures/medications/diagnoses.
Multinomial propensity and standardization are fit on the TRAINING split
only, per the project's leak-fixed convention.
"""

import os

import numpy as np
import pandas as pd

from experiments.common import (
    fit_multinomial_propensity,
    split_indices,
    standardize_train_fit,
)

RAW_CSV = "data/uci/diabetes130.csv"
CACHE_DIR = "data/uci"

# discharge_disposition_id codes for expired / hospice (readmission undefined)
DEATH_HOSPICE = {11, 13, 14, 19, 20, 21}

NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures",
           "num_medications", "number_outpatient", "number_emergency",
           "number_inpatient", "number_diagnoses"]
CATEG = ["race", "gender", "medical_specialty", "admission_type_id",
         "admission_source_id"]
T_ARMS = 3


def _download():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=296)
    df = pd.concat([ds.data.ids, ds.data.features, ds.data.targets], axis=1)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    return df


def load_diabetes(train_frac=0.7, seed=0):
    df = pd.read_csv(RAW_CSV) if os.path.exists(RAW_CSV) else _download()

    # one encounter per patient, in encounter order (standard protocol)
    if "patient_nbr" in df.columns:
        sort_key = "encounter_id" if "encounter_id" in df.columns else None
        if sort_key:
            df = df.sort_values(sort_key)
        df = df.drop_duplicates("patient_nbr", keep="first")
    df = df[~df["discharge_disposition_id"].isin(DEATH_HOSPICE)]
    df = df.reset_index(drop=True)

    # arms from HbA1c measurement x medication change
    a1c = df["A1Cresult"].fillna("None").astype(str)
    measured = a1c.str.strip().ne("None") & a1c.str.strip().ne("nan")
    changed = df["change"].astype(str).str.strip().eq("Ch")
    T = np.where(~measured, 0, np.where(changed, 2, 1)).astype(np.int64)

    Y = (df["readmitted"].astype(str).str.strip() != "<30").astype(float).values

    # age buckets '[70-80)' -> midpoint 75
    age = (df["age"].astype(str).str.extract(r"\[(\d+)-(\d+)\)")
           .astype(float).mean(axis=1).fillna(50.0).values)
    cols = [age]
    for c in NUMERIC:
        cols.append(df[c].astype(float).fillna(0.0).values)
    for c in CATEG:
        s = df[c].fillna("Missing").astype(str).str.strip()
        order = s.value_counts().index
        cols.append(s.map({v: i for i, v in enumerate(order)}).astype(float).values)
    X_raw = np.column_stack(cols)

    N_total = len(T)
    tr, ev, n_train = split_indices(N_total, train_frac, seed)
    X = standardize_train_fit(X_raw, fit_idx=tr)
    E = fit_multinomial_propensity(X, T, T_ARMS, clip=0.02, fit_idx=tr)
    e_T = E[np.arange(N_total), T]

    def _slice(I):
        return {"X": X[I].copy(), "T": T[I].copy(), "Y": Y[I].copy(),
                "e_T": e_T[I].copy(), "E": E[I].copy(),
                "Y_pot": np.full((len(I), T_ARMS), np.nan),
                "Beta": np.zeros((T_ARMS, X.shape[1])),
                "Alpha": np.zeros((T_ARMS, X.shape[1]))}

    cfg = {"N": int(n_train), "T": T_ARMS, "D": int(X.shape[1]), "TAU": 0.1,
           "B": np.array([1.0, 0.25, 0.25])}
    shares = np.bincount(T, minlength=T_ARMS) / N_total
    print(f"[diabetes] N={N_total} (one per patient, death/hospice removed)  "
          f"train={n_train}  D={cfg['D']}  arm shares={np.round(shares, 3)}  "
          f"P(no readmit<30) by arm="
          f"{np.round([Y[T == a].mean() for a in range(T_ARMS)], 3)}")
    return _slice(tr), _slice(ev), cfg

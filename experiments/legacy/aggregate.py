"""
[LEGACY — superseded by the queue-sim cell/sweep pipeline; kept for reproducing the old results/sweep.csv artifacts]
Concatenate per-cell CSVs into one results/sweep.csv."""

import glob
import os

import pandas as pd


CELL_GLOB = "results/cells/cell_*.csv"
SWEEP_CSV = "results/sweep.csv"


def main():
    paths = sorted(glob.glob(CELL_GLOB))
    if not paths:
        print(f"[aggregate] no per-cell CSVs found at {CELL_GLOB}")
        return None

    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["method", "N", "seed"], kind="stable").reset_index(drop=True)

    os.makedirs(os.path.dirname(SWEEP_CSV), exist_ok=True)
    tmp = SWEEP_CSV + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, SWEEP_CSV)

    print(f"[aggregate] {len(paths)} per-cell CSVs -> {SWEEP_CSV}  "
          f"({len(df)} rows, methods={df['method'].nunique()}, "
          f"N={sorted(df['N'].unique())}, "
          f"seeds-per-cell={df.groupby(['method','N']).size().min()}..."
          f"{df.groupby(['method','N']).size().max()})")
    return df


if __name__ == "__main__":
    main()

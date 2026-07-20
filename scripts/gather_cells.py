"""Collect per-cell CSVs written by the SLURM array runner into one sweep CSV.

Safe to run while jobs are still in flight — it takes whatever has landed and
reports what is missing. Uses per-path `stat` rather than a directory listing,
because on NFS the directory entry cache can lag behind files written by other
nodes (a listing can look empty while the files are readable).

Usage:
    python scripts/gather_cells.py criteo
    python scripts/gather_cells.py nonnested --out results/nonnested_sweep_seeds.csv
"""

import argparse
import importlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cell_core import cell_csv_path  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Gather SLURM cell CSVs.")
    p.add_argument("dataset", choices=["criteo", "lalonde", "nonnested", "synth"])
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--worklist", type=str, default=None,
                   help="Defaults to logs/slurm_<dataset>/worklist.txt")
    args = p.parse_args()

    mod = importlib.import_module(f"experiments.run_cell_{args.dataset}")
    cell_dir = mod.CELL_DIR
    worklist = args.worklist or f"logs/slurm_{args.dataset}/worklist.txt"
    out = args.out or f"results/{args.dataset}_sweep_seeds.csv"

    if not os.path.exists(worklist):
        raise SystemExit(f"no worklist at {worklist}; pass --worklist")

    with open(worklist) as f:
        cells = [tuple(int(x) for x in line.split()) for line in f if line.strip()]

    frames, missing = [], []
    for N, seed in cells:
        path = cell_csv_path(cell_dir, N, seed)
        if os.path.exists(path):           # per-path stat, not a dir listing
            try:
                frames.append(pd.read_csv(path))
            except Exception as e:         # noqa: BLE001  partially-written file
                missing.append((N, seed, f"unreadable: {e}"))
        else:
            failed = os.path.join(cell_dir, f"cell_N{N}_seed{seed}.FAILED")
            missing.append((N, seed, "FAILED" if os.path.exists(failed) else "pending"))

    print(f"[gather] {args.dataset}: {len(frames)}/{len(cells)} cells present")
    if missing:
        n_failed = sum(1 for m in missing if m[2] == "FAILED")
        print(f"[gather] missing {len(missing)} ({n_failed} FAILED, "
              f"{len(missing) - n_failed} pending/unreadable)")
        for m in missing[:10]:
            print(f"           N={m[0]} seed={m[1]}: {m[2]}")
        if len(missing) > 10:
            print(f"           … and {len(missing) - 10} more")

    if not frames:
        raise SystemExit("[gather] nothing to write")

    df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[gather] wrote {out}  shape={df.shape}  "
          f"N={sorted(df['N'].unique())}  seeds={df['seed'].nunique()}")


if __name__ == "__main__":
    main()

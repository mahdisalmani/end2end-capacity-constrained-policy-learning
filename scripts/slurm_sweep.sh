#!/bin/bash
# Submit one SLURM job per (N, seed) cell instead of running a local pool.
#
# Cells are already independent and resumable — each writes its own
# results/<dataset>_cells/cell_N{N}_seed{s}.csv and skips if it exists — so
# they map cleanly onto an array job. This scales past the cores on one node,
# which is what the local `sweep_*` multiprocessing drivers are limited to.
#
# Usage:
#   scripts/slurm_sweep.sh criteo   "500 1000 2000 4000 8000 16000 32000" 6
#   scripts/slurm_sweep.sh nonnested "500 1000 2000" 6 --steps 1000
#
# Then aggregate whatever has landed (safe to run while jobs are still going):
#   python -c "from experiments.sweep_core import gather_results; \
#              from experiments.run_cell_criteo import CELL_DIR; \
#              import sys; \
#              df = gather_results(CELL_DIR, [500,1000], list(range(6))); \
#              df.to_csv('results/criteo_sweep_seeds.csv', index=False)"

set -euo pipefail

DATASET="${1:?usage: slurm_sweep.sh <dataset> \"<N values>\" <n_seeds> [extra cell args]}"
N_VALUES="${2:?}"
N_SEEDS="${3:?}"
shift 3
EXTRA=("$@")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$REPO/logs/slurm_$DATASET"
mkdir -p "$LOGDIR"

# Build the (N, seed) work list; the array index selects a line.
WORKLIST="$LOGDIR/worklist.txt"
: > "$WORKLIST"
for N in $N_VALUES; do
  for ((s = 0; s < N_SEEDS; s++)); do
    echo "$N $s" >> "$WORKLIST"
  done
done
NCELLS=$(wc -l < "$WORKLIST")
echo "[slurm] $DATASET: $NCELLS cells -> array 1-$NCELLS"

sbatch --array="1-$NCELLS" \
       --job-name="cell_$DATASET" \
       --output="$LOGDIR/%A_%a.out" \
       --time=02:00:00 \
       --cpus-per-task=1 \
       --mem=8G \
       --wrap "
set -e
cd $REPO
read N SEED < <(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" $WORKLIST)
echo \"[cell] dataset=$DATASET N=\$N seed=\$SEED\"
export OMP_NUM_THREADS=1
python3 -m experiments.run_cell_$DATASET --N \$N --seed \$SEED ${EXTRA[*]:-}
"

#!/bin/bash
# Submit one SLURM job per (N, seed) cell instead of running a local pool.
#
# Cells are already independent and resumable — each writes its own
# results/<dataset>_cells/cell_N{N}_seed{s}.csv and skips if it exists — so
# they map directly onto an array job and scale past the cores on one node.
#
# Usage:
#   scripts/slurm_sweep.sh criteo "500 1000 2000" 10 --variant full --steps 500
#   scripts/slurm_sweep.sh nonnested "500 1000 2000 4000" 10 --steps 1000
#
# Watch:   squeue -u $USER
# Gather:  python scripts/gather_cells.py <dataset>
#
# Note: a generated job script is used rather than `sbatch --wrap`, because
# --wrap does not survive bash process substitution or nested quoting.

set -euo pipefail

DATASET="${1:?usage: slurm_sweep.sh <dataset> \"<N values>\" <n_seeds> [extra cell args]}"
N_VALUES="${2:?}"
N_SEEDS="${3:?}"
shift 3
EXTRA="$*"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# SWEEP_SUFFIX isolates concurrent submissions of the same dataset with
# different EXTRA args (each submission's worklist/job script is read at
# job runtime, so sharing a LOGDIR would race).
LOGDIR="$REPO/logs/slurm_$DATASET${SWEEP_SUFFIX:-}"
mkdir -p "$LOGDIR"

# Work list: one "N seed" pair per line; the array index selects a line.
WORKLIST="$LOGDIR/worklist.txt"
: > "$WORKLIST"
for N in $N_VALUES; do
  for ((s = 0; s < N_SEEDS; s++)); do
    printf '%s %s\n' "$N" "$s" >> "$WORKLIST"
  done
done
NCELLS=$(wc -l < "$WORKLIST")

JOBSCRIPT="$LOGDIR/job.sh"
cat > "$JOBSCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=cell_$DATASET
#SBATCH --output=$LOGDIR/%A_%a.out
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -e
cd "$REPO"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$REPO:\${PYTHONPATH:-}"
LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "$WORKLIST")
N=\$(echo "\$LINE" | awk '{print \$1}')
SEED=\$(echo "\$LINE" | awk '{print \$2}')
echo "[cell] dataset=$DATASET N=\$N seed=\$SEED host=\$(hostname)"
python3 -m experiments.run_cell_$DATASET --N "\$N" --seed "\$SEED" $EXTRA
EOF
chmod +x "$JOBSCRIPT"

echo "[slurm] $DATASET: $NCELLS cells -> array 1-$NCELLS"
sbatch --array="1-$NCELLS" "$JOBSCRIPT"

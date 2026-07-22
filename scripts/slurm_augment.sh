#!/bin/bash
# Submit one SLURM job per existing cell to append missing S2 baselines
# (default: the capacity-matched `mlp`). Reuses each dataset's original
# worklist so the augmentation covers exactly the cells that exist.
#
# Usage:
#   scripts/slurm_augment.sh "adultsemi actg criteo lalonde nonnested diabetes"
#
# Watch:   squeue -u $USER
# Gather:  python3 scripts/gather_cells.py <dataset>   (per dataset, after)

set -euo pipefail

DATASETS="${1:?usage: slurm_augment.sh \"<dataset> [dataset ...]\" [methods]}"
METHODS="${2:-mlp}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$REPO/logs/slurm_augment"
mkdir -p "$LOGDIR"

WORKLIST="$LOGDIR/worklist.txt"
: > "$WORKLIST"
for ds in $DATASETS; do
  src="$REPO/logs/slurm_$ds/worklist.txt"
  if [[ ! -f "$src" ]]; then
    echo "[augment] WARNING: no worklist for $ds at $src — skipped" >&2
    continue
  fi
  while read -r N SEED; do
    [[ -n "$N" ]] && printf '%s %s %s\n' "$ds" "$N" "$SEED" >> "$WORKLIST"
  done < "$src"
done
NCELLS=$(wc -l < "$WORKLIST")

JOBSCRIPT="$LOGDIR/job.sh"
cat > "$JOBSCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=augment_s2
#SBATCH --output=$LOGDIR/%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -e
cd "$REPO"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$REPO:\${PYTHONPATH:-}"
LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "$WORKLIST")
DS=\$(echo "\$LINE" | awk '{print \$1}')
N=\$(echo "\$LINE" | awk '{print \$2}')
SEED=\$(echo "\$LINE" | awk '{print \$3}')
echo "[augment] dataset=\$DS N=\$N seed=\$SEED host=\$(hostname)"
python3 -m experiments.augment_cell --dataset "\$DS" --N "\$N" --seed "\$SEED" --methods $METHODS
EOF
chmod +x "$JOBSCRIPT"

echo "[slurm] augment: $NCELLS cells -> array 1-$NCELLS"
sbatch --array="1-$NCELLS" "$JOBSCRIPT"

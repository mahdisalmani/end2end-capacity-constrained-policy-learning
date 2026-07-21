#!/bin/bash
# Submit the regime map as one SLURM job per (lam, cap_scale, seed) cell.
# Grid: 5 nonlinearity levels x 5 capacity-tightness levels x 5 seeds = 125.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$REPO/logs/slurm_regime"
mkdir -p "$LOGDIR"

LAMS="${1:-0 0.25 0.5 0.75 1.0}"
CAPS="${2:-0.5 0.75 1.0 1.5 2.5}"
NSEEDS="${3:-5}"

WORKLIST="$LOGDIR/worklist.txt"
: > "$WORKLIST"
for lam in $LAMS; do
  for cap in $CAPS; do
    for ((s = 0; s < NSEEDS; s++)); do
      printf '%s %s %s\n' "$lam" "$cap" "$s" >> "$WORKLIST"
    done
  done
done
NCELLS=$(wc -l < "$WORKLIST")

JOBSCRIPT="$LOGDIR/job.sh"
cat > "$JOBSCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=regime_map
#SBATCH --output=$LOGDIR/%A_%a.out
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -e
cd "$REPO"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$REPO:\${PYTHONPATH:-}"
LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "$WORKLIST")
LAM=\$(echo "\$LINE" | awk '{print \$1}')
CAP=\$(echo "\$LINE" | awk '{print \$2}')
SEED=\$(echo "\$LINE" | awk '{print \$3}')
echo "[regime] lam=\$LAM cap=\$CAP seed=\$SEED host=\$(hostname)"
python3 scripts/regime_map.py --lam "\$LAM" --cap-scale "\$CAP" --seed "\$SEED"
EOF
chmod +x "$JOBSCRIPT"

echo "[slurm] regime map: $NCELLS cells -> array 1-$NCELLS"
sbatch --array="1-$NCELLS" "$JOBSCRIPT"

#!/bin/bash
#SBATCH --job-name=direct-rsa-pilot
#SBATCH --partition=psych_day
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-2
#SBATCH --output=logs/direct-rsa-pilot-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/gpfs/milgram/project/turk-browne/kp578/conda_envs/analysis/bin/python3
cd "${SLURM_SUBMIT_DIR}"
SUBJECTS=(sub001 sub002 sub003)
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT=/gpfs/milgram/project/turk-browne/projects/localize_paper/derivatives/direct_rsa_revision_20261107
"$PYTHON" -m analysis.direct_rsa.pilot \
  --config analysis/configs/milgram_direct_rsa.json \
  --subject "$SUBJECT" --n-centers 64 \
  --output "$OUTPUT/pilot/${SUBJECT}.json"

#!/bin/bash
#SBATCH --job-name=audit-legacy-pearson
#SBATCH --partition=psych_day
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/audit-legacy-pearson-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/gpfs/milgram/project/turk-browne/kp578/conda_envs/analysis/bin/python3
cd "${SLURM_SUBMIT_DIR}"
OUTPUT=/gpfs/milgram/project/turk-browne/projects/localize_paper/derivatives/direct_rsa_revision_20261107
"$PYTHON" -m analysis.direct_rsa.audit_legacy \
  --config analysis/configs/milgram_direct_rsa.json \
  --pilot-dir "$OUTPUT/pilot" --output "$OUTPUT/legacy_rad3_audit.json"

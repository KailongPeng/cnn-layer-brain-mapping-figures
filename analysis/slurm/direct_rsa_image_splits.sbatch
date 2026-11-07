#!/bin/bash
#SBATCH --job-name=direct-rsa-image-splits
#SBATCH --partition=psych_day
#SBATCH --array=0-9%2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/direct-rsa-image-splits-%A_%a.out
set -eo pipefail
source /gpfs/milgram/apps/hpc.rhel7/software/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate analysis
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "${SLURM_SUBMIT_DIR}"
ROOT=/gpfs/milgram/project/turk-browne/projects/localize_paper/derivatives/direct_rsa_revision_20261107
"$CONDA_PREFIX/bin/python3" -m analysis.direct_rsa.image_splits batch --root "$ROOT" \
  --batch-index "$SLURM_ARRAY_TASK_ID" --batch-size 10

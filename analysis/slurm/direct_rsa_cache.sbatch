#!/bin/bash
#SBATCH --job-name=direct-rsa-brain-cache
#SBATCH --partition=psych_day
#SBATCH --time=00:30:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/direct-rsa-cache-%A_%a.out
set -eo pipefail
source /gpfs/milgram/apps/hpc.rhel7/software/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate brainiak
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"
CONFIG=analysis/configs/milgram_direct_rsa.json
INDEX=${SLURM_ARRAY_TASK_ID:-0}
SUBJECT=$("$CONDA_PREFIX/bin/python3" -c 'import json,sys;print(json.load(open(sys.argv[1]))["subjects"][int(sys.argv[2])])' "$CONFIG" "$INDEX")
ROOT=/gpfs/milgram/project/turk-browne/projects/localize_paper/derivatives/direct_rsa_revision_20261107
srun --mpi=pmi2 --kill-on-bad-exit=1 "$CONDA_PREFIX/bin/python3" -m analysis.direct_rsa.cache_worker \
  --config "$CONFIG" --root "$ROOT" --subject "$SUBJECT"

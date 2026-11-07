#!/bin/bash
#SBATCH --job-name=direct-rsa-nsd
#SBATCH --partition=psych_day
#SBATCH --time=00:30:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/direct-rsa-nsd-%A_%a.out
set -eo pipefail
source /gpfs/milgram/apps/hpc.rhel7/software/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate brainiak
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"
ROOT=/gpfs/milgram/project/turk-browne/projects/localize_paper/derivatives/direct_rsa_revision_20261107
INDEX=${SLURM_ARRAY_TASK_ID:-0}
MODE=${1:-full}
if [ "$MODE" = check ]; then
  srun --mpi=pmi2 --kill-on-bad-exit=1 "$CONDA_PREFIX/bin/python3" -m analysis.direct_rsa.nsd_worker \
    --manifest "$ROOT/nsd/manifest.json" --job-index "$INDEX" \
    --output-dir "$ROOT/nsd/check/job${INDEX}" --check-centers 64
elif [ "$MODE" = validation ]; then
  srun --mpi=pmi2 --kill-on-bad-exit=1 "$CONDA_PREFIX/bin/python3" -m analysis.direct_rsa.nsd_worker \
    --manifest "$ROOT/nsd/manifest.json" --job-index "$INDEX" \
    --output-dir "$ROOT/nsd/validation/job${INDEX}"
elif [ "$MODE" = full ]; then
  srun --mpi=pmi2 --kill-on-bad-exit=1 "$CONDA_PREFIX/bin/python3" -m analysis.direct_rsa.nsd_worker \
    --manifest "$ROOT/nsd/manifest.json" --job-index "$INDEX" \
    --output-dir "$ROOT/nsd/full/job${INDEX}"
else
  exit 2
fi

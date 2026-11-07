#!/bin/bash
#SBATCH --job-name=direct-rsa-model-identity
#SBATCH --partition=psych_day
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/direct-rsa-model-identity-%j.out
set -eo pipefail
source /gpfs/milgram/apps/hpc.rhel7/software/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/tf1_gpu
set -u
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"
PROJECT=/gpfs/milgram/project/turk-browne/projects/localize_paper
"$CONDA_PREFIX/bin/python3" -m analysis.direct_rsa.verify_stimulus_model \
  --project "$PROJECT" \
  --output "$PROJECT/derivatives/direct_rsa_revision_20261107/provenance/model_identity_${SLURM_JOB_ID}.json"

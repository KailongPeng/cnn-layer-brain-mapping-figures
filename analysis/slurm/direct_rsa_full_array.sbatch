#!/bin/bash
#SBATCH --job-name=direct-rsa-full
#SBATCH --partition=psych_day
#SBATCH --array=0-29%4
#SBATCH --time=00:30:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=logs/direct-rsa-full-%A_%a.out
set -eo pipefail
source /gpfs/milgram/apps/hpc.rhel7/software/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate brainiak
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"
CONFIG=analysis/configs/milgram_direct_rsa.json
SUBJECT=$("$CONDA_PREFIX/bin/python3" -c 'import json,sys; print(json.load(open(sys.argv[1]))["subjects"][int(sys.argv[2])])' "$CONFIG" "$SLURM_ARRAY_TASK_ID")
OUTPUT=/gpfs/milgram/project/turk-browne/projects/localize_paper/derivatives/direct_rsa_revision_20261107
srun --mpi=pmi2 --kill-on-bad-exit=1 "$CONDA_PREFIX/bin/python3" -m analysis.direct_rsa.worker \
  --config "$CONFIG" --subject "$SUBJECT" --output-dir "$OUTPUT/full/$SUBJECT"

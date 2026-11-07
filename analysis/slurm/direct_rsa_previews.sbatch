#!/bin/bash
#SBATCH --job-name=direct-rsa-previews
#SBATCH --partition=psych_day
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=logs/direct-rsa-previews-%j.out
set -eo pipefail
source /gpfs/milgram/apps/hpc.rhel7/software/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate analysis
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"
PROJECT=/gpfs/milgram/project/turk-browne/projects/localize_paper
OUTPUT="$PROJECT/derivatives/direct_rsa_revision_20261107"
"$CONDA_PREFIX/bin/python3" -m analysis.direct_rsa.preview --group-dir "$OUTPUT/group" --output-dir "$OUTPUT/previews"
"$CONDA_PREFIX/bin/python3" -m analysis.direct_rsa.roi_summary \
  --config analysis/configs/milgram_direct_rsa.json --input-dir "$OUTPUT/full" \
  --group-dir "$OUTPUT/group" --atlas-mat "$PROJECT/rebuttal/analyses/vtpm.mat" --output-dir "$OUTPUT/roi"

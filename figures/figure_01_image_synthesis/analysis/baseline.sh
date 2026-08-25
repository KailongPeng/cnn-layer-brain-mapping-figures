#!/usr/bin/env bash
#SBATCH --output=logs/%A_%a.out
#SBATCH --job-name baseline
#SBATCH --partition=psych_scavenge
#SBATCH --time=2:00:00 #20:00:00
#SBATCH --mem=70GB

set -euo pipefail

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/tf1_copy

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

echo python3 -u figures/figure_01_image_synthesis/analysis/baseline.py "$SLURM_ARRAY_TASK_ID"
python3 -u figures/figure_01_image_synthesis/analysis/baseline.py "$SLURM_ARRAY_TASK_ID"

echo "done"

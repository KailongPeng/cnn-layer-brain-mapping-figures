#!/usr/bin/env bash
#SBATCH -p psych_scavenge
#SBATCH --job-name=dataPrepare_brain
#SBATCH --output=logs/%A_%a.out
#SBATCH --time=6:00:00
#SBATCH --mem=100g

set -e
source env_setup_analysis.sh

echo python3 -u figures/figure_04_nsd_localization/analysis/nsd_dataPrepare/brainData/dataPrepare_brain.py "${SLURM_ARRAY_TASK_ID}" "${1}"
python3 -u figures/figure_04_nsd_localization/analysis/nsd_dataPrepare/brainData/dataPrepare_brain.py "${SLURM_ARRAY_TASK_ID}" "${1}"

echo "done"
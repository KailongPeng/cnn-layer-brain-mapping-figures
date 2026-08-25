#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name=nsdResidualTrain
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --mem=10g
#SBATCH --requeue

set -e
source env_setup_analysis.sh

echo python3 -u figures/figure_06_residual_localization/analysis/searchlight_train_linearModel/searchlight_train_linearModel.py "${1}" "${2}"
python3 -u figures/figure_06_residual_localization/analysis/searchlight_train_linearModel/searchlight_train_linearModel.py "${1}" "${2}"

echo "done"
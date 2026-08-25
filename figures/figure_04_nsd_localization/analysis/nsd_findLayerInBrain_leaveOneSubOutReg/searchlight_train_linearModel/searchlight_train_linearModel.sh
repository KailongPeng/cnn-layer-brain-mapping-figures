#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name nsdTrain
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00 #20:00:00
#SBATCH --mem=10g

set -e

source env_setup_analysis.sh
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo python3 -u figures/figure_04_nsd_localization/analysis/nsd_findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/searchlight_train_linearModel.py "${1}" "${2}"
python3 -u figures/figure_04_nsd_localization/analysis/nsd_findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/searchlight_train_linearModel.py "${1}" "${2}"

echo "done"
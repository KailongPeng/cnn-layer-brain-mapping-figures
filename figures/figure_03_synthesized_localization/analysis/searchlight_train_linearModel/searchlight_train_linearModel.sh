#!/bin/bash
#SBATCH -p psych_scavenge   #,psych_day,psych_gpu,psych_week
#SBATCH --job-name=train
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%J.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e

source env_setup_analysis.sh

echo python3 -u figures/figure_03_synthesized_localization/analysis/searchlight_train_linearModel/searchlight_train_linearModel.py
python3 -u figures/figure_03_synthesized_localization/analysis/searchlight_train_linearModel/searchlight_train_linearModel.py

echo "done"
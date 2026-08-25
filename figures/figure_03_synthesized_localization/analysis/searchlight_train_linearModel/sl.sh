#!/bin/bash
#SBATCH -p psych_scavenge  #psych_day,psych_gpu,,psych_week
#SBATCH --job-name=sl_shuffleNorm
#SBATCH --output=logs/%A_%a.out
#SBATCH --requeue
#SBATCH --time=6:00:00
#SBATCH --mem=5g
#SBATCH -n 2

set -e
source env_setup_brainiak.sh


echo python3 -u figures/figure_03_synthesized_localization/analysis/searchlight_train_linearModel/sl.py "${SLURM_ARRAY_TASK_ID}"
python3 -u figures/figure_03_synthesized_localization/analysis/searchlight_train_linearModel/sl.py "${SLURM_ARRAY_TASK_ID}"

echo "done"
#!/bin/bash
#SBATCH -p psych_scavenge  # psych_day,psych_gpu,psych_scavenge,psych_week
#SBATCH --job-name=convertFeat1toStandSpace
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=2:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e

source env_setup_analysis.sh

echo python3 -u Sfig3/Sfig3a/searchlight/convertFeat1toStandSpace.py "${SLURM_ARRAY_TASK_ID}"
python3 -u Sfig3/Sfig3a/searchlight/convertFeat1toStandSpace.py "${SLURM_ARRAY_TASK_ID}"
echo "done"
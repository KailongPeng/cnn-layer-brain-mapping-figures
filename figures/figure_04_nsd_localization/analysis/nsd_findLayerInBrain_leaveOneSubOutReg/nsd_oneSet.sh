#!/usr/bin/env bash
#SBATCH --output=logs/%A_%a.out
#SBATCH --job-name nsd_oneSet
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00 #20:00:00
#SBATCH --mem=10g

set -e

source env_setup_analysis.sh
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/nsd_oneSet.py "${SLURM_ARRAY_TASK_ID}"
python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/nsd_oneSet.py "${SLURM_ARRAY_TASK_ID}"

echo "done"
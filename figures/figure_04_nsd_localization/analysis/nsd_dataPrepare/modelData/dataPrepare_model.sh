#!/usr/bin/env bash
#SBATCH -p psych_scavenge
#SBATCH --job-name=model
#SBATCH --output=logs/%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --mem=200g

set -e
source env_setup_tf1.sh

echo python3 -u Sfig3/Sfig3b/nsd_dataPrepare/modelData/dataPrepare_model.py "${SLURM_ARRAY_TASK_ID}" "${1}"
python3 -u Sfig3/Sfig3b/nsd_dataPrepare/modelData/dataPrepare_model.py "${SLURM_ARRAY_TASK_ID}" "${1}"

echo "done"
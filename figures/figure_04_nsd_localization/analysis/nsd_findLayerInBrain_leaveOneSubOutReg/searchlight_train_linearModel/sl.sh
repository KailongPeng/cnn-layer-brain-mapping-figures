#!/bin/bash
#SBATCH -p psych_scavenge  #psych_day,psych_gpu,,psych_week
#SBATCH --job-name=nsd_train
#SBATCH --output=logs/%A_%a.out
#SBATCH --requeue
#SBATCH --time=6:00:00  ## 6:00:00 for cor ; 12 for reg
#SBATCH --mem=5g
#SBATCH -n 2

set -e
source env_setup_brainiak.sh

echo srun --mpi=pmi2 --kill-on-bad-exit=1 python3 -u figures/figure_04_nsd_localization/analysis/nsd_findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/sl.py "${SLURM_ARRAY_TASK_ID}" "${1}"
srun --mpi=pmi2 --kill-on-bad-exit=1 python3 -u figures/figure_04_nsd_localization/analysis/nsd_findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/sl.py "${SLURM_ARRAY_TASK_ID}" "${1}"

echo "done"
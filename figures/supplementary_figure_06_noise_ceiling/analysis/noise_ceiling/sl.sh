#!/bin/bash
#SBATCH -p psych_scavenge  #psych_day,psych_gpu,psych_week
#SBATCH --job-name=noiseCeil
#SBATCH --output=logs/%A_%a.out
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --mem=100g

set -e
source env_setup_brainiak.sh

echo srun --mpi=pmi2 --kill-on-bad-exit=1 python3 -u rebuttal/analyses/noise_ceiling/sl.py "${SLURM_ARRAY_TASK_ID}"
srun --mpi=pmi2 --kill-on-bad-exit=1 python3 -u rebuttal/analyses/noise_ceiling/sl.py "${SLURM_ARRAY_TASK_ID}"

echo "done"

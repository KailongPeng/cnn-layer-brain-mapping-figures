#!/bin/bash
#SBATCH -p psych_day  #psych_scavenge,psych_gpu,psych_week
#SBATCH --job-name=sl_stability
#SBATCH --output=logs/%A_%a.out
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH --time=1:30:00
#SBATCH --mem=60g

set -e
source env_setup_brainiak.sh

echo srun --mpi=pmi2 --kill-on-bad-exit=1 python3 -u figures/figure_03_synthesized_localization/analysis/stability/sl.py "${SLURM_ARRAY_TASK_ID}"
srun --mpi=pmi2 --kill-on-bad-exit=1 python3 -u figures/figure_03_synthesized_localization/analysis/stability/sl.py "${SLURM_ARRAY_TASK_ID}"

echo "done"
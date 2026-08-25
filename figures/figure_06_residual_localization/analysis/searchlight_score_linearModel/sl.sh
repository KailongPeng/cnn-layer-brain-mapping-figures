#!/usr/bin/env bash
#SBATCH -p psych_scavenge
#SBATCH --job-name=nsd_score
#SBATCH --output=logs/%A_%a.out
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=10g
#SBATCH --requeue

set -e
source env_setup_brainiak.sh

echo srun --mpi=pmi2 --kill-on-bad-exit=1 python3 -u figures/figure_06_residual_localization/analysis/searchlight_score_linearModel/sl.py "${SLURM_ARRAY_TASK_ID}" "${1}"
srun --mpi=pmi2 --kill-on-bad-exit=1 python3 -u figures/figure_06_residual_localization/analysis/searchlight_score_linearModel/sl.py "${SLURM_ARRAY_TASK_ID}" "${1}"

echo "done"
#!/bin/bash
# Usage: sbatch --array=1-16 rebuttal/analyses/noise_ceiling/sl_nsd.sh
# Run from localize_paper root directory.
# 16 parallel jobs (one per NSD image set) × 16 MPI workers each.
# After all finish, run noise_ceiling_nsd_merge.py to average across sets.
#
# Parallelization rationale:
#   NSD data = 8 subjects × 16 images ≈ 880 MB (vs synthesized 30×16 ≈ 3.3 GB).
#   --ntasks=16 → 1 master + 15 workers ≈ 5× faster than ntasks=4.
#   --mem=32G   → sufficient for the smaller NSD data, easier to schedule on scavenge.

#SBATCH --job-name=nc_nsd
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH --mem=64G
#SBATCH --output=/gpfs/milgram/project/turk-browne/projects/localize_paper/rebuttal/analyses/logs/sl_nsd_%A_%a.out
#SBATCH --error=/gpfs/milgram/project/turk-browne/projects/localize_paper/rebuttal/analyses/logs/sl_nsd_%A_%a.err

set -e
source env_setup_brainiak.sh

echo "NSD NC set ${SLURM_ARRAY_TASK_ID}/16  (8 MPI tasks, 64G)"
srun --mpi=pmi2 --kill-on-bad-exit=1 python3 -u \
    rebuttal/analyses/noise_ceiling/sl_nsd.py

echo "done set ${SLURM_ARRAY_TASK_ID}"

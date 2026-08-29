#!/bin/bash
#SBATCH --job-name=09b_bs
#SBATCH --partition=psych_scavenge
#SBATCH --requeue
#SBATCH --array=0-9
#SBATCH --time=0:20:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --output=/gpfs/milgram/project/turk-browne/projects/localize_paper/rebuttal/analyses/logs/job_09b_bs_%A_%a.out
#SBATCH --error=/gpfs/milgram/project/turk-browne/projects/localize_paper/rebuttal/analyses/logs/job_09b_bs_%A_%a.err

source /gpfs/milgram/apps/hpc.rhel7/software/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate tf1

SCRIPT=/gpfs/milgram/project/turk-browne/projects/localize_paper/rebuttal/analyses/analysis_09b_nsd_synthetic.py
cd /gpfs/milgram/project/turk-browne/projects/localize_paper

echo "Host: $(hostname)  Array: ${SLURM_ARRAY_TASK_ID}/10  Start: $(date)"
python3 -u "$SCRIPT" --mode bootstrap \
    --chunk_id ${SLURM_ARRAY_TASK_ID} \
    --n_chunks 10 \
    --n_bootstrap 100
echo "Done: $(date)"

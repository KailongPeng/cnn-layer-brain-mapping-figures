#!/bin/bash
#SBATCH --job-name=09b_merge
#SBATCH --partition=psych_scavenge
#SBATCH --requeue
#SBATCH --time=0:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=/gpfs/milgram/project/turk-browne/projects/localize_paper/rebuttal/analyses/logs/job_09b_merge_%j.out
#SBATCH --error=/gpfs/milgram/project/turk-browne/projects/localize_paper/rebuttal/analyses/logs/job_09b_merge_%j.err

source /gpfs/milgram/apps/hpc.rhel7/software/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate analysis

SCRIPT=/gpfs/milgram/project/turk-browne/projects/localize_paper/rebuttal/analyses/analysis_09b_nsd_synthetic.py
cd /gpfs/milgram/project/turk-browne/projects/localize_paper

echo "Host: $(hostname)  Start: $(date)"
python3 -u "$SCRIPT" --mode merge
echo "Done: $(date)"

#!/bin/bash
#SBATCH --job-name=09b_AB
#SBATCH --partition=psych_scavenge
#SBATCH --requeue
#SBATCH --time=0:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=/gpfs/milgram/project/turk-browne/projects/localize_paper/rebuttal/analyses/logs/job_09b_AB_%j.out
#SBATCH --error=/gpfs/milgram/project/turk-browne/projects/localize_paper/rebuttal/analyses/logs/job_09b_AB_%j.err

source /gpfs/milgram/apps/hpc.rhel7/software/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate tf1

SCRIPT=/gpfs/milgram/project/turk-browne/projects/localize_paper/rebuttal/analyses/analysis_09b_nsd_synthetic.py
cd /gpfs/milgram/project/turk-browne/projects/localize_paper

echo "Host: $(hostname)  Start: $(date)"
python3 -u "$SCRIPT" --mode groupAB
echo "Done: $(date)"

#!/bin/bash
#SBATCH -p psych_day  #,psych_scavenge  #,psych_gpu,psych_scavenge,psych_week
#SBATCH --job-name=nsd_main_analysis
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%J.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e
cd /gpfs/milgram/project/turk-browne/projects/localize/analysis
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/myrtSynth_rt
echo python3 -u /gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/swissRoll/swissRoll.py
python3 -u /gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/swissRoll/swissRoll.py
echo "done"
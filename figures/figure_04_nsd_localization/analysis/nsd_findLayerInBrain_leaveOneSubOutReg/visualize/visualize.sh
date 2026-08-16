#!/bin/bash
#SBATCH -p psych_day,psych_gpu,psych_scavenge,psych_week
#SBATCH --job-name=visualize_sl_RSA_findLayerInBrain
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%J.out
#SBATCH --mem=20g
#SBATCH --requeue

set -e
module load FSL/5.0.10
. /nexsan/apps/hpc/Apps/FSL/5.0.10/etc/fslconf/fsl.sh
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
conda activate GLMsingle
echo python3 -u /gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/findLayerInBrain_leaveOneSubOutReg/visualize/visualize.py
python3 -u /gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/findLayerInBrain_leaveOneSubOutReg/visualize/visualize.py
echo "done"
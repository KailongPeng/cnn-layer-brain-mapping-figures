#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name=nsd_allSets
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --mem=10g
#SBATCH --requeue

set -e
source env_setup_analysis.sh

echo python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/nsd_allSetMeanRandomise/nsd_allSetMeanRandomise.py "${1}"
python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/nsd_allSetMeanRandomise/nsd_allSetMeanRandomise.py "${1}"

echo "done"
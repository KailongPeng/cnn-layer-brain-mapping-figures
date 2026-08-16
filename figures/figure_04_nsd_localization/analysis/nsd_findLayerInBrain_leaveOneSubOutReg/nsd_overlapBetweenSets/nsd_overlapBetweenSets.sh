#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name=sl_RSA_findLayerInBrain
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --mem=10g
#SBATCH --requeue

set -e
source env_setup_analysis.sh

echo python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/nsd_overlapBetweenSets/nsd_overlapBetweenSets.py "${1}"
python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/nsd_overlapBetweenSets/nsd_overlapBetweenSets.py "${1}"

echo "done"
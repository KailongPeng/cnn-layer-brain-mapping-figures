#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name=shuffleNorm_randomise
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --mem=10g
#SBATCH --requeue

set -e
source env_setup_analysis.sh

echo python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/randomise/findLayerInBrain_randomise.py "${1}" "${2}"
python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/randomise/findLayerInBrain_randomise.py "${1}" "${2}"

echo "done"
#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name=nsd_clusterm
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --mem=10g
#SBATCH --requeue

set -e
source env_setup_analysis.sh

echo python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/clustermThreshold/clustermThreshold.py
python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/clustermThreshold/clustermThreshold.py

echo "done"
#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name=residual_clusterm
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --mem=10g
#SBATCH --requeue

set -e
source env_setup_analysis.sh

echo python3 -u nsd_residualModel/clustermThreshold/clustermThreshold.py
python3 -u nsd_residualModel/clustermThreshold/clustermThreshold.py

echo "done"
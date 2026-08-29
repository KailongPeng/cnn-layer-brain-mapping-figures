#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name=clusterm
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --mem=10g
#SBATCH --requeue

set -e
source env_setup_analysis.sh

echo python3 -u fig3a/clustermThreshold/clustermThreshold.py
python3 -u fig3a/clustermThreshold/clustermThreshold.py

echo "done"
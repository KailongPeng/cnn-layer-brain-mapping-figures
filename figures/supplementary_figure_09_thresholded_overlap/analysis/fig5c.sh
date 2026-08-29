#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name fig5c
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --mem=10GB

set -e

source env_setup_analysis.sh
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo python3 -u fig5c/fig5c.py
python3 -u fig5c/fig5c.py

echo "done"
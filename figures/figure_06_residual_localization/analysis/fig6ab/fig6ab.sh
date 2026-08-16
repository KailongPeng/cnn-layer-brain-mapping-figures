#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name fig6ab
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --mem=10GB

set -e

source env_setup_analysis.sh
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo python3 -u fig6ab/fig6ab.py
python3 -u fig6ab/fig6ab.py

echo "done"
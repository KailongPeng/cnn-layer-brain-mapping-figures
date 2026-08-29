#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name Sfig8
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --mem=10GB

set -e

source env_setup_analysis.sh
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo python3 -u Sfig8/Sfig8.py
python3 -u Sfig8/Sfig8.py

echo "done"
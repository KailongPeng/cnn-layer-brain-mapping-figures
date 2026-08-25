#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name=nsd_residual
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --mem=10g
#SBATCH --requeue

set -e
source env_setup_analysis.sh

echo python3 -u figures/figure_06_residual_localization/analysis/nsd_allSetMeanRandomise/nsd_allSetMeanRandomise.py
python3 -u figures/figure_06_residual_localization/analysis/nsd_allSetMeanRandomise/nsd_allSetMeanRandomise.py

echo "done"
#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name=display_surface_plot
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --mem=20g

set -e
source env_setup_analysis.sh

echo python3 -u Sfig3/Sfig3b/display_surface_plot/display_surface_plot.py
python3 -u Sfig3/Sfig3b/display_surface_plot/display_surface_plot.py

echo "done"
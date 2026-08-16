#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name=repSta_RSA_sl
#SBATCH --partition=psych_scavenge
#SBATCH --time=6:00:00
#SBATCH --mem=10g

set -e
source env_setup_analysis.sh

echo python3 -u Sfig3/Sfig3b/nsd_repStability_RSA_sl/searchlight.py "${1}" "${2}"
python3 -u Sfig3/Sfig3b/nsd_repStability_RSA_sl/searchlight.py "${1}" "${2}"

echo "done"
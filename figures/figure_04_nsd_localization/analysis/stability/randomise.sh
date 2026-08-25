#!/bin/bash
#SBATCH -p psych_scavenge
#SBATCH --job-name=randomise_repStability
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e

source env_setup_analysis.sh

whichSet_1=${1}
testMode=${2}
echo whichSet_1="${whichSet_1}"
echo testMode="${testMode}"
read -r radius whichSet scratchResultFolder <<< "$(python3 -u figures/figure_04_nsd_localization/analysis/stability/loadParameter.py ${SLURM_ARRAY_TASK_ID} ${whichSet_1})"
echo "radius=${radius} whichSet=${whichSet}"
echo "scratchResultFolder=${scratchResultFolder}"


ALL_SUBJECTS="sub_i1 sub_i2 sub_i3 sub_i4 sub_i5 sub_i6 sub_i7 sub_i8"
echo "$ALL_SUBJECTS"

fileList=""
for sub in $ALL_SUBJECTS
do
  loadDir="${scratchResultFolder}/nsd_result_folder/result/repStability_RSA_sl/${sub}/rad${radius}/whichSet${whichSet}/"
  output=${loadDir}/repStability_RSA_sl.nii.gz

  echo "${output}"
  if [ -f "$output" ]
  then
    echo "$sub"
    fileList="$fileList $output"
  fi
done

echo fileList = "$fileList"

savePath="${scratchResultFolder}/nsd_result_folder/result/repStability_RSA_sl/randomise/rad${radius}/whichSet${whichSet}/"
mkdir -p "${savePath}"
AllSubs="${savePath}/subs_compiled.nii.gz"
outPre="${savePath}/_"


echo python3 -u figures/figure_04_nsd_localization/analysis/loop_fslmerge.py "${AllSubs}" "${fileList}"
python3 -u figures/figure_04_nsd_localization/analysis/loop_fslmerge.py "${AllSubs}" "${fileList}"


if [ "${testMode}" == 1 ]; then
  echo randomise -i "$AllSubs" -o "$outPre" -1 -n 5 -T -C 2.33
  randomise -i "$AllSubs" -o "$outPre" -1 -n 5 -T -C 2.33
fi

if [ "${testMode}" == 0 ]; then
  echo randomise -i "$AllSubs" -o "$outPre" -1 -n 5000 -T -C 2.33
  randomise -i "$AllSubs" -o "$outPre" -1 -n 5000 -T -C 2.33
fi

echo "done"
#!/usr/bin/env bash
#SBATCH -p psych_scavenge
#SBATCH --job-name=randomise_nsd
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=3:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e

source env_setup_analysis.sh

whichSet_1=${1}
testMode=${2}
echo whichSet_1="${whichSet_1}"
echo testMode="${testMode}"
echo "read -r radius method layer negativeCorrelationFlag whichSet scratchResultFolder <<< python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/randomise/loadParameter.py ${SLURM_ARRAY_TASK_ID} ${whichSet_1})"
read -r radius method layer negativeCorrelationFlag whichSet scratchResultFolder <<< "$(python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/randomise/loadParameter.py ${SLURM_ARRAY_TASK_ID} ${whichSet_1})"

echo "radius=${radius} method=${method} layer=${layer} negativeCorrelationFlag=${negativeCorrelationFlag} whichSet=${whichSet}"
echo "scratchResultFolder=${scratchResultFolder}"

ALL_SUBJECTS="sub_i1 sub_i2 sub_i3 sub_i4 sub_i5 sub_i6 sub_i7 sub_i8"
echo "$ALL_SUBJECTS"

fileList=""
for sub in $ALL_SUBJECTS
do
  loadPath="${scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/searchlight_score_linearModel/${sub}/rad${radius}/whichSet${whichSet}/"
  output="${loadPath}/${sub}_layer${layer}_${method}_zscored_R2.nii.gz"
  echo "$output"
  if [ -f "$output" ]
  then
    echo "$sub"
    fileList="$fileList $output"
  fi
done

echo fileList = "$fileList"

savePath="${scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/randomise/layer${layer}/${method}/rad${radius}/whichSet${whichSet}/"
mkdir -p "${savePath}"
AllSubs="${savePath}/subs_compiled.nii.gz"
finalFile="${savePath}/__tfce_corrp_tstat1.nii.gz"

if [[ ! -f "${finalFile}" ]]
then
  echo python3 -u fig3a/shuffleNorm/loop_fslmerge.py "${AllSubs}" "${fileList}"
  python3 -u fig3a/shuffleNorm/loop_fslmerge.py "${AllSubs}" "${fileList}"


  if [ "$negativeCorrelationFlag" == 0 ]; then
    echo "running positive correlation result"
    outPre="${savePath}/_"
  fi

  if [ "${testMode}" == 1 ]; then
    echo randomise -i "$AllSubs" -o "$outPre" -1 -n 50 -T -C 2.33
    randomise -i "$AllSubs" -o "$outPre" -1 -n 50 -T -C 2.33
  fi

  if [ "${testMode}" == 0 ]; then
    echo randomise -i "$AllSubs" -o "$outPre" -1 -n 5000 -T -C 2.33
    randomise -i "$AllSubs" -o "$outPre" -1 -n 5000 -T -C 2.33
  fi

fi
echo "done"
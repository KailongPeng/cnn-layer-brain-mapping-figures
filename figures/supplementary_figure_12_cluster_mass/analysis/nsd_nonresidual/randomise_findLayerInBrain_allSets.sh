#!/usr/bin/env bash
#SBATCH -p psych_scavenge
#SBATCH --job-name=nsd_randomise
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e

source env_setup_analysis.sh

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
testMode=${1}
echo testMode="${testMode}"

read -r radius method layer negativeCorrelationFlag thresholdName threshold testMode subjectDir scratchResultFolder <<< "$(python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/clustermThreshold/loadParameter.py ${SLURM_ARRAY_TASK_ID})"

echo "radius=${radius} method=${method} layer=${layer} negativeCorrelationFlag=${negativeCorrelationFlag}"
echo "thresholdName=${thresholdName} threshold=${threshold} testMode=${testMode}"
echo "subjectDir=${subjectDir} scratchResultFolder=${scratchResultFolder}"


ALL_SUBJECTS="sub_i1 sub_i2 sub_i3 sub_i4 sub_i5 sub_i6 sub_i7 sub_i8"
echo "$ALL_SUBJECTS"

fileList=""
for sub in $ALL_SUBJECTS
do
  loadPath="${scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/meanAcrossSets/${sub}/rad${radius}/"
  input="${loadPath}/${sub}-layer${layer}-${method}.nii.gz"
  echo "$input"
  if [ -f "$input" ]; then
    echo "$sub"
    fileList="$fileList $input"
  fi
done

echo fileList = "$fileList"

savePath="${scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/testClustermThreshold/randomise/rad${radius}/${method}/layer${layer}/threshold_${thresholdName}/"
echo "${savePath}"
mkdir -p "${savePath}"
AllSubs="${savePath}/subs_compiled.nii.gz"

echo python3 -u fig3a/shuffleNorm/loop_fslmerge.py "${AllSubs}" "${fileList}"
python3 -u fig3a/shuffleNorm/loop_fslmerge.py "${AllSubs}" "${fileList}"

if [ "$negativeCorrelationFlag" == 0 ]; then
  echo "running positive correlation result"
  outPre="${savePath}/layer${layer}_threshold_${thresholdName}_"
fi


if [ "${testMode}" == 1 ]; then
  echo randomise -i "$AllSubs" -o "$outPre" -1 -n 5 -T -C "$threshold"
  randomise -i "$AllSubs" -o "$outPre" -1 -n 5 -T -C "$threshold"
fi

if [ "${testMode}" == 0 ]; then
  echo randomise -i "$AllSubs" -o "$outPre" -1 -n 5000 -T -C "$threshold"
  randomise -i "$AllSubs" -o "$outPre" -1 -n 5000 -T -C "$threshold"
fi
echo "done"
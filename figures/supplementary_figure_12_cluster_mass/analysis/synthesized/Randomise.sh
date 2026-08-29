#!/usr/bin/env bash
#SBATCH -p psych_scavenge
#SBATCH --job-name=shuffleNorm_randomise
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=3:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e

source env_setup_analysis.sh
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

read -r radius method layer negativeCorrelationFlag thresholdName threshold testMode scratchResultFolder <<< "$(python3 -u fig3a/clustermThreshold/loadParameter.py ${SLURM_ARRAY_TASK_ID})"
label=layer${layer}-${method}
echo "radius=${radius} method=${method} layer=${layer} label=${label} negativeCorrelationFlag=${negativeCorrelationFlag}"
echo "thresholdName=${thresholdName} threshold=${threshold}"
echo "testMode=${testMode}"
echo "scratchResultFolder=${scratchResultFolder}"

ALL_SUBJECTS="sub001 sub002 sub003 sub004 sub005 sub007 sub008 sub009 sub010 sub011 sub012 sub013 sub014 sub015 sub016 sub017 sub019 sub020 sub021 sub022 sub023 sub024 sub025 sub027 sub028 sub029 sub030 sub031 sub032 sub033"
echo "$ALL_SUBJECTS"

fileList=""
for sub in $ALL_SUBJECTS
do
  loadFolder="${scratchResultFolder}/findLayerInBrain_leaveOneSubOutReg/searchlight_score_linearModel/searchlight/${sub}/rad${radius}/"
  loadFile="${loadFolder}/zscored_R2_${sub}_layer${layer}_${method}.nii.gz"

  echo "$loadFile"
  if [ -f "$loadFile" ]
  then
    echo "$sub"
    fileList="$fileList $loadFile"
  fi
done

echo fileList = "$fileList"

saveToFolder="${scratchResultFolder}/findLayerInBrain_leaveOneSubOutReg/testClustermThreshold/randomise/layer${layer}/${method}/rad${radius}/threshold_${thresholdName}/"
mkdir -p "${saveToFolder}"
AllSubs="${saveToFolder}/subs_compiled.nii.gz"


echo python3 -u fig3a/shuffleNorm/loop_fslmerge.py "${AllSubs}" "${fileList}"
python3 -u fig3a/shuffleNorm/loop_fslmerge.py "${AllSubs}" "${fileList}"

if [ "$negativeCorrelationFlag" == 0 ]; then
  echo "running positive correlation result"
  outPre="${saveToFolder}/_"
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
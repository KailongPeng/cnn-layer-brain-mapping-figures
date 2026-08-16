#!/usr/bin/env bash
#SBATCH -p psych_scavenge
#SBATCH --job-name=randomise_repStability
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e

source env_setup_analysis.sh

testMode=${1}
echo testMode="${testMode}"
read -r radius thresholdName threshold testMode subjectDir scratchResultFolder <<< "$(python3 -u fig4/nsd_findLayerInBrain_leaveOneSubOutReg/clustermThreshold/repStability_clusterm/loadParameter.py ${SLURM_ARRAY_TASK_ID})"
echo "radius=${radius} thresholdName=${thresholdName} threshold=${threshold} testMode=${testMode}"
echo "subjectDir=${subjectDir} scratchResultFolder=${scratchResultFolder}"

ALL_SUBJECTS="sub_i1 sub_i2 sub_i3 sub_i4 sub_i5 sub_i6 sub_i7 sub_i8"
echo "$ALL_SUBJECTS"

fileList=""
for sub in $ALL_SUBJECTS
do
  input="${scratchResultFolder}/nsd_result_folder/result/repStability_RSA_sl/${sub}/rad${radius}/repStability_allSetsMean.nii.gz"
  echo "${input}"
  if [ -f "$input" ]
  then
    echo "$sub"
    fileList="$fileList $input"
  fi
done

echo fileList = "$fileList"

savePath="${scratchResultFolder}/nsd/repStability_RSA_sl_clusterm/nonresidual/rad${radius}/thresholdName${thresholdName}/"
mkdir -p "${savePath}"
AllSubs="${savePath}/subs_compiled.nii.gz"
outPre="${savePath}/thresholdName${thresholdName}_"

echo python3 -u fig3a/shuffleNorm/loop_fslmerge.py "${AllSubs}" "${fileList}"
python3 -u fig3a/shuffleNorm/loop_fslmerge.py "${AllSubs}" "${fileList}"


if [ "${testMode}" == 1 ]; then
  echo randomise -i "$AllSubs" -o "$outPre" -1 -n 5 -T -C "$threshold"
  randomise -i "$AllSubs" -o "$outPre" -1 -n 5 -T -C "$threshold"
fi

if [ "${testMode}" == 0 ]; then
  echo randomise -i "$AllSubs" -o "$outPre" -1 -n 5000 -T -C "$threshold"
  randomise -i "$AllSubs" -o "$outPre" -1 -n 5000 -T -C "$threshold"
fi

echo "done"
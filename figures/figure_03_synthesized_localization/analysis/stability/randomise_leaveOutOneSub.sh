#!/bin/bash
#SBATCH -p psych_day,psych_gpu,psych_week,psych_scavenge
#SBATCH --job-name=randomise_repStability
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=3:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=10g
#SBATCH --requeue


set -e
module load FSL/5.0.10
. /nexsan/apps/hpc/Apps/FSL/5.0.10/etc/fslconf/fsl.sh

cd /gpfs/milgram/project/turk-browne/projects/localize/analysis/


read -r leaveOutSub radius <<< "$(python3 -u /gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/repStability/RSA/searchlight/loadParameter_leaveOutOneSub.py ${SLURM_ARRAY_TASK_ID})"
echo "leaveOutSub=${leaveOutSub} radius=${radius}"

ALL_SUBJECTS=""
for subj in sub001 sub002 sub003 sub004 sub005 sub007 sub008 sub009 sub010 sub011 sub012 sub013 sub014 sub015 sub016 sub017 sub019 sub020 sub021 sub022 sub023 sub024 sub025 sub027 sub028 sub029 sub030 sub031 sub032 sub033; do
  if [ "$subj" = "$leaveOutSub" ]; then
    echo "skipping ${subj}"
  else
    ALL_SUBJECTS="$ALL_SUBJECTS${subj} "
  fi
done

echo $ALL_SUBJECTS

fileList=""
for sub in $ALL_SUBJECTS
do
  maindir=./subjects/$sub/repStability_RSA_sl/rad${radius}
  output=$maindir/${sub}_repStability_RSA_sl.nii.gz

  echo $maindir
  if [ -f "$output" ]
  then
    echo $sub
    fileList="$fileList $output"
  fi
done

echo fileList = $fileList
pushd /gpfs/milgram/project/turk-browne/projects/localize/analysis > /dev/null


mkdir -p "./results/repStability_RSA_sl/leaveOutSub_${leaveOutSub}/rad${radius}/"
AllSubs=./results/repStability_RSA_sl/leaveOutSub_${leaveOutSub}/rad${radius}/subs_compiled.nii.gz
outPre=./results/repStability_RSA_sl/leaveOutSub_${leaveOutSub}/rad${radius}/_


echo python3 -u /gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/randomise/loop_fslmerge.py "${AllSubs}" "${fileList}"
python3 -u /gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/randomise/loop_fslmerge.py "${AllSubs}" "${fileList}"

echo randomise -i $AllSubs -o $outPre -1 -n 5000 -T -C 2.33
randomise -i $AllSubs -o $outPre -1 -n 5000 -T -C 2.33

echo "done"
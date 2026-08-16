#!/usr/bin/env bash
#SBATCH -p psych_scavenge
#SBATCH --job-name=randomise_repStability
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=3:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e

source env_setup_analysis.sh

read -r radius subjectDir scratchResultFolder <<< "$(python3 -u Sfig3/Sfig3a/searchlight/loadParameter.py ${SLURM_ARRAY_TASK_ID})"
echo "radius=${radius} subjectDir=${subjectDir} scratchResultFolder=${scratchResultFolder}"

ALL_SUBJECTS="sub001 sub002 sub003 sub004 sub005 sub007 sub008 sub009 sub010 sub011 sub012 sub013 sub014 sub015 sub016 sub017 sub019 sub020 sub021 sub022 sub023 sub024 sub025 sub027 sub028 sub029 sub030 sub031 sub032 sub033"
echo "$ALL_SUBJECTS"

fileList=""
for sub in $ALL_SUBJECTS
do
  maindir=${subjectDir}/$sub/repStability_RSA_sl/rad${radius}
  output=$maindir/${sub}_repStability_RSA_sl.nii.gz
  echo $maindir
  if [ -f "$output" ]
  then
    echo "$sub"
    fileList="$fileList $output"
  fi
done

echo fileList = $fileList

mkdir -p "${scratchResultFolder}/repStability_RSA_sl/rad${radius}/"
AllSubs=${scratchResultFolder}/repStability_RSA_sl/rad${radius}/subs_compiled.nii.gz
outPre=${scratchResultFolder}/repStability_RSA_sl/rad${radius}/_


echo python3 -u fig3a/shuffleNorm/loop_fslmerge.py "${AllSubs}" "${fileList}"
python3 -u fig3a/shuffleNorm/loop_fslmerge.py "${AllSubs}" "${fileList}"

echo randomise -i $AllSubs -o $outPre -1 -n 5000 -T -C 2.33
randomise -i $AllSubs -o $outPre -1 -n 5000 -T -C 2.33

echo "done"
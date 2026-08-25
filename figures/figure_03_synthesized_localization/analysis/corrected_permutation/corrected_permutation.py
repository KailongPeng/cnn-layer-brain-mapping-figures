"""
修正置换检验的调度脚本（dispatcher）
回应审稿意见 R1-P3

参考 fig3a/searchlight_score_linearModel/searchlight_score_linearModel.py
创建 jobarray dict 并提交 sbatch 任务。

用法：
    python3 -u figures/figure_03_synthesized_localization/analysis/corrected_permutation/corrected_permutation.py
"""
import os

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")
import numpy as np
import sys
import time

print(sys.executable)
print(sys.version)

current_dir = os.getcwd()
assert current_dir.endswith(('localize_paper', 'cnn-layer-brain-mapping-figures')), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "figures/figure_03_synthesized_localization/analysis"))

from utils import load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray, getSubs

subs = getSubs()

testMode = False


def sl(testMode=False, subs=[]):
    jobarrayDict = {}
    jobarrayID = 1

    if testMode:
        subs = subs[0:1]

    shuffleNormCorrelationSquare_flag = False
    for method in ['reg']:
        for sub in subs:
            for radius in [3]:
                for layer in range(1, 1 + 12):
                    jobarrayDict[jobarrayID] = [sub, radius, method, layer, shuffleNormCorrelationSquare_flag]
                    jobarrayID += 1

    np.save(f"figures/figure_03_synthesized_localization/analysis/corrected_permutation/sl_jobArray.npy", jobarrayDict)

    if testMode:
        cmd = f"sbatch --requeue --array=1-1 figures/figure_03_synthesized_localization/analysis/corrected_permutation/sl.sh"
    else:
        cmd = f"sbatch --requeue --array=1-{len(jobarrayDict)} figures/figure_03_synthesized_localization/analysis/corrected_permutation/sl.sh"
    sbatch_response = custom_run(cmd)
    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    if testMode:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
    else:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))


sl(testMode=testMode, subs=subs)

print("done")

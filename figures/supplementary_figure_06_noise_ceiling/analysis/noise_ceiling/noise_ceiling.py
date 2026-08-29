"""
噪声天花板的调度脚本（dispatcher）
回应审稿意见 R1-P1

提交 BrainIAK searchlight + MPI 任务来计算噪声天花板。
加载所有 30 个被试的 cope 数据，在每个 searchlight 球体计算
Leave-One-Subject-Out RSM 相关。

用法：
    python3 -u rebuttal/analyses/noise_ceiling/noise_ceiling.py
"""
import os

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")
import numpy as np
import sys
import time

print(sys.executable)
print(sys.version)

current_dir = os.getcwd()
assert current_dir.endswith('localize_paper'), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")
sys.path.append(current_dir)

from utils import load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray, getSubs

testMode = False


def sl(testMode=False):
    # For noise ceiling, only 1 job (all subjects processed together)
    jobarrayDict = {}
    jobarrayDict[1] = ['synthesized']

    np.save(f"rebuttal/analyses/noise_ceiling/sl_jobArray.npy", jobarrayDict)

    cmd = f"sbatch --requeue --array=1-1 rebuttal/analyses/noise_ceiling/sl.sh"
    sbatch_response = custom_run(cmd)
    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    completed = check_jobArray(jobID=jobID, jobarrayNumber=1)


sl(testMode=testMode)

print("done")

import os
print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")
import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import urllib.request
from tqdm import tqdm
from pprint import pprint
from os.path import join, exists, split
import sys

print(sys.executable)
print(sys.version)
print(sys.version_info)
import pandas as pd
import argparse
import random
from glob import glob
import subprocess
from tqdm import tqdm
import pickle5 as pickle
import time
import warnings

print(f"numpy version = {np.__version__}")
print(f"pandas version = {pd.__version__}")

current_dir = os.getcwd()

assert current_dir.endswith(('localize_paper', 'cnn-layer-brain-mapping-figures')), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")

sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "figures/figure_03_synthesized_localization/analysis"))

from utils import load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray

from utils import getSubs, setupPaths

subs = getSubs()

[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

testMode = False


def convertFeat1toStandSpace(testMode=False, subs=[]):
    jobarrayDict = {}
    jobarrayID = 1

    if testMode:
        subs = subs[0:1]

    for sub in subs:
        for run in range(1, 9):
            if os.path.exists(f"{subjectDir}/{sub}/preprocess/func0{run}.feat"):
                jobarrayDict[jobarrayID] = [sub, run]
                jobarrayID += 1
    save_obj(jobarrayDict, f"figures/figure_03_synthesized_localization/analysis/stability/convertFeat1toStandSpace_jobArray")
    if testMode:
        cmd = f"sbatch --requeue --array=1-2 figures/figure_03_synthesized_localization/analysis/stability/convertFeat1toStandSpace.sh"
    else:
        cmd = f"sbatch --requeue --array=1-{len(jobarrayDict)} figures/figure_03_synthesized_localization/analysis/stability/convertFeat1toStandSpace.sh"
    sbatch_response = custom_run(cmd)
    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    if testMode:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=2)
    else:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))


convertFeat1toStandSpace(testMode=testMode, subs=subs)


def sl(testMode=False, subs=[]):
    jobarrayDict = {}
    jobarrayID = 1
    for sub in subs:
        for radius in [3]:
            jobarrayDict[jobarrayID] = [sub, radius]
            jobarrayID += 1
    np.save(f"figures/figure_03_synthesized_localization/analysis/stability/sl_jobArray.npy",
            jobarrayDict)
    if testMode:
        cmd = f"sbatch --array=1-1 figures/figure_03_synthesized_localization/analysis/stability/sl.sh"
    else:
        cmd = f"sbatch --array=1-{len(jobarrayDict)} figures/figure_03_synthesized_localization/analysis/stability/sl.sh"
    sbatch_response = custom_run(cmd)
    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    if testMode:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
    else:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))


sl(testMode=testMode, subs=subs)
print("sl finished")


def randomise(testMode=False):
    jobarrayDict = {}
    jobarrayID = 1

    for radius in [3]:
        jobarrayDict[jobarrayID] = [radius]
        jobarrayID += 1
    np.save(
        f"figures/figure_03_synthesized_localization/analysis/stability/randomise_repStability_RSA_searchlight_sl_jobArray.npy",
        jobarrayDict)
    if testMode:
        cmd = (f"sbatch --requeue --array=1-1 "
               f"figures/figure_03_synthesized_localization/analysis/stability/randomise.sh")
    else:
        cmd = (f"sbatch --requeue --array=1-{len(jobarrayDict)} "
               f"figures/figure_03_synthesized_localization/analysis/stability/randomise.sh")
    sbatch_response = custom_run(cmd)
    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    if testMode:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
    else:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))


randomise(testMode=testMode)


print("done")
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

assert current_dir.endswith('localize_paper'), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")

sys.path.append(current_dir)

from utils import load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray

from utils import getSubs

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
    np.save(f"fig3a/searchlight_score_linearModel/sl_jobArray.npy", jobarrayDict)
    if testMode:
        cmd = f"sbatch --requeue --array=1-1 fig3a/searchlight_score_linearModel/sl.sh"
    else:
        cmd = f"sbatch --requeue --array=1-{len(jobarrayDict)} fig3a/searchlight_score_linearModel/sl.sh"
    sbatch_response = custom_run(cmd)
    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    if testMode:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
    else:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))


sl(testMode=testMode, subs=subs)

print("done")
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

from utils import load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray, getSubs, setupPaths, \
    check_jobIDs
from utils import custom_and

[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

from utils import get_nsd_Subs
subs = get_nsd_Subs()


whichSet_1 = int(float(sys.argv[1]))
testMode = sys.argv[2] == "True"
print(f"whichSet_1={whichSet_1}")
print(f"testMode={testMode}")


def randomise(testMode=False, whichSet_1=None):
    jobarrayDict = {}
    jobarrayID = 1

    for negativeCorrelationFlag in [0]:
        for radius in [3]:
            for method in ['reg']:
                for layer in range(1, 13):
                    jobarrayDict[jobarrayID] = [radius, method, layer, negativeCorrelationFlag, whichSet_1+1]
                    jobarrayID += 1
    print(f"jobarrayDict={jobarrayDict}")
    np.save(f"fig4/nsd_findLayerInBrain_leaveOneSubOutReg/randomise/Randomise_jobArray_whichSet_1_{whichSet_1}.npy",
            jobarrayDict)
    if testMode:
        cmd = (f"sbatch --requeue --array=1-1 "
               f"fig4/nsd_findLayerInBrain_leaveOneSubOutReg/randomise/Randomise.sh {whichSet_1} 1")
    else:
        cmd = (f"sbatch --requeue --array=1-{len(jobarrayDict)} "
               f"fig4/nsd_findLayerInBrain_leaveOneSubOutReg/randomise/Randomise.sh {whichSet_1} 0")
    sbatch_response = custom_run(cmd)
    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    if testMode:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
    else:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))


randomise(testMode=testMode, whichSet_1=whichSet_1)

print("done")
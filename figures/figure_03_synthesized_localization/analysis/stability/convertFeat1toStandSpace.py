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
from utils import save_nib, setupPaths

[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

jobarrayDict = load_obj(f"Sfig3/Sfig3a/searchlight/convertFeat1toStandSpace_jobArray")
jobarrayID = int(float(sys.argv[1]))
[sub, currRun] = jobarrayDict[jobarrayID]
print(f"sub={sub} currRun={currRun}")


def convertFeat1toStandSpace(sub='', run=1):
    imgList = list(np.arange(1, 17))
    for currImg in tqdm(imgList):
        zstatFile = f"{subjectDir}/{sub}/preprocess/func0{run}.feat/stats/zstat{currImg}.nii.gz"
        zstatFile_stand = f"{subjectDir}/{sub}/preprocess/func0{run}.feat/stats/zstat{currImg}_stand.nii.gz"
        func2stand = f"{subjectDir}/{sub}/preprocess/func0{run}.feat/reg/example_func2standard.mat"
        stand = "/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz"
        cmd = f"flirt -in {zstatFile} -out {zstatFile_stand} -ref {stand} -init {func2stand} -applyxfm"
        custom_run(cmd)


convertFeat1toStandSpace(sub=sub, run=currRun)


print("done")
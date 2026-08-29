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
import nibabel as nib
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

from utils import load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray, setupPaths, \
    check_jobIDs
from utils import custom_and

[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")


def Sfig4a():
    def clustermThreshold():
        cmd = f"sbatch fig3a/clustermThreshold/clustermThreshold.sh"
        sbatch_response = custom_run(cmd)

    clustermThreshold()


Sfig4a()

def Sfig4b():
    def clustermThreshold():
        cmd = f"sbatch fig4/nsd_findLayerInBrain_leaveOneSubOutReg/clustermThreshold/clustermThreshold.sh"
        sbatch_response = custom_run(cmd)

    clustermThreshold()

Sfig4b()

def Sfig4c():
    def clustermThreshold():
        cmd = f"sbatch nsd_residualModel/clustermThreshold/clustermThreshold.sh"
        sbatch_response = custom_run(cmd)

    clustermThreshold()

Sfig4c()

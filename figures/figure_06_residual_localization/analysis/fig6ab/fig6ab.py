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


def nsd_residualModel(whichSets_1=None, testMode=False):
    if whichSets_1 is None:
        whichSets_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    jobarrayDict = {}
    jobarrayID = 1
    for _whichSets_1 in whichSets_1:
        jobarrayDict[jobarrayID] = [_whichSets_1, testMode]
        jobarrayID += 1
    save_obj(jobarrayDict,
             f"nsd_residualModel/nsd_oneSet_jobdict")
    if testMode:
        cmd = (f"sbatch --requeue --array=1-{1} "
               f"nsd_residualModel/nsd_oneSet.sh")
    else:
        cmd = (f"sbatch --requeue --array=1-{len(jobarrayDict)} "
                   f"nsd_residualModel/nsd_oneSet.sh")
    sbatch_response = custom_run(cmd)

    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    if testMode:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
    else:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))

    def nsd_allSetMeanRandomise(
            whichSets_1):
        max_whichSets_1 = np.max(whichSets_1)
        cmd = (f"sbatch --requeue nsd_residualModel/nsd_allSetMeanRandomise/nsd_allSetMeanRandomise.sh"
               f"{max_whichSets_1}")
        sbatch_response = custom_run(cmd)
        jobID = getjobID_num(sbatch_response)
        waitForEnd(jobID)
        check_jobIDs([jobID])

    nsd_allSetMeanRandomise(whichSets_1)


    def testClustermThreshold():
        cmd = f"sbatch nsd_residualModel/testClustermThreshold/clustermThreshold.sh"
        sbatch_response = custom_run(cmd)


nsd_residualModel(whichSets_1=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                  testMode=False)


def fig6a():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap
    continuousCmap = designCmap(cmap='inferno', colors=1200)
    colorBar = True
    forceRun = False
    valueRange = plot_surface_saveImageReload_pdf(
        resultType="NSD residual mean+randomise: repStabilityMasked TFCE masked tstat",
        views=['posterior'], cmap=continuousCmap, colorBar=colorBar,
        vmax=22
    )


fig6a()


def fig6b():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plotMaximalLayer_pdf, designCmap
    continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)
    colorBar = True
    plotMaximalLayer_pdf(resultType="NSD residual: maximally fit layer",
                         views=['lateral', 'medial', 'posterior'],
                         cmap=continuous_Cmap)

fig6b()
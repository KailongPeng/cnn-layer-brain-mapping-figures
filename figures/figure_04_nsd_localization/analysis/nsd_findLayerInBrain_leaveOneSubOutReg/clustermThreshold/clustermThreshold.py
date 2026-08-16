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
from scipy import stats
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

from utils import get_nsd_Subs
subs = get_nsd_Subs()

testMode = False
print(f"testMode={testMode}")


def randomise_findLayerInBrain_allSets(testMode=None):
    jobarrayDict = {}
    jobarrayID = 1
    for negativeCorrelationFlag in [0]:
        for radius in [3]:
            for method in ['reg']:
                for layer in range(1, 13):
                    def getThreshold():
                        from scipy import stats
                        dof = 29
                        for alpha in [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
                            threshold = stats.t.isf(alpha, dof)
                            print(f"alpha={alpha} threshold={threshold}")

                    for thresholdName, threshold in zip([0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
                                                        [1.699, 2.462, 2.756, 3.396, 3.659, 4.254]):
                        jobarrayDict[jobarrayID] = [radius, method, layer, negativeCorrelationFlag,
                                                    thresholdName, threshold, int(testMode)]
                        jobarrayID += 1
    print(f"jobarrayDict={jobarrayDict}")
    np.save(f"fig4/nsd_findLayerInBrain_leaveOneSubOutReg/clustermThreshold/randomise_findLayerInBrain_allSets_jobArray.npy",
            jobarrayDict)
    if testMode:
        cmd = (f"sbatch --requeue --array=1-1 "
               f"fig4/nsd_findLayerInBrain_leaveOneSubOutReg/clustermThreshold/randomise_findLayerInBrain_allSets.sh 1")
    else:
        cmd = (f"sbatch --requeue --array=1-{len(jobarrayDict)} "
               f"fig4/nsd_findLayerInBrain_leaveOneSubOutReg/clustermThreshold/randomise_findLayerInBrain_allSets.sh 0")
    sbatch_response = custom_run(cmd)
    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    if testMode:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
    else:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))


randomise_findLayerInBrain_allSets(testMode=testMode)


def RepStabilityMeanRandomise(testMode=None):


    def randomise_repStability_allSets(testMode=None):
        jobarrayDict = {}
        jobarrayID = 1

        for radius in [3]:
            for thresholdName, threshold in zip([0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
                                                [1.699, 2.462, 2.756, 3.396, 3.659, 4.254]):
                jobarrayDict[jobarrayID] = [radius, thresholdName, threshold, int(testMode)]
                jobarrayID += 1
        np.save(
            f"fig4/nsd_findLayerInBrain_leaveOneSubOutReg/clustermThreshold/repStability_clusterm/randomise_repStability_allSets_jobArray.npy",
            jobarrayDict)
        if testMode:
            cmd = (f"sbatch --requeue --array=1-1 "
                   f"fig4/nsd_findLayerInBrain_leaveOneSubOutReg/clustermThreshold/repStability_clusterm/randomise_repStability_allSets.sh 1 ")
        else:
            cmd = (f"sbatch --requeue --array=1-{len(jobarrayDict)} "
                   f"fig4/nsd_findLayerInBrain_leaveOneSubOutReg/clustermThreshold/repStability_clusterm/randomise_repStability_allSets.sh 0")
        sbatch_response = custom_run(cmd)
        jobID = getjobID_num(sbatch_response)
        waitForEnd(jobID)
        if testMode:
            completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
        else:
            completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))

    randomise_repStability_allSets(testMode=testMode)


RepStabilityMeanRandomise(testMode=testMode)




def Sfig6b_v1():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap
    continuousCmap = designCmap(cmap='inferno', colors=1200)
    colorBar = True
    forceRun = True
    thresholdName=0.001
    valueRange = plot_surface_saveImageReload_pdf(
        resultType=f"nsd-nonresidual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat",
        views=['posterior'], cmap=continuousCmap, colorBar=colorBar,
        vmax=22,
        thresholdName=thresholdName,
        forceRun=forceRun
    )


Sfig6b_v1()


def Sfig6b_v2():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plotMaximalLayer_pdf, designCmap
    continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)
    colorBar = True
    thresholdName = 0.001
    plotMaximalLayer_pdf(resultType=f"nsd-nonresidual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat",
                         views=['lateral', 'medial', 'posterior'],
                         cmap=continuous_Cmap,
                         thresholdName=thresholdName)

Sfig6b_v2()


print("done")
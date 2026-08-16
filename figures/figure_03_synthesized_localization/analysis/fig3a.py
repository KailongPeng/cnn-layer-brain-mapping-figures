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

from utils import (load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray, getSubs, setupPaths,
                   check_jobIDs, save_nib)
subs = getSubs()
[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")


def findLayerInBrain_leaveOneSubOutReg():
    """
        Input data: Each subject has 16 representations for 16 images.
        Searchlight is run for each subject, for each subject, linear regression is done between the model layer
        fingerprint (as x) and voxel cube fingerprint (as y). Every regression weight and intercept is saved.
        For each cross validation rotation, n-1 subjects’ beta and intercept are averaged across subjects. We run
        searchlight on the left out subject. For each voxel cube inside searchlight, we load the n-1 mean beta and
        intercept as trained model to score the voxel cube fingerprint (the outcome is R2). We record this R2 for each
        voxel cube and save it.
        For all 30 rotations, we finally have 30 volumes of R2. We run randomise with TFCE on these 30 volumes.
        Visualize TFCE > 0.95 result.
    """

    def searchlight_train_linearModel():
        cmd = "sbatch fig3a/searchlight_train_linearModel/searchlight_train_linearModel.sh"
        sbatch_response = custom_run(cmd)
        jobID = getjobID_num(sbatch_response)
        waitForEnd(jobID)
        completed = check_jobIDs([jobID])

    searchlight_train_linearModel()

    def searchlight_score_linearModel():
        cmd = "sbatch fig3a/searchlight_score_linearModel/searchlight_score_linearModel.sh"
        sbatch_response = custom_run(cmd)
        jobID = getjobID_num(sbatch_response)
        waitForEnd(jobID)
        completed = check_jobIDs([jobID])

    searchlight_score_linearModel()

    def shuffleNorm():
        cmd = "sbatch fig3a/shuffleNorm/shuffleNorm.sh"
        sbatch_response = custom_run(cmd)
        jobID = getjobID_num(sbatch_response)
        waitForEnd(jobID)
        completed = check_jobIDs([jobID])

    shuffleNorm()

    def visualize():
        cmd = "sbatch fig3a/visualize/visualize.sh"
        sbatch_response = custom_run(cmd)

    visualize()

findLayerInBrain_leaveOneSubOutReg()


def fig3a():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap
    continuousCmap = designCmap(cmap='inferno', colors=1200)
    colorBar = True
    valueRange = plot_surface_saveImageReload_pdf(resultType="synthesized: repStabilityMasked TFCE masked tstat",
                                                  views=['posterior'],
                                                  cmap=continuousCmap,
                                                  colorBar=colorBar,
                                                  vmax=8,
                                                  forceRun=True
                                                  )

fig3a()
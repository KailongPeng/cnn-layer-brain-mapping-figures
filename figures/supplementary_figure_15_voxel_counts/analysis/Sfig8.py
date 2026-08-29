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


def showMatrix(matrix, title='', axis0_ticks=None, axis1_ticks=None, fill_diagonal=False, cmap='seismic',
               fontsize=20 * 5, removeNanFlag=False, savePath=None, showTextFlag=True, vmin=0, vmax=None):
    if axis1_ticks is None:
        axis1_ticks = []
    if axis0_ticks is None:
        axis0_ticks = []
    import matplotlib.pyplot as plt

    if removeNanFlag:
        nan_rows = np.isnan(matrix).any(axis=1)
        nan_cols = nan_rows

        matrix = matrix[~nan_rows, :]
        matrix = matrix[:, ~nan_cols]
        print(f"matrix.shape={matrix.shape}")

        axis0_ticks = [layer for i, layer in enumerate(axis0_ticks) if not nan_cols[i]]
        axis1_ticks = [layer for i, layer in enumerate(axis1_ticks) if not nan_rows[i]]
        print(f"axis0_ticks={axis0_ticks}")
        print(f"axis1_ticks={axis1_ticks}")

    if fill_diagonal:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if axis0_ticks[i] == axis1_ticks[j]:
                    matrix[i, j] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(9 * 10, 8 * 10))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)


    print(f"axis0_ticks={axis0_ticks}")
    print(f"axis1_ticks={axis1_ticks}")
    print(f"matrix.shape={matrix.shape}")
    plt.suptitle(f'{title}', fontsize=fontsize)
    if showTextFlag:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if axis0_ticks[i] == axis1_ticks[j]:
                    pass
                else:
                    ax.text(j, i, round(matrix[i, j], 4),
                            ha="center", va="center", color="k", fontsize=fontsize / 3 * 2)
    cbar = plt.colorbar(im)

    cbar.ax.tick_params(labelsize=fontsize)

    xticks = np.arange(0, matrix.shape[1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(axis1_ticks, rotation=45, ha='center', fontsize=fontsize)
    yticks = np.arange(0, matrix.shape[0])
    ax.set_yticks(yticks)
    ax.set_yticklabels(axis0_ticks, fontsize=fontsize)

    ax.tick_params(axis='both', which='both', labelsize=fontsize, size=fontsize/3)

    if savePath is not None:
        plt.savefig(savePath, transparent=True)
    plt.show()


def Sfig8a():
    def get_repStabilityMaskedLayers_statistics(max_whichSets_1=15):
        loadFolder = f"{scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/randomise/"
        radius = 3
        voxelNumber = np.zeros((max_whichSets_1 + 1, 12))
        for whichSets_1 in tqdm(range(max_whichSets_1 + 1)):
            for layer in range(1, 13):
                repStabilityMasked_ROIx = nib.load(
                    f"{loadFolder}/layer{layer}/reg/rad{radius}/whichSet{whichSets_1 + 1}/"
                    f"repStabilityMasked_layer{layer}.nii.gz").get_fdata()
                voxelNumber[whichSets_1, layer - 1] = np.sum(repStabilityMasked_ROIx)
        showMatrix(voxelNumber, axis0_ticks=[f'set{_set}' for _set in range(1, 17)],
                   axis1_ticks=[f'layer{layer}' for layer in range(1, 13)], cmap='cool', fontsize=20 * 5,
                   fill_diagonal=False, title=f"voxelNumber NSD nonresidual",
                   showTextFlag=False,
                   vmax=7235,
                   savePath=f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/figures/Sfig8a_voxelNumber_NSD_nonresidual_vmax_7235.pdf")

        return voxelNumber

    nsd_findLayerInBrain_leaveOneSubOutReg_voxelNumber = get_repStabilityMaskedLayers_statistics()

Sfig8a()

def Sfig8b():
    def showVoxelNumber():
        def randomiseResultStatistics(_whichSet=None):
            _voxelNumbers = np.zeros((12))
            for layer in range(1, 13):
                path = f"{scratchResultFolder}/nsd/residualModel/" \
                       f"randomise/layer{layer}/reg/rad3/whichSet{_whichSet}/repStabilityMasked_ROI_layer{layer}_whichSet{_whichSet}.nii.gz"
                brain = nib.load(path).get_fdata()
                voxelNumber = np.sum(brain > 0.95)
                _voxelNumbers[layer - 1] = voxelNumber
            return _voxelNumbers

        voxelNumbers_allSets = np.zeros((16, 12))
        for whichSet_1 in range(16):
            voxelNumbers = randomiseResultStatistics(_whichSet=whichSet_1 + 1)
            voxelNumbers_allSets[whichSet_1] = voxelNumbers

        showMatrix(voxelNumbers_allSets, axis0_ticks=[f'set{_set}' for _set in range(1, 17)],
                   axis1_ticks=[f'layer{layer}' for layer in range(1, 13)], cmap='cool', fontsize=20 * 5,
                   fill_diagonal=False, title=f"voxelNumbers NSD residual",
                   showTextFlag=False,
                   savePath=f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/figures/Sfig8b_voxelNumber_NSD_residual_vmax_7235.pdf",
                   vmax=7235)
        return voxelNumbers_allSets

    voxelNumbers_allSets_residualModel = showVoxelNumber()


Sfig8b()
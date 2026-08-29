"""code: analysis/organizedScripts/findLayerInBrain_leaveOneSubOutReg/visualize/visualize.py"""

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

from utils import load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray

from utils import getSubs

subs = getSubs()
from utils import save_nib, setupPaths

[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")


def visualize_sl_RSA_findLayerInBrain(
        randomiseMethod="tfce", radius=3,
        method='reg'):

    maskedROIx = {}

    sign, negativeFlag = 'positive', '_'
    for layer in range(1, 13):
        repStabilityMasked_ROIx = nib.load(f"{scratchResultFolder}"
                                           f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/"
                                           f"layer{layer}/{method}/rad{radius}/"
                                           f"repStabilityMasked__{randomiseMethod}{negativeFlag}_layer{layer}.nii.gz").get_fdata()
        maskedROIx[f"layer{layer}"] = repStabilityMasked_ROIx
    overlapBtweenROIs = np.zeros((12, 12))

    def calLayer_i(_layer_i_):
        if _layer_i_ >= 13:
            sign_i = 'negative'
            layer_ii = _layer_i_ - 12
        else:
            sign_i = 'positive'
            layer_ii = _layer_i_
        _index_i_ = f"layer{layer_ii}"
        return _index_i_

    for layer_i in range(1, 13):
        index_i = calLayer_i(layer_i)
        for layer_j in range(1, 13):
            index_j = calLayer_i(layer_j)
            overlapBtweenROIs[layer_i - 1, layer_j - 1] = np.nansum(
                maskedROIx[index_i] * maskedROIx[index_j]) / np.nansum(maskedROIx[index_i])
    return overlapBtweenROIs


overlapBetweenROIs = visualize_sl_RSA_findLayerInBrain(randomiseMethod="tfce", radius=3, method='reg')


def showMatrix(matrix, title='', axis0_ticks=None, axis1_ticks=None, fill_diagonal=True, cmap='seismic',
               fontsize=20 * 5, removeNanFlag=False, savePath=None, showTextFlag=True):
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
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)


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
    ax.set_xticklabels(axis0_ticks, rotation=45, ha='center', fontsize=fontsize)
    yticks = np.arange(0, matrix.shape[0])
    ax.set_yticks(yticks)
    ax.set_yticklabels(axis1_ticks, fontsize=fontsize)

    ax.tick_params(axis='both', which='both', labelsize=fontsize, size=fontsize/3)

    if savePath is not None:
        plt.savefig(savePath, transparent=True)
    plt.show()


ticksName = [f"layer-{layer}" for layer in range(1, 13)]
showMatrix(overlapBetweenROIs, title='tfce',
           axis0_ticks=ticksName,
           axis1_ticks=ticksName,
           fill_diagonal=True, cmap='Blues', fontsize=20 * 6,
           showTextFlag=False,
           removeNanFlag=True,
           savePath=f"{scratchResultFolder}/figures/fig5b.pdf")

np.save(f"{scratchResultFolder}/findLayerInBrain_leaveOneSubOutReg/overlapBetweenROIs.npy", overlapBetweenROIs)
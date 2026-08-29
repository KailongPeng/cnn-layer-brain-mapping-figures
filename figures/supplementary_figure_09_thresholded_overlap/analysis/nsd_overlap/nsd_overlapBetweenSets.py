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


def save_nib(affine_mat=None, toFill=None, saveToPath=''):
    sl_nii = nib.Nifti1Image(toFill.astype('double'), affine_mat)
    print(f'saving to {saveToPath}')
    nib.save(sl_nii, saveToPath)


max_whichSets_1 = int(float(sys.argv[1]))
print(f"max_whichSets_1={max_whichSets_1}")

testMode = False
global affine_mat
radius = 3


def repStabilityMasking():
    for whichSets_1 in tqdm(range(max_whichSets_1 + 1)):
        repStabilityMask = nib.load(
            f"{scratchResultFolder}/nsd_result_folder/result/repStability_RSA_sl/"
            f"randomise/rad{radius}/whichSet{whichSets_1 + 1}/__tfce_corrp_tstat1.nii.gz")
        affine_mat = repStabilityMask.affine
        repStabilityMask = repStabilityMask.get_fdata() > 0.95

        for layer in range(1, 13):
            layerX = nib.load(
                f"{scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg"
                f"/randomise/layer{layer}/reg/rad{radius}/whichSet{whichSets_1 + 1}/__tfce_corrp_tstat1.nii.gz").get_fdata()
            layerX = layerX > 0.95
            layerX = layerX * repStabilityMask
            save_nib(affine_mat=affine_mat, toFill=layerX,
                     saveToPath=f"{scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg"
                                f"/randomise/layer{layer}/reg/rad{radius}/whichSet{whichSets_1 + 1}/repStabilityMasked_layer{layer}.nii.gz")


repStabilityMasking()


def get_repStabilityMaskedLayers_statistics(max_whichSets_1=15):
    loadFolder = f"{scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/randomise/"
    radius = 3
    voxelNumber = np.zeros((max_whichSets_1 + 1, 12))
    for whichSets_1 in tqdm(range(max_whichSets_1 + 1)):
        for layer in range(1, 13):
            repStabilityMasked_ROIx = nib.load(f"{loadFolder}/layer{layer}/reg/rad{radius}/whichSet{whichSets_1 + 1}/"
                                               f"repStabilityMasked_layer{layer}.nii.gz").get_fdata()
            voxelNumber[whichSets_1, layer - 1] = np.sum(repStabilityMasked_ROIx)
    showMatrix(voxelNumber, axis0_ticks=[f'set{_set}' for _set in range(1, 17)],
               axis1_ticks=[f'layer{layer}' for layer in range(1, 13)], cmap='cool', fontsize=20 * 5,
               fill_diagonal=False, title=f"voxelNumber NSD nonresidual",
               showTextFlag=False,
               vmax=7235,
               savePath=f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/figures/voxelNumber_NSD_nonresidual_vmax_7235.pdf")

    return voxelNumber


nsd_findLayerInBrain_leaveOneSubOutReg_voxelNumber = get_repStabilityMaskedLayers_statistics()


def nsd_sumAcrossSets():
    loadFolder = f"{scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/randomise/"
    affine_mat = nib.load(
        f"{scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/randomise//layer1/reg/rad3/whichSet16/repStabilityMasked_layer1.nii.gz").affine
    for layer in range(1, 13):
        sumAcrossSets = 0
        for whichSets_1 in tqdm(range(max_whichSets_1 + 1)):
            repStabilityMasked_ROIx = nib.load(f"{loadFolder}/layer{layer}/reg/rad{radius}/whichSet{whichSets_1 + 1}/"
                                               f"repStabilityMasked_layer{layer}.nii.gz").get_fdata()
            sumAcrossSets = sumAcrossSets + repStabilityMasked_ROIx
        os.makedirs(f"{loadFolder}/layer{layer}/reg/rad{radius}/allSets/", exist_ok=True)
        save_nib(affine_mat=affine_mat, toFill=sumAcrossSets,
                 saveToPath=f"{loadFolder}/layer{layer}/reg/rad{radius}/allSets/"
                            f"allSetsSummed_RepStabilityMasked_layer{layer}.nii.gz")


nsd_sumAcrossSets()


def overlapBetweenSets():
    loadFolder = f"{scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/randomise/"
    ratios = {}
    for layer in tqdm(range(1, 13)):
        ratio = np.zeros((max_whichSets_1 + 1, max_whichSets_1 + 1))
        for whichSet_1_i in range(max_whichSets_1 + 1):
            for whichSet_1_j in range(max_whichSets_1 + 1):
                ROIi = nib.load(f"{loadFolder}/layer{layer}/reg/rad{radius}/whichSet{whichSet_1_i + 1}/"
                                f"repStabilityMasked_layer{layer}.nii.gz").get_fdata()
                ROIj = nib.load(f"{loadFolder}/layer{layer}/reg/rad{radius}/whichSet{whichSet_1_j + 1}/"
                                f"repStabilityMasked_layer{layer}.nii.gz").get_fdata()
                ratio[whichSet_1_i, whichSet_1_j] = np.sum(ROIi * ROIj) / np.sum(ROIi)
                ratio[whichSet_1_j, whichSet_1_i] = np.sum(ROIi * ROIj) / np.sum(ROIj)
        showMatrix(ratio, title=f"layer{layer}",
                   axis0_ticks=[f'set{i}' for i in range(1, 17)],
                   axis1_ticks=[f'set{i}' for i in range(1, 17)],
                   fill_diagonal=True, cmap='Blues',
                   fontsize=10 * 5)
        ratios[f"layer{layer}"] = ratio




def overlapWithLocalizeLayers():
    loadFolder = f"{scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/randomise/"
    for layer in range(1, 13):
        ROI_nsd = nib.load(f"{loadFolder}/rad{3}/{'reg'}/layer{layer}/"
                           f"repStabilityMasked_layer{layer}__tfce_corrp_tstat1.nii.gz").get_fdata()
        ROI_localize = nib.load(f"{scratchResultFolder}/"
                                f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/"
                                f"layer{layer}/reg/rad3/repStabilityMasked__tfce__layer{layer}.nii.gz").get_fdata()

        overlap = ROI_nsd + ROI_localize
        divide_NSD = np.sum(overlap == 2) / np.sum(ROI_nsd)
        divide_localize = np.sum(overlap == 2) / np.sum(ROI_localize)
        print(
            f"layer{layer} np.sum(ROI_nsd)={np.sum(ROI_nsd)} np.sum(ROI_localize)={np.sum(ROI_localize)} \ndivide_NSD={divide_NSD} divide_localize={divide_localize}\n\n")




def compareEachSetWithLocalize_NSD_NONresidualModel(method='reg', randomiseMethod='tfce', negativeFlag='_', radius=3):
    overlapDivideNSD = np.zeros((12, 16))
    overlapDivideLOC = np.zeros((12, 16))
    for layer in range(1, 13):
        for whichSet in range(1, 17):
            repStabilityMasked_ROI_nsd = nib.load(
                f"{scratchResultFolder}/"
                f"nsd/findLayerInBrain_leaveOneSubOutReg/randomise/"
                f"layer{layer}/reg/rad{radius}/whichSet{whichSet}/"
                f"repStabilityMasked_layer{layer}.nii.gz").get_fdata()

            repStabilityMasked_ROI_loc = nib.load(f"{scratchResultFolder}/"
                                                  f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/"
                                                  f"layer{layer}/{method}/rad{radius}/"
                                                  f"repStabilityMasked__{randomiseMethod}{negativeFlag}_layer{layer}.nii.gz").get_fdata()
            overlapDivideNSD[layer - 1, whichSet - 1] = np.sum(
                repStabilityMasked_ROI_nsd * repStabilityMasked_ROI_loc) / np.sum(repStabilityMasked_ROI_nsd)
            overlapDivideLOC[layer - 1, whichSet - 1] = np.sum(
                repStabilityMasked_ROI_nsd * repStabilityMasked_ROI_loc) / np.sum(repStabilityMasked_ROI_loc)
    showMatrix(overlapDivideNSD, title="overlapDivideNSD_NONresidualModel",
               axis0_ticks=[f"layer{layer}" for layer in range(1, 13)],
               axis1_ticks=[f"set{whichSet}" for whichSet in range(1, 17)],
               fill_diagonal=False)
    showMatrix(overlapDivideLOC, title="overlapDivideLOC_NONresidualModel",
               axis0_ticks=[f"layer{layer}" for layer in range(1, 13)],
               axis1_ticks=[f"set{whichSet}" for whichSet in range(1, 17)],
               fill_diagonal=False)




def nsd_overlapBetweenROIs_statistics(max_whichSets_1=15,
                                      AllLayerOverlap=True):
    def get_overlapBetweenROIs(_max_whichSets_1=15):
        def get_overlapBetweenROIs_forSet(whichSet=None):
            maskedROIx = {}
            for _layer in range(1, 13):
                repStabilityMasked_ROIx = nib.load(f"{scratchResultFolder}/"
                                                   f"nsd/findLayerInBrain_leaveOneSubOutReg/randomise/"
                                                   f"layer{_layer}/reg/rad{3}/whichSet{whichSet}/"
                                                   f"repStabilityMasked_layer{_layer}.nii.gz").get_fdata()
                maskedROIx[f"layer{_layer}"] = repStabilityMasked_ROIx
            overlapBtweenROIs = np.zeros((12, 12))

            for layer_i in range(1, 13):
                index_i = f"layer{layer_i}"
                for layer_j in range(1, 13):
                    index_j = f"layer{layer_j}"
                    overlapBtweenROIs[layer_i - 1, layer_j - 1] = np.nansum(
                        maskedROIx[index_i] * maskedROIx[index_j]) / np.nansum(maskedROIx[index_i])
            return overlapBtweenROIs

        if not os.path.exists(f"{scratchResultFolder}/nsd/"
                              f"findLayerInBrain_leaveOneSubOutReg/meanAcrossSets/"
                              f"overlapBetweenROIs_allSets_NONresidualModel.npy"):
            overlapBetweenROIs_allSets = np.zeros((1 + _max_whichSets_1, 12, 12))
            for whichSet_1 in tqdm(range(0, 1 + _max_whichSets_1)):
                overlapBetweenROIs = get_overlapBetweenROIs_forSet(whichSet=whichSet_1 + 1)
                overlapBetweenROIs_allSets[whichSet_1] = overlapBetweenROIs
            filename = (f"{scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/meanAcrossSets/"
                        f"overlapBetweenROIs_allSets_NONresidualModel.npy")
            directory = os.path.dirname(filename)

            os.makedirs(directory, exist_ok=True)

            print(f"Directory created: {directory}")

            np.save(filename, overlapBetweenROIs_allSets)
        return

    get_overlapBetweenROIs(_max_whichSets_1=15)

    def getNeighborLayerComparisonIndex():
        _ = np.zeros((12, 12))
        __ = np.zeros((12, 12))
        _[np.triu_indices(12, 1)] = 1
        _[np.tril_indices(12, -1)] = 1
        __[np.triu_indices(12, 2)] = 1
        __[np.tril_indices(12, -2)] = 1
        ___ = _ - __
        return ___ == 1

    def getNonDiagonalElements():
        _ = np.ones((12, 12))
        for ii in range(_.shape[0]):
            _[ii, ii] = 0
        return _ == 1

    if AllLayerOverlap:
        matrixIndices = getNonDiagonalElements()
    else:
        matrixIndices = getNeighborLayerComparisonIndex()

    showAllOverlapBetweenROIs_flag = False
    ticksName = [f"layer-{layer}" for layer in range(1, 13)]

    overlapBetweenROIs_allSets = np.load(f"{scratchResultFolder}/nsd/"
                                         f"findLayerInBrain_leaveOneSubOutReg/meanAcrossSets/"
                                         f"overlapBetweenROIs_allSets_NONresidualModel.npy")
    assert overlapBetweenROIs_allSets.shape == (16, 12, 12)
    nsd_overlapBetweenROIs = np.zeros((max_whichSets_1 + 1, 12, 12))
    nsd_means = []
    for whichSet_1 in range(0, 1 + max_whichSets_1):
        overlapBetweenROIs = overlapBetweenROIs_allSets[whichSet_1]
        assert overlapBetweenROIs.shape == (12, 12)
        nsd_overlapBetweenROIs[whichSet_1] = overlapBetweenROIs
        nsd_means.append(np.nanmean(overlapBetweenROIs[matrixIndices]))
        if showAllOverlapBetweenROIs_flag:
            showMatrix(overlapBetweenROIs,
                       title=f'nsd-tfce-NONresidualModel-whichSet{whichSet_1 + 1}',
                       axis0_ticks=ticksName,
                       axis1_ticks=ticksName,
                       fill_diagonal=True, cmap='Blues', fontsize=20 * 2, vmin=0, vmax=1)

    showMatrix(np.nanmean(nsd_overlapBetweenROIs, axis=0), title=f'nsd-tfce-NONresidualModel-meanAcrossSets',
               axis0_ticks=ticksName,
               axis1_ticks=ticksName,
               fill_diagonal=True, cmap='Blues', fontsize=20 * 5, showTextFlag=False, vmin=0, vmax=1,
               savePath=f"{scratchResultFolder}/figures/fig5c_NSD_nonresidual.pdf")
    nsd_means = np.asarray(nsd_means)

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    plt.tight_layout(pad=0, w_pad=0, h_pad=0, rect=(0, 0, 1, 1))
    for ii in range(16):
        plt.subplot(4, 4, ii + 1)
        temp = nsd_overlapBetweenROIs[ii].copy()
        np.fill_diagonal(temp, 0.0)
        plt.imshow(temp, cmap='Blues', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.show()


    overlapBetweenROIs_path = f"{scratchResultFolder}/findLayerInBrain_leaveOneSubOutReg/overlapBetweenROIs.npy"
    loc_overlapBetweenROIs = np.load(overlapBetweenROIs_path)
    loc_overlapBetweenROIs_mean = np.nanmean(loc_overlapBetweenROIs[matrixIndices])
    if AllLayerOverlap:
        print(f"AllLayerOverlap: loc_overlapBetweenROIs_mean={loc_overlapBetweenROIs_mean}")
    else:
        print(f"NeighboringLayerOverlap: loc_overlapBetweenROIs_mean={loc_overlapBetweenROIs_mean}")
    showMatrix(loc_overlapBetweenROIs, title=f'localize-overlapBetweenROIs',
               axis0_ticks=ticksName,
               axis1_ticks=ticksName,
               fill_diagonal=True, cmap='Blues', fontsize=20 * 5, vmin=0, vmax=1,
               showTextFlag=False)

    resampleNumber = 5000
    bootstrap_mean = np.zeros(resampleNumber)
    for currIter in range(resampleNumber):
        if currIter % 1000 == 0:
            print(f"currIter = {currIter}", end=' ')
        _ = np.mean(nsd_means[np.random.choice(len(nsd_means), len(nsd_means),
                                               replace=True)])
        bootstrap_mean[currIter] = _
    normalizedValue = (loc_overlapBetweenROIs_mean - np.mean(bootstrap_mean)) / np.std(bootstrap_mean)
    print(f"\nnormalizedValue={normalizedValue}")

    def plotHistgram(bootstrap_mean, loc_overlapBetweenROIs_mean, AllLayerOverlap=None, fontsize=80):
        if AllLayerOverlap is None:
            raise Exception("AllLayerOverlap is None")
        fig, ax = plt.subplots(figsize=(1.5*fontsize, fontsize))
        histogram = ax.hist(bootstrap_mean, bins=100, color='grey')

        plt.vlines(x=loc_overlapBetweenROIs_mean, ymin=0, ymax=np.max(histogram[0]), color='red',
                   linewidth=fontsize/2)

        plt.xlim([0, 0.4])
        plt.ylim([0, 200])

        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
        ax.set_yticks([0, 40, 80, 120, 160, 200])

        percentile = np.mean(bootstrap_mean < loc_overlapBetweenROIs_mean)
        if AllLayerOverlap:
            plt.title(f"AllLayerOverlap: {100*np.round(percentile, 4)}%", fontsize=fontsize*3)
        else:
            plt.title(f"NeighboringLayerOverlap: {100*np.round(percentile, 4)}%", fontsize=fontsize*3)
        plt.xlabel(f"mean of overlap matrix", fontsize=fontsize*3)
        plt.ylabel(f"count", fontsize=fontsize*3)

        ax.tick_params(axis='both', which='both', labelsize=fontsize*3, size=fontsize/2)

        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        ax.xaxis.set_tick_params(width=0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if AllLayerOverlap:
            plt.savefig(f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/figures/fig5d_NSD_nonresidual.pdf",
                        transparent=True)
        else:
            plt.savefig(f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/figures/fig5e_NSD_nonresidual.pdf",
                        transparent=True)
        plt.show()
        if AllLayerOverlap:
            print(f"AllLayerOverlap: np.percentile(bootstrap_mean,0.05)={np.percentile(bootstrap_mean, 0.01)}")
        else:
            print(f"NeighboringLayerOverlap: np.percentile(bootstrap_mean,0.05)={np.percentile(bootstrap_mean, 0.01)}")

    plotHistgram(bootstrap_mean, loc_overlapBetweenROIs_mean, AllLayerOverlap=AllLayerOverlap)



nsd_overlapBetweenROIs_statistics(max_whichSets_1=15, AllLayerOverlap=True)
nsd_overlapBetweenROIs_statistics(max_whichSets_1=15, AllLayerOverlap=False)

def test_ROI_wholeBrainDifference():
    ROI = nib.load("/gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/"
                   "sl_RSA_findLayerInBrain/10/cor/rad3/repStabilityMasked__tfce_.nii.gz").get_fdata()
    slResult = None
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


def save_nib(affine_mat=None, toFill=None, saveToPath=''):
    sl_nii = nib.Nifti1Image(toFill.astype('double'), affine_mat)
    print(f'saving to {saveToPath}')
    nib.save(sl_nii, saveToPath)


def showMatrix(M, title='', axis0_ticks=None, axis1_ticks=None, fill_diagonal=False, cmap='seismic', fontsize=20 * 5,
               figsize=(9 * 10, 8 * 10), vmin=0, vmax=None, showNumber=True, showColorBar=True):
    if axis1_ticks is None:
        axis1_ticks = []
    if axis0_ticks is None:
        axis0_ticks = []
    import matplotlib.pyplot as plt

    if fill_diagonal:
        np.fill_diagonal(M, 0.0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    if len(title) > 0:
        plt.suptitle(f'{title}', fontsize=fontsize)
    if showNumber:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, round(M[i, j], 4),
                        ha="center", va="center", color="k", fontsize=fontsize)
    if showColorBar:
        fig.colorbar(im)
    xticks = np.arange(0, M.shape[1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(axis1_ticks, rotation=45, ha='right', fontsize=fontsize)
    yticks = np.arange(0, M.shape[0])
    ax.set_yticks(yticks)
    ax.set_yticklabels(axis0_ticks, fontsize=fontsize)
    plt.show()


max_whichSets_1 = 15
print(f"max_whichSets_1={max_whichSets_1}")

repeated3imgIDs = [3050, 3158, 3172, 3435, 3627, 3688, 3810, 3848, 3857, 4052, 4250, 4424, 4668, 4691, 4787, 4836, 4893,
                   4931, 5035, 5543, 5584, 5603, 6432, 6445, 6490, 6522, 7121, 7367, 7655, 7841, 7860, 8007, 8110, 8510,
                   8647, 8808, 9028, 9049, 9435, 9918, 10047, 10065, 10108, 10245, 10472, 11160, 11522, 11618, 11690,
                   11797, 11838, 11845, 11933, 11943, 11953, 12309, 12797, 12923, 13139, 13313, 13315, 14529, 14595,
                   14794, 14821, 14899, 15365, 15507, 15820, 15940, 16253, 16345, 16703, 16824, 16918, 17049, 17375,
                   17486, 17550, 19075, 19182, 19200, 19934, 20055, 20081, 20207, 20266, 20634, 20651, 20821, 21193,
                   21195, 21219, 21602, 21951, 22264, 22516, 22531, 22588, 22795, 22880, 22958, 22968, 22994, 23037,
                   23716, 23730, 24203, 24215, 24531, 25112, 25269, 25285, 25589, 25703, 25742, 26164, 26352, 26436,
                   26459, 26721, 26896, 26910, 26991, 27243, 27276, 27327, 27436, 27569, 27879, 28069, 28097, 28160,
                   28190, 28320, 28350, 28525, 28596, 28752, 29382, 29712, 29887, 30082, 30396, 30408, 30673, 30857,
                   31029, 31234, 31660, 31748, 31802, 31838, 31937, 31965, 32054, 32233, 32626, 32773, 32892, 32911,
                   33172, 33246, 33522, 33544, 34127, 34239, 34830, 34851, 35744, 35753, 35799, 35987, 36577, 36624,
                   36732, 36911, 36946, 36975, 36979, 37060, 37225, 37495, 38023, 38026, 38298, 38487, 38642, 38830,
                   38854, 38979, 39290, 39370, 39403, 40424, 40576, 40722, 40771, 40841, 40847, 40925, 40936, 41117,
                   41163, 41567, 41654, 41711, 41928, 42167, 42172, 42215, 42225, 42564, 42698, 42782, 42852, 42913,
                   43160, 43165, 43429, 43446, 43620, 43676, 43819, 43951, 44053, 44721, 44845, 44981, 45130, 45214,
                   45596, 45751, 45844, 45946, 46151, 46373, 46461, 46463, 47291, 47294, 47409, 47688, 48158, 48261,
                   48375, 48423, 48509, 48680, 49467, 49732, 49945, 49956, 50115, 50654, 50735, 50756, 50812, 51078,
                   51149, 51522, 51789, 51984, 52071, 52555, 52932, 53053, 53371, 53728, 53729, 53882, 54079, 54258,
                   54362, 55108, 55406, 55409, 55670, 55679, 55969, 56155, 56682, 56724, 56785, 56851, 56868, 56912,
                   57047, 57436, 57554, 57651, 57839, 59047, 59195, 59285, 59586, 59700, 59995, 60095, 60187, 60252,
                   60506, 60554, 61134, 61511, 61678, 61739, 61810, 62016, 62229, 62303, 62684, 64296, 64868, 64881,
                   65149, 65254, 65268, 65446, 65770, 65800, 65873, 65944, 66005, 66217, 66279, 66331, 66465, 66490,
                   66774, 66837, 66947, 67238, 67296, 67743, 67803, 67830, 68279, 68340, 68815, 68843, 68859, 69008,
                   69241, 69840, 69855, 70096, 70194, 70233, 70428, 71233, 71242, 71411, 71895, 72016, 72210, 72313,
                   72949]


def get_allSetsSum(max_whichSets_1=15):
    repStabilityMasked_ROI_path = ''
    for layer in tqdm(range(1, 13)):
        allSet_maskedROIx = np.zeros((1 + max_whichSets_1, 91, 109, 91))
        for whichSet_1 in range(0, 1 + max_whichSets_1):
            repStabilityMasked_ROI_path = f"{scratchResultFolder}/" \
                                          f"nsd/residualModel/randomise/" \
                                          f"layer{layer}/reg/rad{3}/whichSet{whichSet_1 + 1}/" \
                                          f"repStabilityMasked_ROI_layer{layer}_whichSet{whichSet_1 + 1}.nii.gz"
            repStabilityMasked_ROI = nib.load(repStabilityMasked_ROI_path).get_fdata()
            allSet_maskedROIx[whichSet_1] = repStabilityMasked_ROI
        allSet_maskedROIx = np.nansum(allSet_maskedROIx, axis=0)
        assert allSet_maskedROIx.shape == (91, 109, 91)
        affine_mat = nib.load(repStabilityMasked_ROI_path).affine
        os.makedirs(f"{scratchResultFolder}/nsd/residualModel/allSetsSum/", exist_ok=True)
        save_nib(toFill=allSet_maskedROIx, affine_mat=affine_mat,
                 saveToPath=f"{scratchResultFolder}/nsd/residualModel/allSetsSum/"
                            f"/allSetsSum_layer{layer}.nii.gz")

    print(
        f"fslview_deprecated /gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz ",
        end='')
    for layer in range(12, 0, -1):
        print(f"{scratchResultFolder}/nsd/residualModel/allSetsSum/"
              f"allSetsSum_layer{layer}.nii.gz", end=' ')


get_allSetsSum(max_whichSets_1=max_whichSets_1)


def get_overlapBetweenROIs(_max_whichSets_1=15):
    def get_overlapBetweenROIs_forSet(whichSet=None):
        maskedROIx = {}
        for _layer in range(1, 13):
            _repStabilityMasked_ROI_path = f"{scratchResultFolder}/" \
                                           f"nsd/residualModel/randomise/" \
                                           f"layer{_layer}/reg/rad{3}/whichSet{whichSet}/" \
                                           f"repStabilityMasked_ROI_layer{_layer}_whichSet{whichSet}.nii.gz"
            repStabilityMasked_ROIx = nib.load(_repStabilityMasked_ROI_path).get_fdata()
            maskedROIx[f"layer{_layer}"] = repStabilityMasked_ROIx
        overlapBtweenROIs = np.zeros((12, 12))

        for layer_i in range(1, 13):
            index_i = f"layer{layer_i}"
            for layer_j in range(1, 13):
                index_j = f"layer{layer_j}"
                overlapBtweenROIs[layer_i - 1, layer_j - 1] = np.nansum(
                    maskedROIx[index_i] * maskedROIx[index_j]) / np.nansum(maskedROIx[index_i])
        return overlapBtweenROIs

    overlapBetweenROIs_allSets = np.zeros((1 + _max_whichSets_1, 12, 12))
    for whichSet_1 in tqdm(range(0, 1 + _max_whichSets_1)):
        overlapBetweenROIs = get_overlapBetweenROIs_forSet(whichSet=whichSet_1 + 1)
        overlapBetweenROIs_allSets[whichSet_1] = overlapBetweenROIs
    filename =f"{scratchResultFolder}/nsd/residualModel/allSetsSum/overlapBetweenROIs_allSets.npy"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, overlapBetweenROIs_allSets)
    return overlapBetweenROIs_allSets


_overlapBetweenROIs_allSets = get_overlapBetweenROIs(_max_whichSets_1=15)


def compareEachSetWithLocalize_NSD_residualModel(method='reg', randomiseMethod='tfce', negativeFlag='_', radius=3):
    overlapDivideNSD = np.zeros((12, 16))
    overlapDivideLOC = np.zeros((12, 16))
    for layer in range(1, 13):
        for whichSet in range(1, 17):
            repStabilityMasked_ROI_nsd = nib.load(f"{scratchResultFolder}/"
                                                  f"nsd/residualModel/randomise/"
                                                  f"layer{layer}/reg/rad{3}/whichSet{whichSet}/"
                                                  f"repStabilityMasked_ROI_layer{layer}_whichSet{whichSet}.nii.gz").get_fdata()
            repStabilityMasked_ROI_loc = nib.load(f"{scratchResultFolder}/"
                                                  f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/"
                                                  f"layer{layer}/{method}/rad{radius}/"
                                                  f"repStabilityMasked__{randomiseMethod}{negativeFlag}_layer{layer}.nii.gz").get_fdata()
            overlapDivideNSD[layer - 1, whichSet - 1] = np.sum(
                repStabilityMasked_ROI_nsd * repStabilityMasked_ROI_loc) / np.sum(repStabilityMasked_ROI_nsd)
            overlapDivideLOC[layer - 1, whichSet - 1] = np.sum(
                repStabilityMasked_ROI_nsd * repStabilityMasked_ROI_loc) / np.sum(repStabilityMasked_ROI_loc)
    showMatrix(overlapDivideNSD, title="overlapDivideNSD",
               axis0_ticks=[f"layer{layer}" for layer in range(1, 13)],
               axis1_ticks=[f"set{whichSet}" for whichSet in range(1, 17)],
               fill_diagonal=False)
    showMatrix(overlapDivideLOC, title="overlapDivideLOC",
               axis0_ticks=[f"layer{layer}" for layer in range(1, 13)],
               axis1_ticks=[f"set{whichSet}" for whichSet in range(1, 17)],
               fill_diagonal=False)


compareEachSetWithLocalize_NSD_residualModel()




def nsd_overlapBetweenROIs_statistics(max_whichSets_1=15,
                                      AllLayerOverlap=True):

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

        ax.tick_params(axis='both', which='both', labelsize=fontsize, size=fontsize / 3)

        if savePath is not None:
            plt.savefig(savePath, transparent=True)
        plt.show()

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

    overlapBetweenROIs_allSets = np.load(f"{scratchResultFolder}/"
                                         f"nsd/residualModel/allSetsSum/overlapBetweenROIs_allSets.npy")
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
                       title=f'nsd-tfce-residualModel-whichSet{whichSet_1 + 1}',
                       axis0_ticks=ticksName,
                       axis1_ticks=ticksName,
                       fill_diagonal=True, cmap='Blues', fontsize=20 * 2, vmin=0, vmax=1)

    showMatrix(np.nanmean(nsd_overlapBetweenROIs, axis=0), title=f'nsd-tfce-residualModel-meanAcrossSets',
               axis0_ticks=ticksName,
               axis1_ticks=ticksName,
               fill_diagonal=True, cmap='Blues', fontsize=20 * 5, showTextFlag=False, vmin=0, vmax=1,
               savePath=f"{scratchResultFolder}/figures/fig6c_NSD_residual.pdf")
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

    loc_overlapBetweenROIs_path = f"{scratchResultFolder}/findLayerInBrain_leaveOneSubOutReg/overlapBetweenROIs.npy"
    loc_overlapBetweenROIs = np.load(loc_overlapBetweenROIs_path)
    loc_overlapBetweenROIs_mean = np.nanmean(loc_overlapBetweenROIs[matrixIndices])
    if AllLayerOverlap:
        print(f"AllLayerOverlap: loc_overlapBetweenROIs_mean={loc_overlapBetweenROIs_mean}")
    else:
        print(f"NeighboringLayerOverlap: loc_overlapBetweenROIs_mean={loc_overlapBetweenROIs_mean}")
    showMatrix(loc_overlapBetweenROIs, title=f'localize-overlapBetweenROIs',
               axis0_ticks=ticksName,
               axis1_ticks=ticksName,
               fill_diagonal=True, cmap='Blues', fontsize=20 * 2, vmin=0, vmax=1)

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
            plt.savefig(f"{scratchResultFolder}/figures/fig6d_NSD_residual.pdf",
                        transparent=True)
        else:
            plt.savefig(f"{scratchResultFolder}/figures/fig6e_NSD_residual.pdf",
                        transparent=True)
        plt.show()
        if AllLayerOverlap:
            print(f"AllLayerOverlap: np.percentile(bootstrap_mean,0.05)={np.percentile(bootstrap_mean, 0.01)}")
        else:
            print(f"NeighboringLayerOverlap: np.percentile(bootstrap_mean,0.05)={np.percentile(bootstrap_mean, 0.01)}")

    plotHistgram(bootstrap_mean, loc_overlapBetweenROIs_mean, AllLayerOverlap=AllLayerOverlap)


nsd_overlapBetweenROIs_statistics(max_whichSets_1=15, AllLayerOverlap=True)
nsd_overlapBetweenROIs_statistics(max_whichSets_1=15, AllLayerOverlap=False)
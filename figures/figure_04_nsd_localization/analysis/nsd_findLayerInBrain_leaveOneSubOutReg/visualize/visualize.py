import os

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")
import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import urllib.request
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
from pprint import pprint
from os.path import join, exists, split
import sys

print(sys.executable)
print(sys.version)
print(sys.version_info)

sys.path.append('/gpfs/milgram/project/turk-browne/projects/localize/analysis/GLMsingle')
from glmsingle.glmsingle import GLM_single
import pandas as pd
import argparse
import random
from glob import glob
import subprocess
import nibabel as nib
from tqdm import tqdm
import pickle5 as pickle
import time

print(f"numpy version = {np.__version__}")
print(f"pandas version = {pd.__version__}")

testMode = False





def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def getjobID_num(sbatch_response):
    import re
    jobID = re.findall(r'\d+', sbatch_response)[0]
    return jobID


def kp_run(cmd):
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd)
    check(sbatch_response)
    return sbatch_response


def kp_remove(fileName):
    cmd = f"rm {fileName}"
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd)
    print(sbatch_response)


def wait(tmpFile, waitFor=0.1):
    while not os.path.exists(tmpFile):
        time.sleep(waitFor)
    return 1


def check(sbatch_response):
    print(sbatch_response)
    if "Exception" in sbatch_response or "Error" in sbatch_response or "Failed" in sbatch_response or "not" in sbatch_response or 'Unrecognised' in sbatch_response:
        raise Exception(sbatch_response)


def checkEndwithDone(filename):
    with open(filename, 'r') as f:
        last_line = f.readlines()[-1]
    return last_line == "done\n"


def checkDone(jobIDs):
    completed = {}
    for jobID in jobIDs:
        filename = f"./logs/{jobID}.out"
        completed[jobID] = checkEndwithDone(filename)
    if np.mean(list(completed.values())) == 1:
        status = True
    else:
        status = False
    return completed, status


def check_jobIDs(jobIDs):
    completed, status = checkDone(jobIDs)
    if status == True:
        pass
    else:
        print(completed)
        assert status == True
    return completed


def check_jobArray(jobID='', jobarrayNumber=10):
    arrayIDrange = np.arange(1, 1 + jobarrayNumber)
    jobIDs = []
    for arrayID in arrayIDrange:
        jobIDs.append(f"{jobID}_{arrayID}")
    completed = check_jobIDs(jobIDs)
    return completed


def waitForEnd(jobID):
    while jobID_running_myjobs(jobID):
        print(f"waiting for {jobID} to end")
        time.sleep(5)
    print(f"{jobID} finished")


def jobID_running_myjobs(jobID):
    jobID = str(jobID)
    cmd = "squeue -u kp578"
    sbatch_response = subprocess.getoutput(cmd)
    if jobID in sbatch_response:
        return True
    else:
        return False


def showMatrix(M, title='', axis0_ticks=None, axis1_ticks=None, fill_diagonal=True, cmap='seismic', fontsize=20 * 5):
    if axis1_ticks is None:
        axis1_ticks = []
    if axis0_ticks is None:
        axis0_ticks = []
    import matplotlib.pyplot as plt

    if fill_diagonal:
        np.fill_diagonal(M, 0.0)
    fig, ax = plt.subplots(1, 1, figsize=(9 * 10, 8 * 10))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    plt.suptitle(f'{title}', fontsize=fontsize)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if i == j:
                pass
            else:
                ax.text(j, i, round(M[i, j], 4),
                        ha="center", va="center", color="k", fontsize=fontsize)
    fig.colorbar(im)
    xticks = np.arange(0, M.shape[1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(axis1_ticks, rotation=45, ha='right', fontsize=fontsize)
    yticks = np.arange(0, M.shape[0])
    ax.set_yticks(yticks)
    ax.set_yticklabels(axis0_ticks, fontsize=fontsize)
    plt.show()


os.chdir("/gpfs/milgram/project/turk-browne/projects/localize/analysis/")


def getSubs():
    subFolder = "/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/"
    subs = glob(f"{subFolder}/sub*")
    subs.sort()
    subjects = []
    ExcludedSub = ['sub006', 'sub018',
                   'sub026']
    for sub in subs:
        subject = sub.split("/")[-1]
        if subject not in ExcludedSub:
            subjects.append(subject)
    return subjects


subs = getSubs()




def visualize_sl_RSA_findLayerInBrain(
        randomiseMethod="tstat4.275"):
    def save_nib(affine_mat=None, toFill=None, saveToPath=''):
        sl_nii = nib.Nifti1Image(toFill.astype('double'), affine_mat)
        print(f'saving to {saveToPath}')
        nib.save(sl_nii, saveToPath)

    overlapScore = pd.DataFrame(
        columns=['ROIx_layer', 'overlap/ROIx', 'overlap/repStabilityMask',
                 'overlap_size', 'repStabilityMask_size', 'ROIx_size',
                 'radius', 'method'])
    sign, negativeFlag = 'positive', '_'
    for method in ['reg']:
        for radius in [3]:
            for layer in range(1, 13):
                if 'tstat' in randomiseMethod:
                    print(f"randomiseMethod={randomiseMethod}")
                    TThreshold = float(randomiseMethod.split('tstat')[-1])
                    repStabiility_RSA_sl_mask = f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/results/" \
                                                f"repStability_RSA_sl/rad{radius}/__tstat1.nii.gz"
                    stability_mask = nib.load(repStabiility_RSA_sl_mask).get_fdata()
                    stability_mask = stability_mask > TThreshold
                else:
                    repStabiility_RSA_sl_mask = f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/results/" \
                                                f"repStability_RSA_sl/rad{radius}/__tfce_corrp_tstat1.nii.gz"
                    stability_mask = nib.load(repStabiility_RSA_sl_mask).get_fdata()
                    stability_mask = stability_mask > 0.95

                if 'tstat' in randomiseMethod:
                    print(f"randomiseMethod={randomiseMethod}")
                    TThreshold = float(randomiseMethod.split('tstat')[-1])
                    sl_RSA_findLayerInBrain_mask = f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/" \
                                                   f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/" \
                                                   f"layer{layer}/{method}/rad{radius}/" \
                                                   f"{negativeFlag}_tstat1.nii.gz"
                    ROIx_mask = nib.load(sl_RSA_findLayerInBrain_mask).get_fdata()
                    ROIx_mask = ROIx_mask > TThreshold

                else:
                    sl_RSA_findLayerInBrain_mask = f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/" \
                                                   f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/" \
                                                   f"layer{layer}/{method}/rad{radius}/" \
                                                   f"{negativeFlag}_tfce_corrp_tstat1.nii.gz"
                    ROIx_mask = nib.load(sl_RSA_findLayerInBrain_mask).get_fdata()
                    ROIx_mask = ROIx_mask > 0.95

                assert stability_mask.shape == ROIx_mask.shape

                repStabilityMasked_ROIx = stability_mask * ROIx_mask
                overlap_repStabilityMask = np.sum(repStabilityMasked_ROIx) / np.sum(stability_mask)
                overlap_ROIx = np.sum(repStabilityMasked_ROIx) / np.sum(ROIx_mask)
                overlapScore = overlapScore.append({'ROIx_layer': layer,
                                                    'sign': sign,
                                                    'negativeFlag': negativeFlag,
                                                    'overlap/ROIx': overlap_ROIx,
                                                    'overlap/repStabilityMask': overlap_repStabilityMask,

                                                    'overlap_size': np.sum(repStabilityMasked_ROIx),
                                                    'repStabilityMask_size': np.sum(stability_mask),
                                                    'ROIx_size': np.sum(ROIx_mask),

                                                    'radius': radius,
                                                    'method': method
                                                    },
                                                   ignore_index=True)

                affine_mat = nib.load(sl_RSA_findLayerInBrain_mask).affine
                save_nib(toFill=repStabilityMasked_ROIx, affine_mat=affine_mat,
                         saveToPath=f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/"
                                    f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/"
                                    f"layer{layer}/{method}/rad{radius}/"
                                    f"repStabilityMasked__{randomiseMethod}{negativeFlag}_layer{layer}.nii.gz")
    return overlapScore


_overlapScore = visualize_sl_RSA_findLayerInBrain(randomiseMethod="tfce")


def visualize_sl_RSA_findLayerInBrain(
        randomiseMethod="tfce", radius=3,
        method='reg'):

    maskedROIx = {}
    sign, negativeFlag = 'positive', '_'
    for layer in range(1, 13):
        repStabilityMasked_ROIx = nib.load(f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/"
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


_overlapBtweenROIs = visualize_sl_RSA_findLayerInBrain(randomiseMethod="tfce", radius=3, method='reg')

ticksName = [f"ROI-{layer}" for layer in range(1, 13)]
showMatrix(_overlapBtweenROIs, title='tfce',
           axis0_ticks=ticksName,
           axis1_ticks=ticksName,
           fill_diagonal=True, cmap='Blues', fontsize=20 * 2)


def calculateSignificantVoxelNumber():
    randomiseMethod = 'tfce'
    method = 'reg'
    radius = 3
    sign, negativeFlag = 'positive', '_'
    for layer in range(1, 13):
        voxels = nib.load(f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/"
                          f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/"
                          f"layer{layer}/{method}/rad{radius}/"
                          f"repStabilityMasked__{randomiseMethod}{negativeFlag}_layer{layer}.nii.gz").get_fdata()
        voxels = voxels > 0.95
        print(f"layer {layer} number of voxel > 0.95 = {np.sum(voxels)}")


calculateSignificantVoxelNumber()


def calculateSignificantVoxelNumber_notRepStabilityMask():
    randomiseMethod = 'tfce'
    method = 'reg'
    radius = 3
    sign, negativeFlag = 'positive', '_'
    for layer in range(1, 13):
        voxels = nib.load(f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/"
                          f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/"
                          f"layer{layer}/{method}/rad{radius}/"
                          f"{negativeFlag}_tfce_corrp_tstat1.nii.gz").get_fdata()
        voxels = voxels > 0.95
        print(f"layer {layer} number of voxel > 0.95 = {np.sum(voxels)}")


calculateSignificantVoxelNumber_notRepStabilityMask()


def fslview_deprecated(randomiseMethod="tstat4.275"):
    print(
        f"fslview_deprecated /gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz",
        end=' ')
    for layer in range(12, 0, -1):
        print(f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/findLayerInBrain_leaveOneSubOutReg/"
              f"shuffleNorm/randomise/layer{layer}/reg/rad3/repStabilityMasked__tfce__layer{layer}.nii.gz",
              end=' ')

    """
        cd /gpfs/milgram/project/turk-browne/projects/localize/analysis/temp/temp/

        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/1/cor/rad3/repStabilityMasked__tfce_squareCorrelation_.nii.gz 1.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/2/cor/rad3/repStabilityMasked__tfce_squareCorrelation_.nii.gz 2.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/3/cor/rad3/repStabilityMasked__tfce_squareCorrelation_.nii.gz 3.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/4/cor/rad3/repStabilityMasked__tfce_squareCorrelation_.nii.gz 4.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/5/cor/rad3/repStabilityMasked__tfce_squareCorrelation_.nii.gz 5.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/6/cor/rad3/repStabilityMasked__tfce_squareCorrelation_.nii.gz 6.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/7/cor/rad3/repStabilityMasked__tfce_squareCorrelation_.nii.gz 7.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/8/cor/rad3/repStabilityMasked__tfce_squareCorrelation_.nii.gz 8.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/9/cor/rad3/repStabilityMasked__tfce_squareCorrelation_.nii.gz 9.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/10/cor/rad3/repStabilityMasked__tfce_squareCorrelation_.nii.gz 10.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/11/cor/rad3/repStabilityMasked__tfce_squareCorrelation_.nii.gz 11.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/12/cor/rad3/repStabilityMasked__tfce_squareCorrelation_.nii.gz 12.nii.gz

        cp /gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz stand.nii.gz


        fslview_deprecated stand.nii.gz 12.nii.gz 11.nii.gz 10.nii.gz 9.nii.gz 8.nii.gz 7.nii.gz 6.nii.gz 5.nii.gz 4.nii.gz 3.nii.gz 2.nii.gz 1.nii.gz


        cd /gpfs/milgram/project/turk-browne/projects/localize/analysis/temp/negativeCorrelation/tfce/
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/1/cor/rad3/_squareCorrelation__tfce_corrp_tstat1.nii.gz  1.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/2/cor/rad3/_squareCorrelation__tfce_corrp_tstat1.nii.gz  2.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/3/cor/rad3/_squareCorrelation__tfce_corrp_tstat1.nii.gz  3.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/4/cor/rad3/_squareCorrelation__tfce_corrp_tstat1.nii.gz  4.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/5/cor/rad3/_squareCorrelation__tfce_corrp_tstat1.nii.gz  5.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/6/cor/rad3/_squareCorrelation__tfce_corrp_tstat1.nii.gz  6.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/7/cor/rad3/_squareCorrelation__tfce_corrp_tstat1.nii.gz  7.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/8/cor/rad3/_squareCorrelation__tfce_corrp_tstat1.nii.gz  8.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/9/cor/rad3/_squareCorrelation__tfce_corrp_tstat1.nii.gz  9.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/10/cor/rad3/_squareCorrelation__tfce_corrp_tstat1.nii.gz  10.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/11/cor/rad3/_squareCorrelation__tfce_corrp_tstat1.nii.gz  11.nii.gz
        cp /gpfs/milgram/project/turk-browne/projects/localize/analysis/results/randomise/sl_RSA_findLayerInBrain/12/cor/rad3/_squareCorrelation__tfce_corrp_tstat1.nii.gz  12.nii.gz

        cp /gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz stand.nii.gz

        fslview_deprecated stand.nii.gz 12.nii.gz 11.nii.gz 10.nii.gz 9.nii.gz 8.nii.gz 7.nii.gz 6.nii.gz 5.nii.gz 4.nii.gz 3.nii.gz 2.nii.gz 1.nii.gz
    """


print("done")
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
    np.save(f"nsd_residualModel/clustermThreshold/randomise_findLayerInBrain_allSets_jobArray.npy",
            jobarrayDict)
    if testMode:
        cmd = (f"sbatch --requeue --array=1-1 "
               f"nsd_residualModel/clustermThreshold/randomise_findLayerInBrain_allSets.sh 1")
    else:
        cmd = (f"sbatch --requeue --array=1-{len(jobarrayDict)} "
               f"nsd_residualModel/clustermThreshold/randomise_findLayerInBrain_allSets.sh 0")
    sbatch_response = custom_run(cmd)
    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    if testMode:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
    else:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))


randomise_findLayerInBrain_allSets(testMode=testMode)


def showNilearn(thresholdName=None):
    import nilearn
    import nilearn.datasets
    from nilearn import datasets, plotting
    import nilearn.plotting
    from nilearn import surface
    sys.path.append("display_surface_plot")
    from display_surface_plot import designCmap

    def plot_surface_saveImageReload(resultType=None, views=None, h_pad_=-70, w_pad_=-2, cmap=None, colorBar=False, thresholdName=None,
                                     forceRun=False):
        if cmap is None:
            cmap = 'bwr'

        def remove_whiteBorder(array):
            array_grey = np.mean(array, axis=2)
            b1 = np.where(np.mean(array_grey, axis=0) != 255)[0][0]
            b2 = np.where(np.mean(array_grey, axis=0) != 255)[0][-1]
            b3 = np.where(np.mean(array_grey, axis=1) != 255)[0][0]
            b4 = np.where(np.mean(array_grey, axis=1) != 255)[0][-1]

            return array[b3:b4, b1:b2, :]

        def cleanFolder(folder_path):
            if not os.listdir(folder_path):
                print('Folder is empty')
            else:
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')

        temp_arrangementFolder = f"/gpfs/milgram/project/turk-browne/projects/localize/temp/temp_arrangement/"
        os.makedirs(temp_arrangementFolder, exist_ok=True)
        fsaverage = datasets.fetch_surf_fsaverage()
        hemispheres = ['left', 'right']
        layers = np.arange(1, 13)
        for layer in layers:
            niiData = None
            nii = None
            if resultType == f"nsd-residual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat":

                radius = 3
                method = 'reg'
                clustermPath = f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/nsd/" \
                               f"residual/testClustermThreshold/randomise/" \
                               f"rad{radius}/{method}/layer{layer}/threshold_{thresholdName}/"
                clusterm = nib.load(f"{clustermPath}/"
                                    f"layer{layer}_threshold_{thresholdName}__clusterm_corrp_tstat1.nii.gz").get_fdata()
                clusterm[clusterm < 0.95] = 0
                clusterm[clusterm >= 0.95] = 1

                repStabilityMask = nib.load(
                    f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/nsd/repStability_RSA_sl_clusterm/"
                    f"nonresidual/rad{radius}/thresholdName{thresholdName}/"
                    f"thresholdName{thresholdName}__clusterm_corrp_tstat1.nii.gz").get_fdata()
                repStabilityMask[repStabilityMask < 0.95] = 0
                repStabilityMask[repStabilityMask >= 0.95] = 1

                tstatFile = nib.load(f"{clustermPath}/layer{layer}_threshold_{thresholdName}__tstat1.nii.gz")
                tstat = tstatFile.get_fdata()

                affine = nib.load("/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz").affine

                niiData = clusterm * repStabilityMask * tstat
                nii = nib.Nifti1Image(niiData.astype('double'), affine)
            else:
                raise Exception

            leftHemi = niiData[46:-1, :, :]
            leftHemiNumber = np.sum(leftHemi)
            rightHemi = niiData[0:46, :, :]
            rightHemiNumber = np.sum(rightHemi)

            nonZeroVoxelNumber = np.sum(niiData != 0)
            maxNiiValue = np.max(niiData)

            threshold = None
            darkness = None
            for hemi in hemispheres:
                if hemi == 'left':
                    texture = surface.vol_to_surf(nii, fsaverage.pial_left,
                                                  interpolation='nearest')
                    bg_map = fsaverage.sulc_left
                    surf = fsaverage.infl_left
                else:
                    texture = surface.vol_to_surf(nii, fsaverage.pial_right,
                                                  interpolation='nearest')
                    bg_map = fsaverage.sulc_right
                    surf = fsaverage.infl_right
                for view in views:
                    imgFPath = f"{temp_arrangementFolder}/resultType-{resultType}-layer-{layer}-{view}-{hemi}-hemi.png"
                    if (not os.path.exists(imgFPath)) or forceRun:
                        if (leftHemiNumber == 0 and hemi == 'left') or (rightHemiNumber == 0 and hemi == 'right'):
                            def plot_empty_brain_(hemi=None):
                                from nilearn import datasets
                                from nilearn import plotting
                                from nilearn import surface
                                import numpy as np
                                import nibabel as nib
                                fsaverage = datasets.fetch_surf_fsaverage()
                                nii = nib.load(
                                    "/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz")
                                niiData = nii.get_fdata()
                                nii = nib.Nifti1Image(niiData, nii.affine)
                                leftHemi = niiData[0:46, :, :]
                                rightHemi = niiData[46:-1, :, :]

                                maxNiiValue = np.max(niiData)
                                hemispheres = ['left', 'right']

                                if hemi == 'left':
                                    texture = surface.vol_to_surf(nii, fsaverage.pial_left, interpolation='nearest')
                                    bg_map = fsaverage.sulc_left
                                    surf = fsaverage.infl_left
                                else:
                                    texture = surface.vol_to_surf(nii, fsaverage.pial_right,
                                                                  interpolation='nearest')
                                    bg_map = fsaverage.sulc_right
                                    surf = fsaverage.infl_right
                                for view in views:
                                    surf_map = plotting.plot_surf_stat_map(surf, texture, cmap=cmap,
                                                                           hemi=hemi, view=view, bg_on_data=False,
                                                                           bg_map=bg_map,
                                                                           bg_img='MNI152',
                                                                           black_bg=True,
                                                                           colorbar=colorBar,
                                                                           threshold=1000000,
                                                                           darkness=0.3,
                                                                           )

                            plot_empty_brain_(hemi=hemi)
                        else:
                            surf_map = plotting.plot_surf_stat_map(surf, texture, cmap=cmap,
                                                                   hemi=hemi, view=view, bg_on_data=True,
                                                                   bg_map=bg_map,
                                                                   bg_img='MNI152',
                                                                   colorbar=colorBar,
                                                                   black_bg=False,
                                                                   threshold=threshold,
                                                                   darkness=darkness,
                                                                   )
                        plt.savefig(imgFPath)
                        plt.close()
                        print(f"resultType-{resultType}: layer-{layer} {view} {hemi}-hemi "
                              f"voxelNumber-left{leftHemiNumber}-right{rightHemiNumber} "
                              f"maxValue-{maxNiiValue}")

        from PIL import Image
        arrangement = "3-8"
        if arrangement == "6-4":
            fig, axes = plt.subplots(nrows=6, ncols=4, subplot_kw={'projection': '3d'}, figsize=(20, 20))
            w_pad = -2
            h_pad = -4
        elif arrangement == "3-8":
            fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(40, 40))
            if views[0] == 'posterior':
                w_pad = w_pad_
                h_pad = h_pad_
            else:
                w_pad = -3
                h_pad = -200
        else:
            raise Exception
        plt.tight_layout(pad=0, w_pad=w_pad, h_pad=h_pad, rect=(0, 0, 1, 1))

        for layer in range(1, 13):
            for hemi in hemispheres:
                if arrangement == "6-4":
                    if layer - 1 > 5.5:
                        figureAxis_x = layer - 1 - 6
                        if hemi == 'left':
                            figureAxis_y = 2
                        else:
                            figureAxis_y = 3
                    else:
                        figureAxis_x = layer - 1
                        if hemi == 'left':
                            figureAxis_y = 0
                        else:
                            figureAxis_y = 1
                elif arrangement == "3-8":
                    def testArangement():
                        for layer in range(1, 13):
                            figureAxis_x = np.floor((layer - 1) / 4)
                            for hemi in range(2):
                                figureAxis_y = int(2 * (layer - 1 - np.floor((layer - 1) / 4) * 4) + hemi)
                                print(figureAxis_x, figureAxis_y)

                    figureAxis_x = int(np.floor((layer - 1) / 4))
                    if hemi == 'left':
                        figureAxis_y = int(2 * (layer - 1 - np.floor((layer - 1) / 4) * 4) + 0)
                    else:
                        figureAxis_y = int(2 * (layer - 1 - np.floor((layer - 1) / 4) * 4) + 1)
                else:
                    raise Exception
                for view in views:
                    imgFPath = f"{temp_arrangementFolder}/resultType-{resultType}-layer-{layer}-{view}-{hemi}-hemi.png"
                    array = np.array(Image.open(imgFPath))
                    image_noBorder = remove_whiteBorder(array)

                    axes[figureAxis_x, figureAxis_y].imshow(image_noBorder)
                    axes[figureAxis_x, figureAxis_y].set_title(f'layer-{layer}', fontsize=28)
                    axes[figureAxis_x, figureAxis_y].set_xticks([])
                    axes[figureAxis_x, figureAxis_y].set_yticks([])
                    axes[figureAxis_x, figureAxis_y].axis('off')
        _ = plt.show()
        print(f"resultType={resultType}, views={views}")

    _newCmap = designCmap(cmap='gist_rainbow')
    colorBar = False
    forceRun = True
    plot_surface_saveImageReload(
        resultType=f"nsd-residual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat",
        thresholdName=thresholdName,
        views=['posterior'],
        cmap=_newCmap, colorBar=colorBar, forceRun=forceRun)

    def plotMaximalLayer(resultType=None, views=None, cmap=None, colorBar=True, thresholdName=None):
        from nilearn import datasets
        from nilearn import plotting
        from nilearn import surface
        import numpy as np
        import nibabel as nib
        fsaverage = datasets.fetch_surf_fsaverage()
        if cmap is None:
            cmap = 'bwr'
        nii = None
        niiData = None
        if resultType == f"nsd-residual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat":
            tstat_allLayer = np.zeros((12, 91, 109, 91))
            affine = None
            for layer in range(1, 13):
                radius = 3
                method = 'reg'
                clustermPath = f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/nsd/" \
                               f"residual/testClustermThreshold/randomise/" \
                               f"rad{radius}/{method}/layer{layer}/threshold_{thresholdName}/"
                clusterm = nib.load(f"{clustermPath}/"
                                    f"layer{layer}_threshold_{thresholdName}__clusterm_corrp_tstat1.nii.gz").get_fdata()
                clusterm[clusterm < 0.95] = 0
                clusterm[clusterm >= 0.95] = 1

                repStabilityMask = nib.load(
                    f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/nsd/repStability_RSA_sl_clusterm/"
                    f"nonresidual/rad{radius}/thresholdName{thresholdName}/"
                    f"thresholdName{thresholdName}__clusterm_corrp_tstat1.nii.gz").get_fdata()
                repStabilityMask[repStabilityMask < 0.95] = 0
                repStabilityMask[repStabilityMask >= 0.95] = 1

                tstatFile = nib.load(f"{clustermPath}/layer{layer}_threshold_{thresholdName}__tstat1.nii.gz")

                tstat = tstatFile.get_fdata()
                affine = nib.load("/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz").affine
                niiData = clusterm * repStabilityMask * tstat
                tstat_allLayer[layer - 1, :, :, :] = niiData
            maxValue = np.nanmax(tstat_allLayer,
                                 axis=0)
            tstat_maxLayerID = np.argmax(tstat_allLayer, axis=0) + 1
            tstat_maxLayerID[maxValue == 0] = 0
            print(f"tstat_maxLayerID.shape={tstat_maxLayerID.shape}")
            print(f"np.unique(tstat_maxLayerID)={np.unique(tstat_maxLayerID)}")
            nii = nib.Nifti1Image(tstat_maxLayerID.astype('double'), affine)
            niiData = tstat_maxLayerID.astype('double')
        else:
            raise Exception
        leftHemi = niiData[46:-1, :, :]
        leftHemiNumber = np.sum(leftHemi)
        rightHemi = niiData[0:46, :, :]
        rightHemiNumber = np.sum(rightHemi)

        nonZeroVoxelNumber = np.sum(niiData != 0)
        maxNiiValue = np.max(niiData)
        hemispheres = ['left', 'right']
        threshold = None
        darkness = None
        for view in views:
            for hemi in hemispheres:
                if hemi == 'left':
                    texture = surface.vol_to_surf(nii, fsaverage.pial_left,
                                                  interpolation='nearest')
                    bg_map = fsaverage.sulc_left
                    surf = fsaverage.infl_left
                else:
                    texture = surface.vol_to_surf(nii, fsaverage.pial_right,
                                                  interpolation='nearest')
                    bg_map = fsaverage.sulc_right
                    surf = fsaverage.infl_right
                surf_map = plotting.plot_surf_stat_map(surf, texture, cmap=cmap,
                                                       hemi=hemi, view=view, bg_on_data=True, bg_map=bg_map,
                                                       bg_img='MNI152',
                                                       colorbar=colorBar,
                                                       black_bg=False,
                                                       threshold=threshold,
                                                       darkness=darkness,
                                                       )
                print(f"resultType-{resultType}: {view} {hemi}-hemi "
                      f"voxelNumber-left{leftHemiNumber}-right{rightHemiNumber} "
                      f"maxValue-{maxNiiValue}")
                plotting.show()

    _newCmap = designCmap(cmap='gist_rainbow')
    plotMaximalLayer(resultType=f"nsd-residual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat",
                     views=['lateral', 'medial', 'posterior'],
                     thresholdName=thresholdName,
                     cmap=_newCmap)


showNilearn(thresholdName=0.001)


def Sfig6c_v1():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap
    continuousCmap = designCmap(cmap='inferno', colors=1200)
    colorBar = True
    forceRun = True
    thresholdName=0.001
    valueRange = plot_surface_saveImageReload_pdf(
        resultType=f"nsd-residual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat",
        views=['posterior'], cmap=continuousCmap, colorBar=colorBar,
        vmax=21,
        thresholdName=thresholdName,
        forceRun=forceRun
    )


Sfig6c_v1()


def Sfig6c_v2():
    sys.path.append("display_surface_plot")
    from display_surface_plot import plotMaximalLayer_pdf, designCmap
    continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)
    colorBar = True
    thresholdName = 0.001
    plotMaximalLayer_pdf(resultType=f"nsd-residual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat",
                         views=['lateral', 'medial', 'posterior'],
                         cmap=continuous_Cmap,
                         thresholdName=thresholdName)

Sfig6c_v2()

print("done")
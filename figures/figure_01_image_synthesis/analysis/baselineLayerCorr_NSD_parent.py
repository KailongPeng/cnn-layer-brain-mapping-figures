import os

current_dir = os.getcwd()

assert current_dir.endswith('localize_paper'), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")

projectDir = current_dir

import sys
sys.path.append(projectDir)
import pdb

import imageio
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pickle5 as pickle
import subprocess
import re

print(f"CONDA_DEFAULT_ENV={os.environ['CONDA_DEFAULT_ENV']}")

sys.path.append(current_dir)
from utils import load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray, setupPaths, \
    check_jobIDs
from utils import custom_and

[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

from utils import cfg_loading

os.chdir(projectDir)
workingDir = projectDir
iterationNumber = 5000
from utils import check_jobArray
from utils import getjobID_num
from utils import waitForEnd


def showMatrix(M, title='', axis0_ticks=[], axis1_ticks=[], fill_diagonal=False, cmap=None, vmin=-1,
               vmax=1, savePath=None, showText=False, labelsize=75):
    import matplotlib.pyplot as plt

    if fill_diagonal:
        np.fill_diagonal(M, 0.0)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.suptitle(f'{title}')
    if showText:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, round(M[i, j], 4),
                        ha="center", va="center", color="k")

    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=labelsize)
    xticks = np.arange(0, M.shape[1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(axis1_ticks, rotation=45, ha='right')
    yticks = np.arange(0, M.shape[0])
    ax.set_yticks(yticks)
    ax.set_yticklabels(axis0_ticks)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')
    if savePath is not None:
        plt.savefig(savePath, transparent=True)
    plt.show()

def mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def randomSelectNSDPhotos(workingDir,
                          iterationNumber=5000):
    import skimage.io as io
    import cv2
    import random
    from PIL import Image
    savePath = f"{workingDir}/NSDPhotoSelection_{iterationNumber}"
    if not os.path.exists(f"{savePath}.pkl"):
        NSDimageFolder = "./images/NSDimageFolder/stimuli"
        NSDimageList = glob(f"{NSDimageFolder}/*.png")

        print(f"len(NSDimageList)={len(NSDimageList)}")

        def generate5000Setof16Images(num_sets=5000):
            import random
            from collections import Counter

            image_pool = list(range(1, 359))


            set_size = 16

            unique_sets = []
            for i in range(num_sets):
                while True:
                    selected_images = random.sample(image_pool, set_size)
                    if selected_images not in unique_sets:
                        if len(selected_images) == len(np.unique(selected_images)):
                            unique_sets.append(selected_images)
                            break

                progress = (i + 1) / num_sets * 100
                print(f"Generating sets: {i + 1}/{num_sets} ({progress:.2f}%)", end="\r")

            print("\nSets generation completed!")

            for set_index, image_set in enumerate(unique_sets):
                assert len(image_set) == len(set(image_set)), f"Duplicate images found in set {set_index + 1}"

            image_counts = Counter(image for image_set in unique_sets for image in image_set)

            for image, count in image_counts.items():
                print("Image", image, ":", count, "occurrences")
            unique_sets = np.array(unique_sets)
            print(f"unique_sets.shape={unique_sets.shape}")
            unique_sets = unique_sets - 1
            print(f"np.min(unique_sets)={np.min(unique_sets)}")
            print(f"np.max(unique_sets)={np.max(unique_sets)}")
            return unique_sets

        selected5000setIDs = generate5000Setof16Images(num_sets=iterationNumber)
        print(f"selected5000setIDs.shape={selected5000setIDs.shape}")

        def checkDim3(imgPath):
            im = Image.open(imgPath)
            if len(np.array(im).shape) == 3:
                return True
            else:
                print(f"np.array(im).shape={np.array(im).shape}")
                return False

        NSDimageDict = {}
        for currIter in tqdm(range(1, 1 + iterationNumber)):
            iter_imgList = []
            for ID in selected5000setIDs[currIter - 1, :]:
                imgPath = NSDimageList[ID]
                iter_imgList.append(imgPath)
            NSDimageDict[currIter] = iter_imgList

        AllSelectedImages = []
        for dictKey in NSDimageDict.keys():
            AllSelectedImages = AllSelectedImages + NSDimageDict[dictKey]
            assert len(np.unique(NSDimageDict[dictKey])) == len(NSDimageDict[dictKey])

        AllSelectedImages_unique = np.unique(AllSelectedImages)
        totalNum = 0
        for uniqueID in AllSelectedImages_unique:
            print(f"{uniqueID} : {list(AllSelectedImages).count(uniqueID)}")
            totalNum += list(AllSelectedImages).count(uniqueID)
        print(f"totalNum={totalNum}")
        assert totalNum == 5000 * 16
        assert len(AllSelectedImages_unique) == len(NSDimageList)
        print(f"save to {savePath}")
        save_obj([NSDimageDict, selected5000setIDs], savePath)
    else:
        print(f"{savePath} exists")


randomSelectNSDPhotos(workingDir, iterationNumber=iterationNumber)


jobarrayNumber = iterationNumber
cmd = f"sbatch --requeue --array=1-{jobarrayNumber + 1} NSD_similarity/baseline.sh"

sbatch_response = custom_run(cmd)
jobID = getjobID_num(sbatch_response)
waitForEnd(jobID)
completed = check_jobArray(jobID=jobID, jobarrayNumber=(1 + jobarrayNumber))

OffDiagonal_mean_C, OffDiagonal_max_C = {}, {}
upperTri_Mean_C, upperTri_Max_C = {}, {}
correlationOfFingerprints_C = []
for jobarrayID in tqdm(range(1, 1 + jobarrayNumber)):
    [ActualCorrelation, correlationOfFingerprints, meanMax_dict] = load_obj(
        f"{scratchResultFolder}/"
        f"nsd/baselineLayerCorr/ActualCorrelation_{jobarrayID}")
    OffDiagonal_mean_C[jobarrayID] = meanMax_dict['OffDiagonal_mean']
    OffDiagonal_max_C[jobarrayID] = meanMax_dict['OffDiagonal_max']
    upperTri_Mean_C[jobarrayID] = meanMax_dict['upperTri_Mean']
    upperTri_Max_C[jobarrayID] = meanMax_dict['upperTri_Max']
    correlationOfFingerprints_C.append(correlationOfFingerprints)
correlationOfFingerprints_C = np.asarray(correlationOfFingerprints_C)


def showNaturalImageSecondOrderCorrelation(correlationOfFingerprints_C):
    from tqdm import tqdm
    correlationOfFingerprints_C_mean = np.mean(correlationOfFingerprints_C, axis=0)
    showMatrix(correlationOfFingerprints_C_mean, title='', axis0_ticks=[], axis1_ticks=[], fill_diagonal=True,
               cmap='seismic',
               vmin=-1, vmax=1,
               savePath=f"/gpfs/milgram/project/turk-browne/projects/localize/result/figures/"
                        f"fingerprints/NaturalImageSecondOrderCorrelation_NSD.pdf")


showNaturalImageSecondOrderCorrelation(correlationOfFingerprints_C)

[ActualCorrelation, correlationOfFingerprints, meanMax_dict] = load_obj(
    f"{scratchResultFolder}/"
    f"nsd/baselineLayerCorr/ActualCorrelation_{1 + jobarrayNumber}")
synthesizedImageResult = [ActualCorrelation, correlationOfFingerprints, meanMax_dict]

mkdir(f"{scratchResultFolder}/"
      f"nsd/baselineLayerCorr/resultSummary/")
save_obj([OffDiagonal_mean_C,
          OffDiagonal_max_C,
          upperTri_Mean_C,
          upperTri_Max_C,
          synthesizedImageResult],
         f"{scratchResultFolder}/"
         f"nsd/baselineLayerCorr/resultSummary/meanMax_summary")


def plotSynthesizedImagesStatistics(synthesizedImageResult):
    correlationOfFingerprints = synthesizedImageResult[1]
    layerList = list([f"layer{i}" for i in np.arange(1, 13)])
    showMatrix(correlationOfFingerprints, title='', axis0_ticks=[], axis1_ticks=[], fill_diagonal=True,
               cmap='seismic',
               vmin=-1, vmax=1,
               savePath=f"/gpfs/milgram/project/turk-browne/projects/localize/result/figures/"
                        f"fingerprints/SynthesizedImagesSecondOrderCorrelation.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    for layer in range(1, 13):
        if layer == 1:
            plt.plot(correlationOfFingerprints[layer - 1, :], label=f"layer{layer}", c='red')
        else:
            plt.plot(correlationOfFingerprints[layer - 1, :], label=f"layer{layer}")
    layerList = list([f"layer{i}" for i in np.arange(1, 13)])
    xticks = np.arange(0, 12)
    ax.set_xticks(xticks)
    ax.set_xticklabels(layerList, rotation=45, ha='right')
    plt.plot([0, 11], [0, 0], c='k')
    plt.legend()

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    for layer in [1, 3, 5, 7, 10, 12]:
        if layer == 1:
            plt.plot(correlationOfFingerprints[layer - 1, :], label=f"layer{layer}", c='red')
        else:
            plt.plot(correlationOfFingerprints[layer - 1, :], label=f"layer{layer}")
    layerList = list([f"layer{i}" for i in np.arange(1, 13)])
    xticks = np.arange(0, 12)
    ax.set_xticks(xticks)
    ax.set_xticklabels(layerList, rotation=45, ha='right')
    plt.plot([0, 11], [0, 0], c='k')
    plt.legend()

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    for layer in [2, 4, 7, 9]:
        if layer == 1:
            plt.plot(correlationOfFingerprints[layer - 1, :], label=f"layer{layer}", c='red')
        else:
            plt.plot(correlationOfFingerprints[layer - 1, :], label=f"layer{layer}")
    layerList = list([f"layer{i}" for i in np.arange(1, 13)])
    xticks = np.arange(0, 12)
    ax.set_xticks(xticks)
    ax.set_xticklabels(layerList, rotation=45, ha='right')
    plt.plot([0, 11], [0, 0], c='k')
    plt.legend()

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    for layer in [1, 2, 3, 4, 5, 7, 9, 10, 12]:
        if layer == 1:
            plt.plot(correlationOfFingerprints[layer - 1, :], label=f"layer{layer}", c='red')
        else:
            plt.plot(correlationOfFingerprints[layer - 1, :], label=f"layer{layer}")
    layerList = list([f"layer{i}" for i in np.arange(1, 13)])
    xticks = np.arange(0, 12)
    ax.set_xtielscks(xticks)
    ax.set_xticklab(layerList, rotation=45, ha='right')
    plt.plot([0, 11], [0, 0], c='k')
    plt.legend()

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    for layer in [6, 8, 11]:
        if layer == 1:
            plt.plot(correlationOfFingerprints[layer - 1, :], label=f"layer{layer}", c='red')
        else:
            plt.plot(correlationOfFingerprints[layer - 1, :], label=f"layer{layer}")
    layerList = list([f"layer{i}" for i in np.arange(1, 13)])
    xticks = np.arange(0, 12)
    ax.set_xticks(xticks)
    ax.set_xticklabels(layerList, rotation=45, ha='right')
    plt.plot([0, 11], [0, 0], c='k')
    plt.legend()


def percentileOfNaturalImages(OffDiagonal_mean_C, OffDiagonal_mean, title='',
                              histColor='grey',
                              fontSize=75, saveToFile=None):
    naturalImages = np.asarray(list(OffDiagonal_mean_C.values()))
    plt.figure(figsize=(20, 20))
    histHeight = plt.hist(naturalImages, bins=50, color=histColor)

    zValue = (OffDiagonal_mean - np.mean(naturalImages)) / np.std(naturalImages)
    import scipy
    p_values = scipy.stats.norm.sf(abs(zValue))
    percentile = np.sum(naturalImages < OffDiagonal_mean)
    print(f"{title} percentile={percentile} zValue={zValue} pValue-{p_values}")

    max_histHeight = np.max(histHeight[0])
    print(f"max_histHeight={max_histHeight}")
    plt.plot([OffDiagonal_mean] * 100,
             np.arange(0, max_histHeight, max_histHeight / 100),
             color='red', linewidth=10)
    plt.title(f"{title}-percentile_{percentile}-zValue_{zValue}-pValue_{p_values}")
    plt.xticks(fontsize=fontSize)
    plt.yticks(fontsize=fontSize)
    plt.xlim([-0.1, 1])

    if saveToFile is not None:
        plt.savefig(saveToFile, transparent=True)


[OffDiagonal_mean_C,
 OffDiagonal_max_C,
 upperTri_Mean_C,
 upperTri_Max_C,
 synthesizedImageResult] = load_obj(f"{scratchResultFolder}/"
                                    f"nsd/baselineLayerCorr/resultSummary/meanMax_summary")

[ActualCorrelation, correlationOfFingerprints, meanMax_dict] = synthesizedImageResult

percentileOfNaturalImages(OffDiagonal_mean_C, meanMax_dict['OffDiagonal_mean'], title='OffDiagonal_mean',
                          saveToFile=f"{scratchResultFolder}/"
                                     f"nsd/baselineLayerCorr/resultSummary/OffDiagonal_mean.pdf")
percentileOfNaturalImages(OffDiagonal_max_C, meanMax_dict['OffDiagonal_max'], title='OffDiagonal_max',
                          saveToFile=f"{scratchResultFolder}/"
                                     f"nsd/baselineLayerCorr/resultSummary/OffDiagonal_max.pdf")
percentileOfNaturalImages(upperTri_Mean_C, meanMax_dict['upperTri_Mean'], title='upperTri_Mean',
                          saveToFile=f"{scratchResultFolder}/"
                                     f"nsd/baselineLayerCorr/resultSummary/upperTri_Mean.pdf")
percentileOfNaturalImages(upperTri_Max_C, meanMax_dict['upperTri_Max'], title='upperTri_Max',
                          saveToFile=f"{scratchResultFolder}/"
                                     f"nsd/baselineLayerCorr/resultSummary/upperTri_Max.pdf")







def plotUpperTriangleDiagram():
    triu = np.triu_indices(12, 1)
    M = np.zeros((12, 12))
    M[:] = -1
    M[triu] = 1
    colors = ['white', (136 / 255, 198 / 255, 86 / 255)]
    cmap = plt.cm.colors.ListedColormap(colors)
    showMatrix(M, title='', axis0_ticks=[], axis1_ticks=[], fill_diagonal=False,
               cmap=cmap,
               vmin=-1, vmax=1,
               savePath=f"/gpfs/milgram/project/turk-browne/projects/localize/result/figures/"
                        f"triu_indices.pdf")
    M = np.zeros((12, 12))
    M[:] = -1
    np.fill_diagonal(M[:, 1:], 1)
    showMatrix(M, title='', axis0_ticks=[], axis1_ticks=[], fill_diagonal=False,
               cmap=cmap,
               vmin=-1, vmax=1,
               savePath=f"/gpfs/milgram/project/turk-browne/projects/localize/result/figures/"
                        f"OffDiagonal.pdf")

    M = np.ones((12, 12))
    np.fill_diagonal(M, -1)
    colors = ['white', (136 / 255, 198 / 255, 86 / 255)]
    cmap = plt.cm.colors.ListedColormap(colors)
    showMatrix(M, title='', axis0_ticks=[], axis1_ticks=[], fill_diagonal=False,
               cmap=cmap,
               vmin=-1, vmax=1,
               savePath=f"/gpfs/milgram/project/turk-browne/projects/localize/result/figures/"
                        f"allExceptDiagonal_indices.pdf")

    M = np.zeros((12, 12))
    M[:] = -1
    np.fill_diagonal(M[:, 1:], 1)
    M = M.T + M
    showMatrix(M, title='', axis0_ticks=[], axis1_ticks=[], fill_diagonal=False,
               cmap=cmap,
               vmin=-1, vmax=1,
               savePath=f"/gpfs/milgram/project/turk-browne/projects/localize/result/figures/"
                        f"OffDiagonal_all.pdf")


plotUpperTriangleDiagram()

print('done')
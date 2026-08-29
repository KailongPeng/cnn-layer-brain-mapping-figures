import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import math
import os

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")
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


def showMatrix(M, title='', axis0_ticks=None, axis1_ticks=None, fill_diagonal=True, cmap='Blues', fontsize=10 * 5,
               figsize=(9 * 10, 8 * 10), vmin=None, vmax=None):
    if axis1_ticks is None:
        axis1_ticks = []
    if axis0_ticks is None:
        axis0_ticks = []
    import matplotlib.pyplot as plt

    if fill_diagonal:
        np.fill_diagonal(M, 0.0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
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


toyDataType = "Sshape"
squareCorrFlag = True
totalDotNumber = 2000
imageNumber = 64
fontSize = 10

minImageIDs = random.sample(range(totalDotNumber), imageNumber)

def getLayerActivations_swissRoll(_totalDotNumber=None):
    sr_points, _sr_color = datasets.make_swiss_roll(n_samples=totalDotNumber, random_state=0)

    fig = plt.figure(figsize=(3*fontSize, 3*fontSize))
    ax = fig.add_subplot(111, projection="3d")
    fig.add_axes(ax)
    ax.scatter(
        sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=_sr_color, s=50, alpha=0.8
    )
    ax.set_title("Swiss Roll in Ambient Space")

    ax.view_init(azim=-66, elev=12)
    _ = ax.text2D(0.8, 0.05, s=f"n_samples={_totalDotNumber}", transform=ax.transAxes)

    _layer1_activation = sr_points
    assert sr_points.shape == (_totalDotNumber, 3)

    sr_lle, sr_err = manifold.locally_linear_embedding(
        sr_points, n_neighbors=12, n_components=2
    )
    sr_tsne = manifold.TSNE(n_components=2, perplexity=40, random_state=0).fit_transform(
        sr_points
    )
    fig, axs = plt.subplots(figsize=(3*fontSize, 3*fontSize), nrows=2)
    axs[0].scatter(sr_lle[:, 0], sr_lle[:, 1], c=_sr_color)
    axs[0].set_title("LLE Embedding of Swiss Roll")
    axs[1].scatter(sr_tsne[:, 0], sr_tsne[:, 1], c=_sr_color)
    _ = axs[1].set_title("t-SNE Embedding of Swiss Roll")
    _layer2_activation = sr_lle
    assert _layer2_activation.shape == (_totalDotNumber, 2)
    return _layer1_activation, _layer2_activation, _sr_color

if toyDataType == "swissRoll":
    layer1_activation, layer2_activation, sr_color = getLayerActivations_swissRoll(_totalDotNumber=totalDotNumber)


def getLayerActivations_S_shape(_totalDotNumber=None):
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    import mpl_toolkits.mplot3d

    from sklearn import manifold, datasets

    n_samples = _totalDotNumber
    S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)
    print(f"S_points.shape={S_points.shape}")
    _layer1_activation = S_points
    _sr_color = S_color

    def plot_3d(points, points_color, title):
        x, y, z = points.T

        x_minImageIDs = x[minImageIDs]
        y_minImageIDs = y[minImageIDs]
        z_minImageIDs = z[minImageIDs]

        fig, ax = plt.subplots(
            figsize=(3*fontSize, 3*fontSize),
            facecolor="white",
            tight_layout=True,
            subplot_kw={"projection": "3d"},
        )
        fig.suptitle(title, size=8*fontSize)

        col = ax.scatter(x, y, z, c=points_color, s=50*fontSize, alpha=0.5)

        ax.scatter(
            x_minImageIDs, y_minImageIDs, z_minImageIDs,
            c='red',
            s=100 * fontSize, alpha=1, label='min_dots'
        )
        ax.view_init(azim=-66, elev=12)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

        fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)

        ax.axis('off')
        plt.savefig(f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/temp/figures/{title}.pdf", transparent=True)
        plt.show()

    def plot_2d(points, points_color, title):

        fig, ax = plt.subplots(figsize=(3 * fontSize, 3 * fontSize), nrows=1)
        ax.scatter(points[:, 0], points[:, 1], c=points_color,
                   s=50 * fontSize, alpha=0.5)
        ax.scatter(points[minImageIDs, 0], points[minImageIDs, 1], c='red',
                   s=100 * fontSize, alpha=1)
        fig.suptitle(title, size=8 * fontSize)
        ax.axis('off')
        plt.savefig(f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/temp/figures/{title}.pdf", transparent=True)
        plt.show()

    def add_2d_scatter(ax, points, points_color=None, title=None, alpha=None, s=None):
        x, y = points.T
        ax.scatter(x, y, c=points_color, s=s, alpha=alpha)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())

    plot_3d(S_points, S_color, "layer1_random_dots")

    n_neighbors = 12
    n_components = 2

    params = {
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "eigen_solver": "auto",
        "random_state": 0,
    }

    lle_mod = manifold.LocallyLinearEmbedding(method="modified", **params)
    S_mod = lle_mod.fit_transform(S_points)

    showAllmethods = False
    if showAllmethods:

        lle_standard = manifold.LocallyLinearEmbedding(method="standard", **params)
        S_standard = lle_standard.fit_transform(S_points)

        lle_ltsa = manifold.LocallyLinearEmbedding(method="ltsa", **params)
        S_ltsa = lle_ltsa.fit_transform(S_points)

        lle_hessian = manifold.LocallyLinearEmbedding(method="hessian", **params)
        S_hessian = lle_hessian.fit_transform(S_points)


        fig, axs = plt.subplots(
            nrows=2, ncols=2, figsize=(3*fontSize, 3*fontSize), facecolor="white", constrained_layout=True
        )
        fig.suptitle("Locally Linear Embeddings", size=8*fontSize)

        lle_methods = [
            ("Standard locally linear embedding", S_standard),
            ("Local tangent space alignment", S_ltsa),
            ("Hessian eigenmap", S_hessian),
            ("Modified locally linear embedding", S_mod),
        ]
        for ax, method in zip(axs.flat, lle_methods):
            name, points = method
            add_2d_scatter(ax, points, S_color, name)

        plt.show()

    _layer2_activation = S_mod

    plot_2d(_layer2_activation, _sr_color, "layer2_random_dots")


    return _layer1_activation, _layer2_activation, _sr_color

if toyDataType == "Sshape":
    layer1_activation, layer2_activation, sr_color = getLayerActivations_S_shape(_totalDotNumber=totalDotNumber)


def get_distanceMatrix(cordinates, returnFingerprint=False):
    distanceMatrix = np.zeros((cordinates.shape[0], cordinates.shape[0]))
    for i in range(cordinates.shape[0]):
        for j in range(i + 1, cordinates.shape[0]):
            distanceMatrix[i, j] = math.dist(cordinates[i], cordinates[j])
    triu = np.triu_indices(imageNumber,
                           1)
    if returnFingerprint:
        return distanceMatrix
    else:
        return distanceMatrix[triu]


def plotBestSet(_minImageID, plot3D=False):

    fig, ax = plt.subplots(figsize=(3*fontSize, 3*fontSize), nrows=1)
    ax.scatter(layer2_activation[:, 0], layer2_activation[:, 1], c=sr_color,
               s=50*fontSize, alpha=0.5)
    ax.scatter(layer2_activation[_minImageID, 0], layer2_activation[_minImageID, 1], c='red',
               s=100*fontSize, alpha=1)
    title = "layer2_synthesized_dots"
    fig.suptitle(title, size=8 * fontSize)
    ax.axis('off')
    plt.savefig(f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/temp/figures/{title}.pdf", transparent=True)
    plt.show()

    corr, layer1_fingerprint, layer2_fingerprint = getCorrForSetID(_minImageID, returnFingerprint=True)
    showFingerprint = False
    if showFingerprint:
        showMatrix(layer1_fingerprint, title='layer1_fingerprint', fill_diagonal=True, fontsize=50)
        showMatrix(layer2_fingerprint, title='layer2_fingerprint', fill_diagonal=True, fontsize=50)

    if plot3D:
        pass
    fig, ax = plt.subplots(
        figsize=(3 * fontSize, 3 * fontSize),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    title = "layer1_synthesized_dots"
    fig.suptitle(title, size=8 * fontSize)

    col = ax.scatter(
        layer1_activation[:, 0], layer1_activation[:, 1], layer1_activation[:, 2], c=sr_color,
        s=50*fontSize, alpha=0.5,
        label='all_dots'
    )
    ax.scatter(
        layer1_activation[_minImageID, 0], layer1_activation[_minImageID, 1], layer1_activation[_minImageID, 2],
        c='red',
        s=100*fontSize, alpha=1,
        label='min_dots'
    )

    ax.view_init(azim=-66, elev=12)

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    ax.axis('off')
    plt.savefig(f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/temp/figures/{title}.pdf", transparent=True)
    plt.show()



def randomSampleSets_plot():
    def sampleImagesAndCalculateCorr():
        imageID = random.sample(list(np.arange(totalDotNumber)), imageNumber)

        layer1_activation_16 = layer1_activation[imageID, :]
        layer2_activation_16 = layer2_activation[imageID, :]

        layer1_fingerprint = get_distanceMatrix(layer1_activation_16)
        layer2_fingerprint = get_distanceMatrix(layer2_activation_16)

        corr = np.corrcoef(layer1_fingerprint, layer2_fingerprint)[0, 1]
        return corr, imageID

    corrs = []
    imageIDs = []
    for _ in tqdm(range(10000)):
        _corr, _imageID = sampleImagesAndCalculateCorr()
        corrs.append(_corr)
        imageIDs.append(_imageID)

    _ = plt.hist(corrs, bins=100)

    smallestID = 0
    corrs_sort = corrs.copy()
    corrs_sort.sort()
    minID = np.where(corrs == corrs_sort[smallestID])[0][0]
    print(f"corrs[minID]={corrs[minID]}")
    _minImageID = imageIDs[minID]
    plotBestSet(_minImageID)




def getCorrForSetID(updatedSetID, returnFingerprint=False):
    imageID = updatedSetID

    layer1_activation_16 = layer1_activation[imageID, :]
    layer2_activation_16 = layer2_activation[imageID, :]

    if returnFingerprint:
        layer1_fingerprint_matrix = get_distanceMatrix(layer1_activation_16, returnFingerprint=returnFingerprint)
        layer2_fingerprint_matrix = get_distanceMatrix(layer2_activation_16, returnFingerprint=returnFingerprint)

        triu = np.triu_indices(imageNumber, 1)
        layer1_fingerprint = layer1_fingerprint_matrix[triu]
        layer2_fingerprint = layer2_fingerprint_matrix[triu]

    else:
        layer1_fingerprint = get_distanceMatrix(layer1_activation_16, returnFingerprint=returnFingerprint)
        layer2_fingerprint = get_distanceMatrix(layer2_activation_16, returnFingerprint=returnFingerprint)
        layer1_fingerprint_matrix, layer2_fingerprint_matrix = None, None

    corr = np.corrcoef(layer1_fingerprint, layer2_fingerprint)[0, 1]
    if squareCorrFlag:
        corr = np.square(corr)
    if returnFingerprint:
        return corr, layer1_fingerprint_matrix, layer2_fingerprint_matrix
    else:
        return corr


def discreteDerivative():
    currSetIDs = np.asarray(minImageIDs)
    updatedCorrs = []
    updatedCorr = None
    _iter = 0
    while True:
        for dotID in range(len(currSetIDs)):
            _iter += 1
            if _iter % 50 == 0:
                print(f"_iter={_iter}")

            def findBestNeighbor(_dotID, _currSetID):
                updatedSetID = _currSetID.copy()
                coordinate = layer2_activation[_currSetID[_dotID]]
                distances = np.zeros(totalDotNumber)
                for currDot in range(totalDotNumber):
                    distances[currDot] = math.dist(coordinate, layer2_activation[currDot])
                distances_copy = distances.copy()
                distances_copy.sort()
                _updatedCorrs = {}
                for closestNeighbor in range(0, 10):
                    neighborID = np.where(distances == distances_copy[closestNeighbor])[0][0]
                    updatedSetID[_dotID] = neighborID
                    _updatedCorr = getCorrForSetID(updatedSetID)
                    _updatedCorrs[_updatedCorr] = neighborID
                best_updatedCorr = np.min(list(_updatedCorrs.keys()))
                updatedSetID[_dotID] = _updatedCorrs[best_updatedCorr]
                return updatedSetID, best_updatedCorr

            currSetIDs, updatedCorr = findBestNeighbor(dotID, currSetIDs)
            if len(updatedCorrs) > 10:
                if updatedCorrs[-9] == updatedCorr:
                    break
            updatedCorrs.append(updatedCorr)
        if len(updatedCorrs) > 10:
            if updatedCorrs[-9] == updatedCorr:
                break
    _ = plt.figure()
    _ = plt.plot(updatedCorrs)
    _ = plt.title("Loss curve")
    _ = plt.show()
    print(
        f"currSetIDs={currSetIDs}")
    print(f"updatedCorr={updatedCorr}")

    plotBestSet(currSetIDs)
    return currSetIDs


BestSetIDs = discreteDerivative()
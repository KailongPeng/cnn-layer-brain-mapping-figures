
testMode = True
partialCorrelationFlag = False
import numpy as np

print(f"np.__version__={np.__version__}")
import sys

print(f"sys.executable={sys.executable}")
print(f"sys.path={sys.path}")
import os

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")

import nibabel as nib
import sys
import time
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import linregress


def save_nib(_nii, fileName):
    _nii = _nii.astype('double')
    _nii[np.isnan(_nii)] = 0
    sl_nii = nib.Nifti1Image(_nii.astype('double'), affine_mat)
    nib.save(sl_nii, fileName)
    print(f"{fileName} saved")


os.chdir("/gpfs/milgram/project/turk-browne/projects/localize/analysis")
jobarrayDict = np.load(f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/"
                       f"findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/sl_jobArray.npy",
                       allow_pickle=True)

jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
jobarrayID = 1
[subject, radius, method, layer, shuffleNormCorrelationSquare_flag] = jobarrayDict[jobarrayID]
layer_1 = layer - 1
print(f"subject={subject}, radius={radius}, method={method}, layer={layer}, layer_1={layer_1}, "
      f"shuffleNormCorrelationSquare_flag={shuffleNormCorrelationSquare_flag}")

subjectDir = "/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/"


copeIm = []
affine_mat = []
copes = []
for cope in range(1, 17):
    print(cope, end='--')
    thisCope = f'{subjectDir}/{subject}/preprocess/allruns.gfeat/cope{cope}.feat/stats/cope1.nii.gz'
    copeIm = nib.load(thisCope)
    copes.append(copeIm)
    affine_mat = copeIm.affine
nii = nib.funcs.concat_images(copes, check_affines=True)

dimsize = copeIm.header.get_zooms()

data = nii.get_data()
print(data.shape)

mask = nib.load(
    f"{subjectDir}/{subject}/preprocess/allruns.gfeat/mask.nii.gz").get_data()
if testMode:
    mask = np.zeros(mask.shape)
    mask[int(91 / 2):(int(91 / 2) + 7), int(109 / 2):(int(109 / 2) + 7), int(91 / 2):(int(91 / 2) + 7)] = 1
print('mask dimensions: {}'.format(mask.shape))
print('number of voxels in mask: {}'.format(np.sum(mask)))

sl_rad = radius
max_blk_edge = 5
pool_size = 1
bcvar = np.load(f'{subjectDir}/../data/inception/features.npy')
__data = data[mask > 0, :]
_sl_mask = np.zeros((7, 7, 7))


def showMatrix(M, title='', axis0_ticks=[], axis1_ticks=[], fill_diagonal=False, cmap='seismic', vmin=-1,
               vmax=1, savePath=None):
    import matplotlib.pyplot as plt

    if fill_diagonal:
        np.fill_diagonal(M, 0.0)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.suptitle(f'{title}')


    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=20)
    xticks = np.arange(0, M.shape[1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(axis1_ticks, rotation=45, ha='right')
    yticks = np.arange(0, M.shape[0])
    ax.set_yticks(yticks)
    ax.set_yticklabels(axis0_ticks)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.savefig(savePath, transparent=True)
    plt.show()


def ModelReg(_data, sl_mask, myrad, bcvar):

    features = bcvar
    triu = np.triu_indices(16,
                           1)
    assert features.shape == (12, 16, 16)
    assert sl_mask.shape == (7, 7, 7)
    data4d = _data.T

    bolddata_sl = data4d
    assert bolddata_sl.shape == (16, 343)
    voxelCubeFingerprint = np.corrcoef(bolddata_sl)
    CNNlayerFingerprints = []
    assert voxelCubeFingerprint.shape == (16, 16)
    ExampleVoxelCubeFingerprint = voxelCubeFingerprint
    showMatrix(ExampleVoxelCubeFingerprint, title='', axis0_ticks=[], axis1_ticks=[], fill_diagonal=True,
               cmap='seismic',
               vmin=-1, vmax=1,
               savePath=f"/gpfs/milgram/project/turk-browne/projects/localize/temp/fingerprints/"
                        f"ExampleVoxelCubeFingerprint.pdf")


ModelReg(__data, _sl_mask, 3, bcvar)



print('done')
import os

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")

testMode = False
partialCorrelationFlag = False
import numpy as np

print(f"np.__version__={np.__version__}")
import sys

print(f"sys.executable={sys.executable}")
print(f"sys.path={sys.path}")

import nibabel as nib
import sys
import time
from scipy.stats import spearmanr
from mpi4py import MPI
from brainiak.searchlight.searchlight import Searchlight
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import linregress

current_dir = os.getcwd()

assert current_dir.endswith('localize_paper'), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")

sys.path.append(current_dir)

from utils import save_nib, setupPaths


jobarrayDict = np.load(f"fig3a/searchlight_train_linearModel/sl_jobArray.npy",
                       allow_pickle=True)

jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
jobarrayID = int(float(sys.argv[1]))
[subject, radius, method, layer, shuffleNormCorrelationSquare_flag] = jobarrayDict[jobarrayID]
layer_1 = layer - 1
print(f"subject={subject}, radius={radius}, method={method}, layer={layer}, layer_1={layer_1}, "
      f"shuffleNormCorrelationSquare_flag={shuffleNormCorrelationSquare_flag}")

[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")
saveToFolder = f'{scratchResultFolder}/findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/' \
               f'/searchlight/{subject}/rad{radius}/'

if not os.path.exists(saveToFolder):
    try:
        os.makedirs(saveToFolder)
    except:
        pass

finalSavingFile = f'{saveToFolder}/{subject}_layer{layer}_{method}_coef.nii.gz'

if not os.path.exists(finalSavingFile):
    def ModelReg(_data, sl_mask, myrad, bcvar):
        features = bcvar
        triu = np.triu_indices(16,
                               1)
        assert features.shape == (12, 16, 16)
        assert sl_mask.shape == (7, 7, 7)
        data4d = _data[0]
        bolddata_sl = data4d.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2], data4d.shape[3]).T
        voxelCubeFingerprint = np.corrcoef(bolddata_sl)
        CNNlayerFingerprints = []
        assert voxelCubeFingerprint.shape == (16, 16)
        voxelCubeFingerprint = voxelCubeFingerprint[triu]

        CNNlayerFingerprint = np.copy(features[layer_1])
        CNNlayerFingerprint = CNNlayerFingerprint[triu]

        assert CNNlayerFingerprint.shape == (120,)
        assert voxelCubeFingerprint.shape == (120,)

        def kp_linear(_x_, _y_):
            _x_ = _x_.reshape(-1, 1)
            _y_ = _y_.reshape(-1, 1)

            model = LinearRegression()
            model.fit(_x_, _y_)
            _R2_ = model.score(_x_, _y_)
            if testMode:
                print(f"model.score(x, y)={_R2_}")
                print(f"model.coef_={model.coef_}")
                print(f"model.intercept_={model.intercept_}")
            return model.coef_[0][0], model.intercept_[0], _R2_

        _coef, _intercept, _R2 = kp_linear(CNNlayerFingerprint, voxelCubeFingerprint)
        print(f"_coef={_coef}")

        return np.asarray([_coef, _intercept, _R2])

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

    data = nii.get_fdata()
    print(data.shape)

    mask = nib.load(
        f"{subjectDir}/{subject}/preprocess/allruns.gfeat/mask.nii.gz").get_fdata()
    if testMode:
        mask = np.zeros(mask.shape)
        mask[int(91 / 2 - 1), int(109 / 2), int(91 / 2)] = 1
        mask[int(91 / 2), int(109 / 2), int(91 / 2)] = 1
        mask[int(91 / 2), int(109 / 2 - 1), int(91 / 2)] = 1

    print('mask dimensions: {}'.format(mask.shape))
    print('number of voxels in mask: {}'.format(np.sum(mask)))

    sl_rad = radius
    max_blk_edge = 5
    pool_size = 1
    bcvar = np.load(f'{subjectDir}/../data/inception/features.npy')

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    print(f"MPI.COMM_WORLD = {comm}")
    print(f"comm.rank = {rank}")
    print(f"comm.size = {size}")

    sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge)

    sl.distribute([data], mask)
    sl.broadcast(bcvar)
    starttime = time.time()
    assert method == 'reg'
    sl_result = sl.run_searchlight(ModelReg)

    total = time.time() - starttime
    print('total time: {}s, estimated time for whole brain: {} h'.format(total,
                                                                         (total / np.sum(mask)) * 239565 / 60 / 60))

    if rank == 0:
        output = f'{saveToFolder}/{subject}_layer{layer}_{method}.npy'

        coef, intercept, R2 = np.zeros(sl_result.shape), np.zeros(sl_result.shape), np.zeros(sl_result.shape)
        for x in range(sl_result.shape[0]):
            for y in range(sl_result.shape[1]):
                for z in range(sl_result.shape[2]):
                    if sl_result[x, y, z] is not None:
                        coef[x, y, z] = sl_result[x, y, z][0]
        for x in range(sl_result.shape[0]):
            for y in range(sl_result.shape[1]):
                for z in range(sl_result.shape[2]):
                    if sl_result[x, y, z] is not None:
                        intercept[x, y, z] = sl_result[x, y, z][1]
        for x in range(sl_result.shape[0]):
            for y in range(sl_result.shape[1]):
                for z in range(sl_result.shape[2]):
                    if sl_result[x, y, z] is not None:
                        R2[x, y, z] = sl_result[x, y, z][2]

        save_nib(coef, '{}_coef.nii.gz'.format(output.split('.npy')[0]), affine_mat=affine_mat)
        save_nib(intercept, '{}_intercept.nii.gz'.format(output.split('.npy')[0]), affine_mat=affine_mat)
        save_nib(R2, '{}_R2.nii.gz'.format(output.split('.npy')[0]), affine_mat=affine_mat)
else:
    print(f'{finalSavingFile} already run')

print('done')
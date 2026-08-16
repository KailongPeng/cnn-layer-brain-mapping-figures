import os

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")

testMode = False
print(f"testMode={testMode}")

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
from sklearn.metrics import r2_score

current_dir = os.getcwd()

assert current_dir.endswith('localize_paper'), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")

sys.path.append(current_dir)

from utils import save_nib, setupPaths, getSubs
[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

jobarrayDict = np.load(f"fig3a/searchlight_score_linearModel/sl_jobArray.npy", allow_pickle=True)

jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]

jobarrayID = int(float(sys.argv[1]))
[subject, radius, method, layer, shuffleNormCorrelationSquare_flag] = jobarrayDict[jobarrayID]
layer_1 = layer - 1
if testMode:
    repetitionNumber = 5
else:
    repetitionNumber = 500
print(f"subject={subject}, radius={radius}, method={method}, layer={layer}, layer_1={layer_1}, "
      f"shuffleNormCorrelationSquare_flag={shuffleNormCorrelationSquare_flag}, repetitionNumber={repetitionNumber}")

saveToFolder = f'{scratchResultFolder}/findLayerInBrain_leaveOneSubOutReg/searchlight_score_linearModel/' \
               f'/searchlight/{subject}/rad{radius}/'

if not os.path.exists(saveToFolder):
    try:
        os.makedirs(saveToFolder)
    except:
        pass

finalSavingFile = f'{saveToFolder}/{subject}_layer{layer}_{method}_all_R2.npy'

otherSubs = getSubs(minusSub=subject)

if not os.path.exists(finalSavingFile):
    def ModelReg(_data, sl_mask, myrad, bcvar):
        features = bcvar
        triu = np.triu_indices(16, 1)
        assert features.shape == (12, 16, 16)
        assert sl_mask.shape == (7, 7, 7)
        data4d = _data[0]
        if testMode:
            print(f"data4d.shape={data4d.shape}")
        assert data4d.shape == (7, 7, 7, 18)

        _coef_ = data4d[3, 3, 3, 0]
        _intercept_ = data4d[3, 3, 3, 1]
        if testMode:
            print(f"_coef_={_coef_}")
            print(f"_intercept_={_intercept_}")

        bolddata_sl = data4d[:, :, :, 2:]
        if testMode:
            print(f"bolddata_sl.shape={bolddata_sl.shape}")
        assert bolddata_sl.shape == (7, 7, 7, 16)

        bolddata_sl = bolddata_sl.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2],
                                          bolddata_sl.shape[3]).T

        assert bolddata_sl.shape == (16, 7 * 7 * 7)

        voxelCubeFingerprint = np.corrcoef(bolddata_sl)
        CNNlayerFingerprints = []
        assert voxelCubeFingerprint.shape == (16, 16)
        voxelCubeFingerprint = voxelCubeFingerprint[triu]

        CNNlayerFingerprint = np.copy(features[layer_1])
        CNNlayerFingerprint = CNNlayerFingerprint[triu]

        assert CNNlayerFingerprint.shape == (120,)
        assert voxelCubeFingerprint.shape == (120,)

        def custom_linear_score(_x_, _y_, _coef, _intercept):
            _x_ = _x_.reshape(-1, 1)
            _y_ = _y_.reshape(-1, 1)
            y_pred = _coef * _x_ + _intercept
            _R2 = r2_score(_y_.reshape(-1, 1), y_pred.reshape(-1, 1))
            if testMode:
                print(f"_R2 = {_R2}")
                print(f"y_pred.shape = {y_pred.shape}")
            return _R2

        true_R2_ = custom_linear_score(CNNlayerFingerprint, voxelCubeFingerprint, _coef_, _intercept_)
        print(f"true_R2_={true_R2_}")
        shuffled_R2 = np.zeros(repetitionNumber)
        for iii in range(repetitionNumber):
            CNNlayerFingerprint_shuffled = CNNlayerFingerprint.copy()
            np.random.shuffle(CNNlayerFingerprint_shuffled)
            random_R2 = custom_linear_score(CNNlayerFingerprint_shuffled, voxelCubeFingerprint,
                                        _coef_, _intercept_)
            shuffled_R2[iii] = random_R2

        All_R2_ = np.zeros(repetitionNumber + 1)
        All_R2_[0] = true_R2_
        All_R2_[1:] = shuffled_R2
        return np.asarray(All_R2_)

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
    print(f"\ndata.shape={data.shape}")
    assert data.shape == (91, 109, 91, 16)

    coefs = np.zeros((91, 109, 91, 29))
    intercepts = np.zeros((91, 109, 91, 29))
    for sub_i, _sub_ in enumerate(otherSubs):
        if _sub_ != subject:
            loadFolder = (f'{scratchResultFolder}/findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/'
                          f'/searchlight/{_sub_}/rad{radius}/')
            coef = nib.load(f"{loadFolder}/{_sub_}_layer{layer}_{method}_coef.nii.gz").get_fdata()
            intercept = nib.load(f"{loadFolder}/{_sub_}_layer{layer}_{method}_intercept.nii.gz").get_fdata()
            coefs[:, :, :, sub_i] = coef
            intercepts[:, :, :, sub_i] = intercept

    coefs = np.nanmean(coefs, axis=3)
    assert coefs.shape == (91, 109, 91)

    coefs = np.expand_dims(coefs, axis=3)
    intercepts = np.expand_dims(
        np.nanmean(intercepts, axis=3),
        axis=3)

    assert coefs.shape == (91, 109, 91, 1)
    assert intercepts.shape == (91, 109, 91, 1)
    weight = np.concatenate((coefs, intercepts), axis=3)
    assert weight.shape == (91, 109, 91, 2)

    data = np.concatenate((weight, data), axis=3)

    assert data.shape == (91, 109, 91, 18)

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
        output = (
            f'{saveToFolder}/{subject}_layer{layer}_{method}_all_R2.npy')

        All_R2 = np.zeros((91, 109, 91, repetitionNumber+1))
        print(f"loading All_R2")
        for x in range(sl_result.shape[0]):
            for y in range(sl_result.shape[1]):
                for z in range(sl_result.shape[2]):
                    if sl_result[x, y, z] is not None:
                        All_R2[x, y, z] = sl_result[x, y, z]
                        assert sl_result[x, y, z].shape == (repetitionNumber+1,)
        print("epoch is:", time.asctime(time.gmtime()))
        print(f"saving {output}")
        print(f"All_R2.shape={All_R2.shape}")
        np.save(output, All_R2)
else:
    print(f'{finalSavingFile} already run')

print('done')
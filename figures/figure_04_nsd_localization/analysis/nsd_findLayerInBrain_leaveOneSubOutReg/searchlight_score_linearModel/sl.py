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

assert current_dir.endswith(('localize_paper', 'cnn-layer-brain-mapping-figures')), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")

sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "figures/figure_04_nsd_localization/analysis"))

from utils import save_nib, setupPaths, get_nsd_Subs
[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

jobarrayDict = np.load(f"figures/figure_04_nsd_localization/analysis/nsd_findLayerInBrain_leaveOneSubOutReg/searchlight_score_linearModel/"
                       f"sl_jobArray_whichSet_1_{int(float(sys.argv[2]))}.npy",
                       allow_pickle=True)

jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
jobarrayID = int(float(sys.argv[1]))
[subject, radius, method, layer, whichSet, setOf16Images, testMode] = jobarrayDict[jobarrayID]
scoreKind = sys.argv[3] if len(sys.argv) > 3 else "zscored"
if scoreKind not in {"zscored", "raw"}:
    raise ValueError(f"scoreKind must be 'zscored' or 'raw', got {scoreKind!r}")
layer_1 = layer - 1
if scoreKind == "raw":
    repetitionNumber = 0
elif testMode:
    repetitionNumber = 5
else:
    repetitionNumber = 500
print(f"subject={subject}, radius={radius}, method={method}, layer={layer}, layer_1={layer_1}, "
      f"repetitionNumber={repetitionNumber}, setOf16Images={setOf16Images}")
print(f"using which set of image? stating from 1 : whichSet = {whichSet}")
print(f"testMode={testMode}")
print(f"scoreKind={scoreKind}")

saveToFolder = (f'{scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/searchlight_score_linearModel/'
                f'{subject}/rad{radius}/whichSet{whichSet}/')

if not os.path.exists(saveToFolder):
    try:
        os.makedirs(saveToFolder)
    except:
        pass

scoreSuffix = "zscored_R2" if scoreKind == "zscored" else "raw_R2"
finalSavingFile = f'{saveToFolder}/{subject}_layer{layer}_{method}_{scoreSuffix}.nii.gz'


def get_nsd_Subs(minusSub=None):
    subjects = [f"sub_i{_sub_i_}" for _sub_i_ in range(1, 9)]
    subjects = [value for value in subjects if value != minusSub]
    return subjects


otherSubs = get_nsd_Subs(minusSub=subject)


def permute_rsm_stimulus_labels(rsm, permutation):
    """Reorder both axes of an RSM with one stimulus-label permutation."""
    rsm = np.asarray(rsm)
    permutation = np.asarray(permutation)
    if rsm.shape != (16, 16):
        raise ValueError(f"expected a 16 x 16 RSM, got {rsm.shape}")
    if not np.array_equal(np.sort(permutation), np.arange(16)):
        raise ValueError("permutation must contain each stimulus index exactly once")
    return rsm[np.ix_(permutation, permutation)]


if not os.path.exists(finalSavingFile):
    def ModelReg(_data, sl_mask, myrad, bcvar):
        features = bcvar
        triu = np.triu_indices(16,
                               1)
        assert features.shape == (12, 16, 16)
        assert sl_mask.shape == (7, 7, 7)
        data4d = _data[0]
        if scoreKind != "raw" or testMode:
            print(f"data4d.shape={data4d.shape}")
        assert data4d.shape == (7, 7, 7, 18)

        _coef_ = data4d[3, 3, 3, 0]
        _intercept_ = data4d[3, 3, 3, 1]
        if scoreKind != "raw" or testMode:
            print(f"_coef_={_coef_}")
            print(f"_intercept_={_intercept_}")

        bolddata_sl = data4d[:, :, :, 2:]
        if scoreKind != "raw" or testMode:
            print(f"bolddata_sl.shape={bolddata_sl.shape}")
        assert bolddata_sl.shape == (7, 7, 7, 16)

        bolddata_sl = bolddata_sl.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2],
                                          bolddata_sl.shape[3]).T

        assert bolddata_sl.shape == (16, 7 * 7 * 7)

        voxelCubeRSM = np.corrcoef(bolddata_sl)
        CNNlayerFingerprints = []
        assert voxelCubeRSM.shape == (16, 16)
        voxelCubeFingerprint = voxelCubeRSM[triu]
        CNNlayerFingerprint = np.copy(features[layer_1])
        CNNlayerFingerprint = CNNlayerFingerprint[triu]

        assert CNNlayerFingerprint.shape == (120,)
        assert voxelCubeFingerprint.shape == (120,)

        def custom_linear_score(_x_, _y_, _coef, _intercept):
            nonNanID = ~np.isnan(_y_)
            _x_ = _x_[nonNanID]
            _y_ = _y_[nonNanID]

            _x_ = _x_.reshape(-1, 1)
            _y_ = _y_.reshape(-1, 1)

            y_pred = _coef * _x_ + _intercept
            _R2 = r2_score(_y_.reshape(-1, 1), y_pred.reshape(-1, 1))
            if testMode:
                print(f"_R2 = {_R2}")
                print(f"y_pred.shape = {y_pred.shape}")
            return _R2

        true_R2_ = custom_linear_score(CNNlayerFingerprint, voxelCubeFingerprint, _coef_, _intercept_)

        # The original NSD pipeline saved only the permutation-normalized z-score.
        # Reviewer feedback now requires the actual cross-validated R² map as well.
        # In raw mode return the already-computed true score immediately, avoiding
        # the 500 permutations that are irrelevant to this figure.
        if scoreKind == "raw":
            return true_R2_

        shuffled_R2 = np.zeros(repetitionNumber)
        for iii in range(repetitionNumber):
            permutation = np.random.permutation(16)
            voxelCubeFingerprint_shuffled = permute_rsm_stimulus_labels(
                voxelCubeRSM, permutation
            )[triu]
            random_R2 = custom_linear_score(
                CNNlayerFingerprint,
                voxelCubeFingerprint_shuffled,
                _coef_,
                _intercept_,
            )
            shuffled_R2[iii] = random_R2

        All_R2_ = np.zeros(repetitionNumber + 1)
        All_R2_[0] = true_R2_
        All_R2_[1:] = shuffled_R2
        zscored = (true_R2_ - np.nanmean(shuffled_R2)) / np.nanstd(shuffled_R2)
        print(f"zscored={zscored}")
        return zscored

    cope = []
    copes = []
    affine_mat = []
    for imgID in setOf16Images:
        cope_ = nib.load(f"{scratchResultFolder}/nsd_result_folder/brainData/data/{subject}_imgID{imgID}_2mm.nii.gz")
        affine_mat = cope_.affine
        cope = cope_.get_fdata()
        assert cope.shape == (91, 109, 91, 3)
        cope = np.mean(cope, axis=3)
        assert cope.shape == (91, 109, 91)
        copes.append(cope)
    copes = np.asarray(copes)
    copes = np.transpose(copes, (1, 2, 3, 0))
    assert copes.shape == (91, 109, 91, 16)

    maskPath = f"{scratchResultFolder}/nsd_result_folder/brainData/data/{subject}_mask_whichSet{whichSet}_2mm.nii.gz"
    mask = nib.load(maskPath).get_fdata()
    print(f"mask[0,0,0]={mask[0, 0, 0]}\n\n")
    if testMode:
        mask = np.zeros(mask.shape)
        mask[int(91 / 2 - 1), int(109 / 2), int(91 / 2)] = 1
        mask[int(91 / 2), int(109 / 2), int(91 / 2)] = 1
        mask[int(91 / 2), int(109 / 2 - 1), int(91 / 2)] = 1

    data = copes
    print(f"data.shape={data.shape}")

    print('mask dimensions: {}'.format(mask.shape))
    print('number of voxels in mask: {}'.format(np.sum(mask)))

    coefs = np.zeros((91, 109, 91, len(otherSubs)))
    intercepts = np.zeros((91, 109, 91, len(otherSubs)))
    for sub_i, _sub_ in enumerate(otherSubs):
        if _sub_ != subject:
            loadFolder = f'{scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/' \
                         f'{_sub_}/rad{radius}/whichSet{whichSet}/'
            coef = nib.load(f"{loadFolder}/{_sub_}_layer{layer}_{method}_coef.nii.gz").get_fdata()
            intercept = nib.load(f"{loadFolder}/{_sub_}_layer{layer}_{method}_intercept.nii.gz").get_fdata()
            coefs[:, :, :, sub_i] = coef
            intercepts[:, :, :, sub_i] = intercept
        else:
            raise Exception

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

    sl_rad = radius
    max_blk_edge = 5
    pool_size = 1
    ActualFingerprints = np.load(f"{scratchResultFolder}/nsd_result_folder/"
                                 f"modelData/data/ActualFingerprints_whichSet_{whichSet}.npy")
    bcvar = ActualFingerprints
    print(f"bcvar.shape={bcvar.shape}")

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
    if partialCorrelationFlag:
        raise Exception
    elif method == 'cor':
        raise Exception
    else:
        sl_result = sl.run_searchlight(ModelReg)

    total = time.time() - starttime
    print('total time: {}s, estimated time for whole brain: {} h'.format(total,
                                                                         (total / np.sum(mask)) * 239565 / 60 / 60))

    if rank == 0:
        output = (
            f'{saveToFolder}/{subject}_layer{layer}_{method}.npy')

        scoreMap = np.zeros((91, 109, 91))
        print(f"loading {scoreSuffix}")
        for x in range(sl_result.shape[0]):
            for y in range(sl_result.shape[1]):
                for z in range(sl_result.shape[2]):
                    if sl_result[x, y, z] is not None:
                        scoreMap[x, y, z] = sl_result[x, y, z]
        print("epoch is:", time.asctime(time.gmtime()))
        print(f"saving {output}")
        print(f"{scoreSuffix}.shape={scoreMap.shape}")
        save_nib(scoreMap, f"{output.split('.npy')[0]}_{scoreSuffix}.nii.gz")
else:
    print(f'{finalSavingFile} already run')

print('done')

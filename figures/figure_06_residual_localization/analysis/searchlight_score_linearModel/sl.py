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
from scipy import stats

current_dir = os.getcwd()

assert current_dir.endswith(('localize_paper', 'cnn-layer-brain-mapping-figures')), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")

sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "figures/figure_06_residual_localization/analysis"))

from utils import save_nib, setupPaths, get_nsd_Subs
from sfig18_nifti_validation import (
    EXPECTED_AFFINE,
    atomic_save_nifti,
    validate_nifti,
)
[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

jobarrayDict = np.load(f"figures/figure_06_residual_localization/analysis/searchlight_score_linearModel/"
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
    repetitionNumber = 500
else:
    repetitionNumber = 500
print(f"subject={subject}, radius={radius}, method={method}, layer={layer}, layer_1={layer_1}, "
      f"repetitionNumber={repetitionNumber}, setOf16Images={setOf16Images}")
print(f"using which set of image? stating from 1 : whichSet = {whichSet}")
print(f"testMode={testMode}")
print(f"scoreKind={scoreKind}")

saveToFolder = f'{scratchResultFolder}/nsd/residualModel/searchlight_score_linearModel/{subject}/rad{radius}/whichSet{whichSet}/'

if not os.path.exists(saveToFolder):
    try:
        os.makedirs(saveToFolder)
    except:
        pass

scoreSuffix = "zscored_R2" if scoreKind == "zscored" else "raw_R2"
finalSavingFile = f'{saveToFolder}/{subject}_layer{layer}_{method}_{scoreSuffix}.nii.gz'

outputReady = os.path.exists(finalSavingFile)
if outputReady and scoreKind == "raw":
    try:
        validate_nifti(finalSavingFile, require_nonzero=True)
        print(f"validated existing raw output: {finalSavingFile}")
    except Exception as exc:
        print(f"existing raw output is invalid and will be recomputed: {exc}")
        outputReady = False


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


if not outputReady:
    def ModelReg(_data, sl_mask, myrad, bcvar):
        triu = np.triu_indices(16, 1)

        assert sl_mask.shape == (7, 7, 7)
        data4d = _data[0]
        if testMode:
            print(f"data4d.shape={data4d.shape}")
        assert data4d.shape == (7, 7, 7, 2 + 16)

        _coef_residual = data4d[3, 3, 3, 0]
        _intercept_residual = data4d[3, 3, 3, 1]
        if testMode:
            print(f"_coef_residual={_coef_residual}")
            print(f"_intercept_residual={_intercept_residual}")

        bolddata_sl = data4d[:, :, :, 2:]
        if testMode:
            print(f"bolddata_sl.shape={bolddata_sl.shape}")
            assert bolddata_sl.shape == (7, 7, 7, 16)

        bolddata_sl = bolddata_sl.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2],
                                          bolddata_sl.shape[3]).T
        if testMode:
            assert bolddata_sl.shape == (16, 7 * 7 * 7)

        voxelCubeRSM = np.corrcoef(bolddata_sl)

        if testMode:
            assert voxelCubeRSM.shape == (16, 16)
        voxelCubeFingerprint = voxelCubeRSM[triu]
        voxelCubeFingerprint = stats.zscore(voxelCubeFingerprint)
        if testMode:
            print(f"np.mean(voxelCubeFingerprint)={np.mean(voxelCubeFingerprint)}")
            assert np.mean(voxelCubeFingerprint) < 1e-10
        if scoreKind != "raw" or testMode:
            print(f"np.mean(voxelCubeFingerprint) = {np.mean(voxelCubeFingerprint)}")

        features = bcvar
        assert features.shape == (12, 120)

        CNNlayerFingerprint = features[layer_1]
        assert CNNlayerFingerprint.shape == (120,)
        assert voxelCubeFingerprint.shape == (120,)

        nonNanID = ~np.isnan(voxelCubeFingerprint)
        CNNlayerFingerprint = CNNlayerFingerprint[nonNanID]
        voxelCubeFingerprint = voxelCubeFingerprint[nonNanID]
        if len(voxelCubeFingerprint) == 0:
            print(f"voxelCubeFingerprint={voxelCubeFingerprint}")
            print(f"CNNlayerFingerprint={CNNlayerFingerprint}")
            zscored = 0
        else:
            def custom_linear_score(_x_=None, _y_=None,
                                _coef_residual=None, _intercept_residual=None):
                if testMode:
                    pass

                _y_pred_residual = _coef_residual * _x_ + _intercept_residual
                _R2_ = r2_score(_y_, _y_pred_residual)
                if testMode:
                    print(f"_y_pred_residual.shape = {_y_pred_residual.shape}")

                return _R2_, _y_pred_residual

            true_R2_, y_pred_residual = custom_linear_score(_x_=CNNlayerFingerprint,
                                                        _y_=voxelCubeFingerprint,
                                                        _coef_residual=_coef_residual,
                                                        _intercept_residual=_intercept_residual)
            if scoreKind != "raw" or testMode:
                print(f"true_R2_={true_R2_}")

            # The original residual pipeline saved only the permutation-normalized
            # score.  S18 also needs the actual cross-validated R².  Return that
            # already-computed value immediately in raw mode; the 500 shuffles do
            # not contribute to the raw map.
            if scoreKind == "raw":
                return true_R2_

            def custom_linear_score_shuffle(_y_=None,
                                        _y_pred_residual=None):
                _R2_random = r2_score(_y_, _y_pred_residual)
                return _R2_random

            shuffled_R2 = np.zeros(repetitionNumber)
            for iii in range(repetitionNumber):
                permutation = np.random.permutation(16)
                voxelCubeFingerprint_shuffled = permute_rsm_stimulus_labels(
                    voxelCubeRSM, permutation
                )[triu]
                voxelCubeFingerprint_shuffled = stats.zscore(
                    voxelCubeFingerprint_shuffled
                )
                valid = ~np.isnan(voxelCubeFingerprint_shuffled)
                y_pred_residual_shuffled = (
                    _coef_residual * features[layer_1][valid]
                    + _intercept_residual
                )
                random_R2 = custom_linear_score_shuffle(
                    _y_=voxelCubeFingerprint_shuffled[valid],
                    _y_pred_residual=y_pred_residual_shuffled,
                )
                shuffled_R2[iii] = random_R2

            shuffled_R2_mean = np.nanmean(shuffled_R2)
            shuffled_R2_std = np.nanstd(shuffled_R2)

            if shuffled_R2_std == 0:
                zscored = 0

                print(f"shuffled_R2_mean={shuffled_R2_mean}")
                print(f"shuffled_R2_std={shuffled_R2_std}")
                print(f"shuffled_R2={shuffled_R2}")
                print(f"CNNlayerFingerprint={CNNlayerFingerprint}")
                print(f"voxelCubeFingerprint={voxelCubeFingerprint}")

                print(f"_coef_full={_coef_residual}")
                print(f"_intercept_full={_intercept_residual}")
            else:
                zscored = (true_R2_ - shuffled_R2_mean) / shuffled_R2_std

        print(f"zscored={zscored}\n")
        return zscored

    cope = []
    copes = []
    affine_mat = []
    for imgID in setOf16Images:
        cope_ = nib.load(f"{scratchResultFolder}/nsd_result_folder/brainData/"
                         f"data/{subject}_imgID{imgID}_2mm.nii.gz")
        affine_mat = cope_.affine
        if not np.allclose(affine_mat, EXPECTED_AFFINE, atol=1e-4):
            raise ValueError(
                f"NSD image {imgID} affine is not the expected MNI 2-mm affine"
            )
        cope = cope_.get_fdata()
        assert cope.shape == (91, 109, 91, 3)
        cope = np.mean(cope, axis=3)
        assert cope.shape == (91, 109, 91)
        copes.append(cope)
    copes = np.asarray(copes)
    copes = np.transpose(copes, (1, 2, 3, 0))
    assert copes.shape == (91, 109, 91, 16)
    assert np.all(np.isfinite(copes))
    if not np.allclose(affine_mat, EXPECTED_AFFINE, atol=1e-4):
        raise ValueError(f"NSD input affine is not the expected MNI 2-mm affine: {affine_mat}")

    maskPath = (f"{scratchResultFolder}/nsd_result_folder/brainData/"
                f"data/{subject}_mask_whichSet{whichSet}_2mm.nii.gz")
    mask_nii = nib.load(maskPath)
    mask = mask_nii.get_fdata()
    if mask.shape != (91, 109, 91) or not np.all(np.isfinite(mask)):
        raise ValueError(f"invalid NSD searchlight mask: {maskPath}")
    if not np.allclose(mask_nii.affine, EXPECTED_AFFINE, atol=1e-4):
        raise ValueError(f"NSD searchlight mask affine is not MNI 2-mm: {maskPath}")
    if not np.any(mask > 0):
        raise ValueError(f"empty NSD searchlight mask: {maskPath}")
    mask_fullVoxelNumber = np.sum(mask)
    print(f"mask_fullVoxelNumber={mask_fullVoxelNumber}")
    if testMode:
        mask = np.zeros(mask.shape)
        mask[int(91 / 2 - 1), int(109 / 2), int(91 / 2)] = 1
        mask[int(91 / 2), int(109 / 2), int(91 / 2)] = 1
        mask[int(91 / 2), int(109 / 2 - 1), int(91 / 2)] = 1

    data = copes
    print(f"data.shape={data.shape}")

    print('mask dimensions: {}'.format(mask.shape))
    print('number of voxels in mask: {}'.format(np.sum(mask)))

    coefs_residual = np.zeros((91, 109, 91, len(otherSubs), 2))
    for sub_i, _sub_ in enumerate(otherSubs):
        if _sub_ != subject:
            loadFolder = f'{scratchResultFolder}/nsd/residualModel/searchlight_train_linearModel/' \
                         f'{_sub_}/rad{radius}/whichSet{whichSet}/'
            coef = nib.load(f"{loadFolder}/{_sub_}_layer{layer}_{method}_coef.nii.gz").get_fdata()
            assert coef.shape == (91, 109, 91, 2)
            if not np.all(np.isfinite(coef)):
                raise ValueError(
                    f"Non-finite residual coefficient values for {_sub_}, "
                    f"layer {layer}, set {whichSet}"
                )
            coefs_residual[:, :, :, sub_i] = coef
        else:
            raise Exception
    coefs_residual = np.nanmean(coefs_residual, axis=3)
    assert coefs_residual.shape == (91, 109, 91, 2)
    if not np.all(np.isfinite(coefs_residual)):
        raise ValueError("LOSO residual coefficients contain non-finite values")

    data = np.concatenate((coefs_residual, data), axis=3)

    assert data.shape == (
        91, 109, 91,
        2 + 16)

    sl_rad = radius
    max_blk_edge = 5
    pool_size = 1
    ActualFingerprints = np.load(f"{scratchResultFolder}/nsd_result_folder/"
                                 f"modelData/data/ActualFingerprints_whichSet_{whichSet}_normalized_upperTriangleArray.npy")
    bcvar = ActualFingerprints
    print(f"bcvar.shape={bcvar.shape}")
    if bcvar.shape != (12, 120) or not np.all(np.isfinite(bcvar)):
        raise ValueError(
            f"Invalid residual CNN fingerprints for set {whichSet}: {bcvar.shape}"
        )

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
    if scoreKind == "raw":
        print(f'total time for raw whole-brain searchlight: {total}s')
    elif testMode:
        print(f'total time: {total}s, estimated time for whole brain: '
              f'{(total / np.sum(mask)) * mask_fullVoxelNumber / repetitionNumber * 500 / 60 / 60} h')
    else:
        print(f'total time: {total}s, estimated time for whole brain: '
              f'{(total / np.sum(mask)) * mask_fullVoxelNumber / repetitionNumber * 500 / 60 / 60} h')

    if rank == 0:
        output = (
            f'{saveToFolder}/{subject}_layer{layer}_{method}.npy')
        if scoreKind == "raw":
            scoreMap = np.zeros((91, 109, 91), dtype=np.float64)
            for x in range(sl_result.shape[0]):
                for y in range(sl_result.shape[1]):
                    for z in range(sl_result.shape[2]):
                        if sl_result[x, y, z] is not None:
                            scoreMap[x, y, z] = sl_result[x, y, z]
        else:
            scoreMap = sl_result
        print("epoch is:", time.asctime(time.gmtime()))
        print(f"saving {output}")
        print(f"{scoreSuffix}.shape={scoreMap.shape}")
        if scoreKind == "raw":
            atomic_save_nifti(
                scoreMap,
                affine_mat,
                finalSavingFile,
                require_nonzero=True,
            )
            print(f"validated and atomically saved {finalSavingFile}")
        else:
            save_nib(
                scoreMap,
                f"{output.split('.npy')[0]}_{scoreSuffix}.nii.gz",
            )
else:
    print(f'{finalSavingFile} already run')

print('done')

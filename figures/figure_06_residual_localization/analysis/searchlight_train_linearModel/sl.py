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

from utils import save_nib, setupPaths
[subjectDir, scratchResultFolder] = setupPaths()
print(f"scratchResultFolder={scratchResultFolder}")


jobarrayDict = np.load(f"figures/figure_06_residual_localization/analysis/searchlight_train_linearModel/"
                       f"sl_jobArray_whichSet_1_{int(float(sys.argv[2]))}.npy", allow_pickle=True)
jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]

jobarrayID = int(float(sys.argv[1]))
[subject, radius, method, layer, whichSet, setOf16Images, testMode] = jobarrayDict[jobarrayID]
layer_1 = layer - 1
print(f"subject={subject}, radius={radius}, method={method}, layer={layer}, layer_1={layer_1}, "
      f"setOf16Images={setOf16Images}")
print(f"using which set of image? stating from 1 : whichSet = {whichSet}")
print(f"testMode={testMode}")


VariableNumber = 11
print(f"VariableNumber={VariableNumber}")

saveToFolder = (f'{scratchResultFolder}/nsd/residualModel/searchlight_train_linearModel/'
                f'{subject}/rad{radius}/whichSet{whichSet}/')

if not os.path.exists(saveToFolder):
    try:
        os.makedirs(saveToFolder)
    except:
        pass

finalSavingFile = f'{saveToFolder}/{subject}_layer{layer}_{method}_coef.nii.gz'

if not os.path.exists(finalSavingFile):
    print(f"obtaining {finalSavingFile}")
    def ModelReg(_data, sl_mask, myrad, bcvar):

        triu = np.triu_indices(16,
                               1)
        assert sl_mask.shape == (7, 7, 7)
        data4d = _data[0]
        bolddata_sl = data4d.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2], data4d.shape[3]).T
        voxelCubeFingerprint = np.corrcoef(bolddata_sl)

        assert voxelCubeFingerprint.shape == (16, 16)
        voxelCubeFingerprint = voxelCubeFingerprint[triu]
        voxelCubeFingerprint = stats.zscore(voxelCubeFingerprint)
        voxelCubeFingerprint = voxelCubeFingerprint
        if testMode:
            print(f"np.mean(voxelCubeFingerprint)={np.mean(voxelCubeFingerprint)}")
            assert np.mean(voxelCubeFingerprint) < 1e-10

        features = bcvar
        CNNlayerFingerprint_X = features[layer_1]
        feature_minusX = np.delete(features, layer_1, 0)

        assert feature_minusX.shape == (VariableNumber, 120)

        CNNlayerFingerprint_minusX = np.transpose(feature_minusX)
        assert voxelCubeFingerprint.shape == (120,)
        assert CNNlayerFingerprint_minusX.shape == (120, VariableNumber)
        assert CNNlayerFingerprint_X.shape == (120,)

        def custom_linear(_CNNlayerFingerprint_minusX=None, _voxelCubeFingerprint=None, _CNNlayerFingerprint_X=None):
            if testMode:
                assert _voxelCubeFingerprint.shape == (120,)
                assert _CNNlayerFingerprint_minusX.shape == (120, VariableNumber)
                print(f"_voxelCubeFingerprint.shape={_voxelCubeFingerprint.shape}")
                print(f"_CNNlayerFingerprint_minusX.shape={_CNNlayerFingerprint_minusX.shape}")
                print(f"_voxelCubeFingerprint.shape={_voxelCubeFingerprint.shape}")
            nonNanID = ~np.isnan(_voxelCubeFingerprint)
            if np.sum(nonNanID) == 0:
                print(f"nonNanID={nonNanID}")
                return np.asarray([0, 0])
            else:
                if testMode:
                    print(f"nonNanID={nonNanID}")
                _CNNlayerFingerprint_minusX = _CNNlayerFingerprint_minusX[nonNanID, :]
                _voxelCubeFingerprint = _voxelCubeFingerprint[nonNanID].reshape(-1, 1)
                _CNNlayerFingerprint_X = _CNNlayerFingerprint_X[nonNanID].reshape(-1, 1)
                if len(_voxelCubeFingerprint) == 0:
                    return np.asarray([0, 0])
                model = LinearRegression()
                try:
                    model.fit(_CNNlayerFingerprint_minusX, _voxelCubeFingerprint)
                except:
                    print(f"_x_={_CNNlayerFingerprint_minusX}")
                    print(f"_y_={_voxelCubeFingerprint}")
                    model.fit(_CNNlayerFingerprint_minusX, _voxelCubeFingerprint)
                    raise Exception
                y_pred = model.predict(_CNNlayerFingerprint_minusX)

                residual = _voxelCubeFingerprint - y_pred
                if testMode:
                    print(f"residual.shape={residual.shape}")
                assert residual.shape == (len(_voxelCubeFingerprint), 1)
                model_residual = LinearRegression()
                model_residual.fit(_CNNlayerFingerprint_X, residual)

                if testMode:
                    print(f"model.coef_={model.coef_}")
                    print(f"model.intercept_={model.intercept_}")
                    assert len(model_residual.coef_) == 1
                return np.asarray([model_residual.coef_[0][0], model_residual.intercept_[0]])

        _coef = custom_linear(_CNNlayerFingerprint_minusX=CNNlayerFingerprint_minusX,
                          _voxelCubeFingerprint=voxelCubeFingerprint,
                          _CNNlayerFingerprint_X=CNNlayerFingerprint_X)

        assert _coef.shape == (2,)
        print(f"_coef={_coef}")
        return _coef

    cope = []
    copes = []
    affine_mat = []
    for imgID in setOf16Images:
        cope_ = nib.load(f"{scratchResultFolder}/nsd_result_folder/brainData/"
                         f"data/{subject}_imgID{imgID}_2mm.nii.gz")
        affine_mat = cope_.affine
        cope = cope_.get_fdata()
        assert cope.shape == (91, 109, 91, 3)
        cope = np.mean(cope, axis=3)
        assert cope.shape == (91, 109, 91)
        copes.append(cope)
    copes = np.asarray(copes)
    copes = np.transpose(copes, (1, 2, 3, 0))
    assert copes.shape == (91, 109, 91, 16)
    assert np.sum(np.isnan(copes)) == 0

    maskPath = (f"{scratchResultFolder}/nsd_result_folder/brainData/"
                f"data/{subject}_mask_whichSet{whichSet}_2mm.nii.gz")
    if os.path.exists(maskPath):
        try:
            mask = nib.load(maskPath).get_fdata()
        except:
            mask = 1 * (np.mean(copes, axis=3) != 0)
            save_nib(mask, f"{scratchResultFolder}/nsd_result_folder/brainData/"
                           f"data/{subject}_mask_whichSet{whichSet}_2mm.nii.gz")
    else:
        mask = 1 * (np.mean(copes, axis=3) != 0)
        save_nib(mask, f"{scratchResultFolder}/nsd_result_folder/brainData/"
                       f"data/{subject}_mask_whichSet{whichSet}_2mm.nii.gz")

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

    sl_rad = radius
    max_blk_edge = 5
    pool_size = 1

    ActualFingerprints = np.load(f"{scratchResultFolder}/nsd_result_folder/"
                                 f"modelData/data/ActualFingerprints_whichSet_{whichSet}_normalized_upperTriangleArray.npy")
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
        sl_result = sl.run_searchlight(ModelSim_partialCorrelation)
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

        coef = np.zeros(sl_result.shape + (2,))
        print(f"coef.shape={coef.shape}")
        for x in range(sl_result.shape[0]):
            for y in range(sl_result.shape[1]):
                for z in range(sl_result.shape[2]):
                    if sl_result[x, y, z] is not None:
                        coef[x, y, z] = sl_result[x, y, z]

        save_nib(coef, '{}_coef.nii.gz'.format(output.split('.npy')[0]))
else:
    print(f'{finalSavingFile} already run')

print('done')
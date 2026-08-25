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

from utils import save_nib, setupPaths

[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

jobarrayDict = np.load(f"figures/figure_04_nsd_localization/analysis/nsd_findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/"
                       f"sl_jobArray_whichSet_1_{int(float(sys.argv[2]))}.npy", allow_pickle=True)
jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]

jobarrayID = int(float(sys.argv[1]))
[subject, radius, method, layer, whichSet, setOf16Images, testMode] = jobarrayDict[jobarrayID]
layer_1 = layer - 1
print(f"subject={subject}, radius={radius}, method={method}, layer={layer}, layer_1={layer_1}, "
      f"setOf16Images={setOf16Images}")
print(f"using which set of image? stating from 1 : whichSet = {whichSet}")
print(f"testMode={testMode}")

saveToFolder = (f'{scratchResultFolder}/nsd/findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/'
                f'{subject}/rad{radius}/whichSet{whichSet}/')

if not os.path.exists(saveToFolder):
    try:
        os.makedirs(saveToFolder)
    except:
        pass

finalSavingFile = f'{saveToFolder}/{subject}_layer{layer}_{method}_coef.nii.gz'

if not os.path.exists(finalSavingFile):
    def ModelReg(_data, sl_mask, myrad, bcvar):
        features = bcvar
        triu = np.triu_indices(16, 1)
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
            nonNanID = ~np.isnan(_y_)
            _x_ = _x_[nonNanID]
            _y_ = _y_[nonNanID]

            _x_ = _x_.reshape(-1, 1)
            _y_ = _y_.reshape(-1, 1)

            model = LinearRegression()
            try:
                model.fit(_x_, _y_)
            except:
                print(f"_x_={_x_}")
                print(f"_y_={_y_}")
                model.fit(_x_, _y_)
                raise Exception
            _R2_ = model.score(_x_, _y_)
            if testMode:
                print(f"model.score(x, y)={_R2_}")
                print(f"model.coef_={model.coef_}")
                print(f"model.intercept_={model.intercept_}")
            return model.coef_[0][0], model.intercept_[0], _R2_

        _coef, _intercept, _R2 = kp_linear(CNNlayerFingerprint, voxelCubeFingerprint)
        print(f"_coef={_coef}")

        return np.asarray([_coef, _intercept, _R2])

    cope = []
    copes = []
    affine_mat = []
    for imgID in setOf16Images:
        cope_ = nib.load(f"{scratchResultFolder}/nsd_result_folder/"
                         f"brainData/data/{subject}_imgID{imgID}_2mm.nii.gz")

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

    maskPath = (f"{scratchResultFolder}/nsd_result_folder/"
                f"brainData/data/{subject}_mask_whichSet{whichSet}_2mm.nii.gz")
    if os.path.exists(maskPath):
        try:
            mask = nib.load(maskPath).get_fdata()
        except:
            mask = 1 * (np.mean(copes, axis=3) != 0)
            save_nib(mask, f"{scratchResultFolder}/nsd_result_folder/"
                           f"brainData/data/{subject}_mask_whichSet{whichSet}_2mm.nii.gz")
    else:
        mask = 1 * (np.mean(copes, axis=3) != 0)
        save_nib(mask, f"{scratchResultFolder}/nsd_result_folder/"
                       f"brainData/data/{subject}_mask_whichSet{whichSet}_2mm.nii.gz")

    print(f"mask[0,0,0]={mask[0,0,0]}\n\n")
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
        coef, intercept = np.zeros(sl_result.shape), np.zeros(sl_result.shape)
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
        save_nib(coef, '{}_coef.nii.gz'.format(output.split('.npy')[0]))
        save_nib(intercept, '{}_intercept.nii.gz'.format(output.split('.npy')[0]))
else:
    print(f'{finalSavingFile} already run')

print('done')
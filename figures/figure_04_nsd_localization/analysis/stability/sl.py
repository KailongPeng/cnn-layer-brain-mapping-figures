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

import gc

print(f"int(float(sys.argv[2]))={int(float(sys.argv[2]))}")

jobarrayDict = np.load(
    f"Sfig3/Sfig3b/"
    f"nsd_repStability_RSA_sl/sl__nsd_repStability_RSA_searchlight_sl_jobArray_whichSet_1_{int(float(sys.argv[2]))}.npy",
    allow_pickle=True)
jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
print(f"jobarrayDict={jobarrayDict}")

jobarrayID = int(float(sys.argv[1]))
[subject, radius, whichSet, setOf16Images, testMode] = jobarrayDict[jobarrayID]
print(f"subject={subject}, radius={radius}, setOf16Images={setOf16Images}")
print(f"using which set of image? stating from 1 : whichSet = {whichSet}")
print(f"testMode = {testMode}")

repStability_RSA_sl_result_folder = f"{scratchResultFolder}/nsd_result_folder/result/repStability_RSA_sl/"
savePath = f'{repStability_RSA_sl_result_folder}{subject}/rad{radius}/whichSet{whichSet}/'

if not os.path.exists(savePath):
    print(f"os.path.exists({savePath}=")
    print(os.path.exists(savePath))
    try:
        os.makedirs(savePath)
    except:
        pass

alreadyRunFile = f'{savePath}/repStability_RSA_sl.nii.gz'
ForceReRun = False
if (not os.path.exists(alreadyRunFile)) or ForceReRun:
    def _num_diagonal(
            m):
        n = int((m * m - m) / 2 * 16)
        return n

    def _num_Noniagonal(
            m):
        assert int(((m * 16) ** 2 - m ** 2 * 16 - m * 16 * (16 - 1)) / 2) == int(16 * (16 - 1) * m * (m - 1) / 2)
        n = int(((m * 16) ** 2 - m ** 2 * 16 - m * 16 * (16 - 1)) / 2)
        return n

    def getRSA_repStability_child(matrix=None, runNumber=0):
        if testMode:
            print(f"matrix.shape={matrix.shape}")
        assert runNumber == 3
        num_diagonal = int(np.nansum(YellowIndex))
        num_Noniagonal = int(np.nansum(BlueIndex))

        diagonal_data = matrix[YellowIndex == 1]
        Non_diagonal_data = matrix[BlueIndex == 1]
        resampleNumber = 5000
        bootstrap_mean = np.zeros(resampleNumber)
        for currIter in range(resampleNumber):
            if testMode:
                if currIter % 1000 == 0:
                    print(f"currIter = {currIter}", end=' ')
            _ = np.mean(Non_diagonal_data[np.random.choice(num_Noniagonal, num_diagonal,
                                                           replace=False)])
            bootstrap_mean[currIter] = _
        diagonal_mean = np.mean(diagonal_data)
        normalizedValue = (diagonal_mean - np.mean(bootstrap_mean)) / np.std(bootstrap_mean)

        del diagonal_data
        del Non_diagonal_data

        gc.collect()
        return normalizedValue

    def repStability_sl(data, sl_mask, myrad, bcvar):
        data4d = data[0]
        runNumber = int(data4d.shape[3] / 16)

        bolddata_sl = data4d.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2], data4d.shape[3]).T
        if testMode:
            assert bolddata_sl.shape == (16 * runNumber, sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2])
        print(f"bolddata_sl.shape={bolddata_sl.shape}")
        bolddata_sl = np.corrcoef(np.asarray(bolddata_sl))
        normalizedValue = getRSA_repStability_child(matrix=bolddata_sl, runNumber=runNumber)

        if testMode:
            print(
                f"data.shape={[_.shape for _ in data]}")
            print(f"sl_mask.shape={sl_mask.shape}")
            print(f"myrad={myrad}")
            print(f"runNumber={runNumber}")
            print(f"normalizedValue = {normalizedValue}")
            print(f"bolddata_sl.shape={bolddata_sl.shape}")
        del bolddata_sl
        del data4d
        del data
        gc.collect()

        print(f"\n5 check memory usage")

        return normalizedValue

    cope = []
    copes = []
    affine_mat = []
    for imgID in setOf16Images:
        cope_ = nib.load(f"{scratchResultFolder}/nsd_result_folder/"
                         f"brainData/data/{subject}_imgID{imgID}_2mm.nii.gz")
        cope = cope_.get_fdata()
        affine_mat = cope_.affine
        assert cope.shape == (91, 109, 91, 3)
        for curr_rep in range(3):
            copes.append(cope[:, :, :, curr_rep])
    copes = np.asarray(copes)
    copes = np.transpose(copes, (1, 2, 3, 0))
    assert copes.shape == (91, 109, 91, 3 * 16)

    mask = 1 * (cope[:, :, :, 0] > 0)
    print(f"mask[0,0,0]={mask[0, 0, 0]}\n\n")
    if testMode:
        mask = np.zeros(mask.shape)
        mask[int(91 / 2 - 1), int(109 / 2), int(91 / 2)] = 1
        mask[int(91 / 2), int(109 / 2), int(91 / 2)] = 1
        mask[int(91 / 2), int(109 / 2 - 1), int(91 / 2)] = 1

    data = copes
    print(f"data.shape = {data.shape}")

    allMaskSum = np.nansum(mask)
    print('number of voxels in mask: {}'.format(allMaskSum))

    print('mask dimensions: {}'.format(mask.shape))
    print('number of voxels in mask: {}'.format(np.sum(mask)))

    sl_rad = radius

    max_blk_edge = 5

    print(f"max_blk_edge={max_blk_edge}")

    pool_size = 1
    bcvar = 0

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    print(f"MPI.COMM_WORLD = {comm}")
    print(f"comm.rank = {rank}")
    print(f"comm.size = {size}")

    print(f"\n1 check memory usage")
    YellowIndex = np.load(f"{savePath}/YellowIndex.npy")

    BlueIndex = np.load(f"{savePath}/BlueIndex.npy")

    startTime = time.time()

    sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge)

    sl.distribute([data], mask)

    sl.broadcast(bcvar)

    sl_result = sl.run_searchlight(repStability_sl)

    if rank == 0:
        output = f'{savePath}/repStability_RSA_sl.nii.gz'
        print(f"sl_result.shape = {sl_result.shape}")
        toFill = sl_result

        toFill = toFill.astype('double')
        toFill[np.isnan(toFill)] = 0

        sl_nii = nib.Nifti1Image(toFill.astype('double'), affine_mat)
        print(f'saving to {output}')
        nib.save(sl_nii, output)

        total = time.time() - startTime
        print(f'total time: {total} seconds')
        print(f'estimated time for whole brain: {total / np.sum(mask) * allMaskSum / 60 / 60} hours')
        print(f"allMaskSum / np.sum(mask) = {allMaskSum / np.sum(mask)}")
        print(f"allMaskSum {allMaskSum}")
        print(f"np.sum(mask) = {np.sum(mask)}")

print('done')
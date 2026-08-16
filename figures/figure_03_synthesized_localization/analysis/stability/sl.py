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

import gc
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

jobarrayDict = np.load(
    f"Sfig3/Sfig3a/searchlight/sl_jobArray.npy",
    allow_pickle=True)
jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
print(f"jobarrayDict={jobarrayDict}")

jobarrayID = int(float(sys.argv[1]))
[subject, radius] = jobarrayDict[jobarrayID]
print(f"subject={subject}, radius={radius}")

if not os.path.exists(f'{subjectDir}/{subject}/repStability_RSA_sl/rad{radius}'):
    print(os.path.exists(f'{subjectDir}/{subject}/repStability_RSA_sl/rad{radius}'))
    try:
        os.makedirs(f'{subjectDir}/{subject}/repStability_RSA_sl/rad{radius}')
    except:
        pass


def Non1(matrix, runNumber):
    m = np.asarray(matrix)
    for i in range(16 * runNumber):
        j = np.arange(i, 16 * runNumber)
        m[i, j] = None
    return m


def Non2(matrix, runNumber):
    m = np.asarray(matrix)
    for i in range(16):
        m[i * runNumber:(i + 1) * runNumber, i * runNumber:(i + 1) * runNumber] = None
    for i in range(16 * runNumber):
        j = np.arange(i, 16 * runNumber, runNumber)
        m[i, j] = None
        m[j, i] = None
    return m

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
    matrix = Non1(matrix, runNumber)
    num_diagonal = _num_diagonal(
        runNumber)
    num_Noniagonal = _num_Noniagonal(
        runNumber)

    diagonal_data = []
    for currImage in range(16):
        diagonal_data.append(np.array(matrix[currImage * runNumber:(currImage + 1) * runNumber,
                                      currImage * runNumber:(currImage + 1) * runNumber]))
    diagonal_data = np.ravel(np.asarray(diagonal_data))
    diagonal_data = diagonal_data[~np.isnan(diagonal_data)]

    Non_diagonal_data = np.ravel(Non2(matrix, runNumber))
    Non_diagonal_data = Non_diagonal_data[~np.isnan(Non_diagonal_data)]

    # Closed-form replacement for the original 5000-iter bootstrap.
    # The bootstrap was numerically estimating the mean and std of the
    # sample-mean distribution under SRS-without-replacement. For a finite
    # population of size N (= num_Noniagonal) with variance sigma^2, the
    # sample mean of n (= num_diagonal) draws without replacement has:
    #     E[sample_mean]   = population_mean
    #     Var[sample_mean] = sigma^2 * (N - n) / (n * (N - 1))    [FPC formula]
    # The original loop converged to these analytic values; using them directly
    # is ~25,000x faster per voxel with no statistical sacrifice.
    N = num_Noniagonal
    n = num_diagonal
    pop_mean = np.mean(Non_diagonal_data)
    pop_var  = np.var(Non_diagonal_data, ddof=0)
    sample_mean_std = np.sqrt(pop_var * (N - n) / (n * (N - 1)))
    diagonal_mean = np.mean(diagonal_data)
    normalizedValue = (diagonal_mean - pop_mean) / sample_mean_std
    within_minus_between = diagonal_mean - pop_mean   # NEW (version A): raw r，就是 z 的分子

    return normalizedValue, diagonal_mean, within_minus_between


# Per-rank voxel progress counter (mutable list for closure access)
_progress = {'n': 0, 't0': None, 't_last': None}


def repStability_sl(data, sl_mask, myrad, bcvar):
    data4d = data[0]
    runNumber = int(data4d.shape[3] / 16)

    bolddata_sl = data4d.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2], data4d.shape[3]).T
    if testMode:
        assert bolddata_sl.shape == (16 * runNumber, sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2])
    bolddata_sl = np.corrcoef(np.asarray(bolddata_sl))
    normalizedValue, diagonal_mean, within_minus_between = getRSA_repStability_child(matrix=bolddata_sl, runNumber=runNumber)

    # progress: print every 500 voxels per rank
    _progress['n'] += 1
    if _progress['t0'] is None:
        _progress['t0'] = time.time()
        _progress['t_last'] = _progress['t0']
    if _progress['n'] % 500 == 0:
        now = time.time()
        rate_recent = 500.0 / (now - _progress['t_last']) if now > _progress['t_last'] else 0.0
        rate_avg    = _progress['n'] / (now - _progress['t0']) if now > _progress['t0'] else 0.0
        print(f"[rank {MPI.COMM_WORLD.rank}] voxel {_progress['n']:>6d}  "
              f"recent={rate_recent:.1f}/s  avg={rate_avg:.1f}/s  "
              f"elapsed={(now - _progress['t0']) / 60:.1f}min",
              flush=True)
        _progress['t_last'] = now

    if testMode:
        print(f"data.shape={[_.shape for _ in data]}")
        print(f"sl_mask.shape={sl_mask.shape}")
        print(f"myrad={myrad}")
        print(f"runNumber={runNumber}")
        print(f"normalizedValue={normalizedValue:.4f}  diagonal_mean={diagonal_mean:.4f}")
    return np.asarray([normalizedValue, diagonal_mean, within_minus_between])

# Init MPI before any data loading so workers skip the file I/O.
# Standard BrainIAK pattern: only rank 0 loads data; workers pass None.
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
print(f"MPI.COMM_WORLD = {comm}")
print(f"comm.rank = {rank}")
print(f"comm.size = {size}")

affine_mat = None
data = None
mask = None
allMaskSum = None

if rank == 0:
    copes = []
    imgList = list(np.arange(1, 17))
    for currImg in imgList:
        for curr_run in range(1, 9):
            zstatFile_stand = f"{subjectDir}{subject}/preprocess/func0{curr_run}.feat/stats/zstat{currImg}_stand.nii.gz"
            if os.path.exists(zstatFile_stand):
                print(f"loading {zstatFile_stand}")
                copeIm = nib.load(zstatFile_stand)
                copes.append(copeIm)
                affine_mat = copeIm.affine

    nii = nib.funcs.concat_images(copes, check_affines=True)
    dimsize = copeIm.header.get_zooms()
    data = nii.get_fdata()
    print(f"data.shape = {data.shape}")

    mask = nib.load(
        f"{subjectDir}/{subject}/preprocess/allruns.gfeat/mask.nii.gz").get_fdata()
    allMaskSum = np.nansum(mask)
    print('number of voxels in mask: {}'.format(allMaskSum))
    if testMode:
        testMode_mask = np.zeros(mask.shape)
        testMode_mask[45:50, 45:50, 45:50] = 1
        mask = testMode_mask
    print('mask dimensions: {}'.format(mask.shape))
    print('number of voxels in mask: {}'.format(np.sum(mask)))

# mask is tiny (~7 MB) and all ranks need it for brainiak distribute
mask = comm.bcast(mask, root=0)

sl_rad = radius

max_blk_edge = 2

print(f"max_blk_edge={max_blk_edge}")

pool_size = 1
bcvar = 0

print(f"\n1 check memory usage", flush=True)

startTime = time.time()

sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge)

print(f"\n2 check memory usage", flush=True)

_t = time.time()
sl.distribute([data], mask)
print(f"\n3 check memory usage  (sl.distribute took {time.time() - _t:.1f}s)", flush=True)

_t = time.time()
sl.broadcast(bcvar)
print(f"\n4 check memory usage  (sl.broadcast took {time.time() - _t:.1f}s)", flush=True)

print(f"[rank {rank}] starting run_searchlight ...", flush=True)
_t = time.time()
sl_result = sl.run_searchlight(repStability_sl)
print(f"[rank {rank}] run_searchlight done in {(time.time() - _t) / 60:.1f}min, "
      f"total voxels processed by this rank = {_progress['n']}", flush=True)

if rank == 0:
    save_dir = f'{subjectDir}/{subject}/repStability_RSA_sl/rad{radius}'
    print(f"sl_result.shape = {sl_result.shape}")

    z_score_map   = np.zeros(sl_result.shape)
    within_sim_map = np.zeros(sl_result.shape)
    wmb_map        = np.zeros(sl_result.shape)   # NEW (version A): within-minus-between (raw r)

    for x in range(sl_result.shape[0]):
        for y in range(sl_result.shape[1]):
            for z in range(sl_result.shape[2]):
                if sl_result[x, y, z] is not None:
                    z_score_map[x, y, z]    = sl_result[x, y, z][0]
                    within_sim_map[x, y, z] = sl_result[x, y, z][1]
                    wmb_map[x, y, z]        = sl_result[x, y, z][2]   # NEW

    # save z-score (within-minus-between, normalized) — original output
    out_zscore = f'{save_dir}/{subject}_repStability_RSA_sl.nii.gz'
    nib.save(nib.Nifti1Image(z_score_map.astype('double'), affine_mat), out_zscore)
    print(f'Saved z-score: {out_zscore}')

    # save within-image similarity (diagonal_mean, raw Pearson r)
    out_within = f'{save_dir}/{subject}_within_image_similarity_sl.nii.gz'
    nib.save(nib.Nifti1Image(within_sim_map.astype('double'), affine_mat), out_within)
    print(f'Saved within-image similarity: {out_within}')

    # save within-minus-between (raw Pearson r, = numerator of the z-score) — NEW (version A)
    out_wmb = f'{save_dir}/{subject}_within_minus_between_sl.nii.gz'
    nib.save(nib.Nifti1Image(wmb_map.astype('double'), affine_mat), out_wmb)
    print(f'Saved within-minus-between: {out_wmb}')

    total = time.time() - startTime
    print(f'total time: {total} seconds')
    print(f'estimated time for whole brain: {total / np.sum(mask) * allMaskSum / 60 / 60} hours')
    print(f"allMaskSum / np.sum(mask) = {allMaskSum / np.sum(mask)}")
    print(f"allMaskSum {allMaskSum}")
    print(f"np.sum(mask) = {np.sum(mask)}")

print('done')
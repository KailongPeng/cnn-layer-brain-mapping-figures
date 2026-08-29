"""
噪声天花板 searchlight worker（MPI 版本）
回应审稿意见 R1-P1

使用 BrainIAK searchlight 框架并行计算每个 searchlight 球体的噪声天花板。
方法：Leave-One-Subject-Out RSM 相关 (Nili et al., 2014)
- 下界：left-out subject RSM vs. mean of OTHER N-1 subjects' RSMs
- 上界：left-out subject RSM vs. mean of ALL N subjects' RSMs

用法：由 sl.sh 通过 srun --mpi=pmi2 调用
  srun --mpi=pmi2 python3 -u .../sl.py ${SLURM_ARRAY_TASK_ID}
"""
import os

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")

testMode = False
print(f"testMode={testMode}")

import numpy as np

print(f"np.__version__={np.__version__}")
import sys

print(f"sys.executable={sys.executable}")
print(f"sys.path={sys.path}")

import nibabel as nib
import time
from mpi4py import MPI
from brainiak.searchlight.searchlight import Searchlight

current_dir = os.getcwd()

assert current_dir.endswith('localize_paper'), f"The directory {current_dir} does not end with 'localize_paper'"

print(f"Current directory is: {current_dir}")

sys.path.append(current_dir)

from utils import save_nib, setupPaths, getSubs

[subjectDir, scratchResultFolder] = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

# Accept SLURM_ARRAY_TASK_ID (always 1 for synthesized)
jobarrayID = int(float(sys.argv[1]))
print(f"jobarrayID={jobarrayID}")

subs = getSubs()
n_subs = len(subs)
n_images = 16
radius = 3

# Output directory
saveToFolder = f'{scratchResultFolder}/rebuttal_analyses/noise_ceiling/'

if not os.path.exists(saveToFolder):
    try:
        os.makedirs(saveToFolder)
    except:
        pass

finalSavingFile = f'{saveToFolder}/noise_ceiling_lower_synthesized.nii.gz'

if not os.path.exists(finalSavingFile):

    # ============================================================
    # BrainIAK searchlight callback
    # ============================================================
    def NoiseCeiling(_data, sl_mask, myrad, bcvar):
        """
        Compute noise ceiling at one searchlight sphere.

        _data[0]: (7, 7, 7, n_subs * n_images) all subjects' cope data stacked
        sl_mask: (7, 7, 7) sphere mask
        bcvar: unused placeholder

        Returns: [nc_lower, nc_upper]
        """
        data4d = _data[0]
        triu = np.triu_indices(n_images, k=1)

        # Reshape: (n_voxels_in_cube, n_subs * n_images)
        n_vox = data4d.shape[0] * data4d.shape[1] * data4d.shape[2]
        flat = data4d.reshape(n_vox, n_subs * n_images)

        # Compute RSM for each subject
        rsm_vecs = []
        for s in range(n_subs):
            bolddata = flat[:, s * n_images:(s + 1) * n_images].T  # (16, n_vox)

            # Remove zero-variance columns
            col_std = np.std(bolddata, axis=0)
            if np.any(col_std == 0):
                valid_cols = col_std > 0
                if np.sum(valid_cols) < 3:
                    return np.asarray([np.nan, np.nan])
                bolddata = bolddata[:, valid_cols]

            rsm = np.corrcoef(bolddata)  # (16, 16)
            rsm_vec = rsm[triu]  # (120,)
            if np.any(np.isnan(rsm_vec)):
                return np.asarray([np.nan, np.nan])
            rsm_vecs.append(rsm_vec)

        rsm_vecs = np.array(rsm_vecs)  # (n_subs, 120)
        rsm_mean_all = np.mean(rsm_vecs, axis=0)

        r_lower_list = []
        r_upper_list = []
        for i in range(n_subs):
            # Upper: corr with mean of ALL N subjects
            r_upper = np.corrcoef(rsm_vecs[i], rsm_mean_all)[0, 1]
            r_upper_list.append(r_upper)

            # Lower: corr with mean of OTHER N-1 subjects
            mask_others = np.ones(n_subs, dtype=bool)
            mask_others[i] = False
            rsm_mean_others = np.mean(rsm_vecs[mask_others], axis=0)
            r_lower = np.corrcoef(rsm_vecs[i], rsm_mean_others)[0, 1]
            r_lower_list.append(r_lower)

        return np.asarray([np.mean(r_lower_list), np.mean(r_upper_list)])

    # ============================================================
    # Data loading: all subjects' cope data
    # ============================================================
    print(f"Loading cope data for {n_subs} subjects ({n_subs * n_images} total images)...")
    all_copes = []
    affine_mat = None
    for s_idx, sub in enumerate(subs):
        print(f"  Loading {sub} ({s_idx + 1}/{n_subs})...")
        copes = []
        for cope in range(1, n_images + 1):
            thisCope = f'{subjectDir}/{sub}/preprocess/allruns.gfeat/cope{cope}.feat/stats/cope1.nii.gz'
            copeIm = nib.load(thisCope)
            copes.append(copeIm)
            if affine_mat is None:
                affine_mat = copeIm.affine
        nii = nib.funcs.concat_images(copes, check_affines=True)
        all_copes.append(nii.get_fdata())  # (91, 109, 91, 16)

    # Concatenate all subjects: (91, 109, 91, n_subs * 16)
    data = np.concatenate(all_copes, axis=3)
    del all_copes  # Free memory
    print(f"data.shape={data.shape}")
    assert data.shape == (91, 109, 91, n_subs * n_images)

    # Load mask from first subject
    mask = nib.load(
        f"{subjectDir}/{subs[0]}/preprocess/allruns.gfeat/mask.nii.gz").get_fdata()
    if testMode:
        mask = np.zeros(mask.shape)
        mask[int(91 / 2 - 1), int(109 / 2), int(91 / 2)] = 1
        mask[int(91 / 2), int(109 / 2), int(91 / 2)] = 1
        mask[int(91 / 2), int(109 / 2 - 1), int(91 / 2)] = 1
    print('mask dimensions: {}'.format(mask.shape))
    print('number of voxels in mask: {}'.format(np.sum(mask)))

    sl_rad = radius
    max_blk_edge = 5

    # bcvar not needed for noise ceiling, pass placeholder
    bcvar = np.array([n_subs, n_images])

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

    sl_result = sl.run_searchlight(NoiseCeiling)

    total = time.time() - starttime
    print('total time: {}s, estimated time for whole brain: {} h'.format(total,
                                                                         (total / np.sum(mask)) * 239565 / 60 / 60))

    if rank == 0:
        nc_lower = np.full((91, 109, 91), np.nan)
        nc_upper = np.full((91, 109, 91), np.nan)

        print("Collecting results...")
        for x in range(sl_result.shape[0]):
            for y in range(sl_result.shape[1]):
                for z in range(sl_result.shape[2]):
                    if sl_result[x, y, z] is not None:
                        nc_lower[x, y, z] = sl_result[x, y, z][0]
                        nc_upper[x, y, z] = sl_result[x, y, z][1]

        print("epoch is:", time.asctime(time.gmtime()))
        print(f"saving to {saveToFolder}")

        save_nib(nc_lower, f'{saveToFolder}/noise_ceiling_lower_synthesized.nii.gz',
                 affine_mat=affine_mat)
        save_nib(nc_upper, f'{saveToFolder}/noise_ceiling_upper_synthesized.nii.gz',
                 affine_mat=affine_mat)

        # Print summary
        valid = ~np.isnan(nc_lower) & (nc_lower != 0)
        print(f"\n=== Noise Ceiling Summary ===")
        print(f"Valid voxels: {np.sum(valid)}")
        if np.sum(valid) > 0:
            print(f"Lower: mean={np.nanmean(nc_lower[valid]):.4f}, max={np.nanmax(nc_lower):.4f}")
            print(f"Upper: mean={np.nanmean(nc_upper[valid]):.4f}, max={np.nanmax(nc_upper):.4f}")
else:
    print(f'{finalSavingFile} already run')

print('done')

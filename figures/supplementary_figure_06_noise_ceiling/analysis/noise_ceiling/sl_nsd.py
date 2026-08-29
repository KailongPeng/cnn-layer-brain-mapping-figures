# Usage: called via sl_nsd.sh (SLURM array 1-16)
# Run from localize_paper root.
# Each array task computes NC for one set of 16 NSD images.
# After all 16 finish, run noise_ceiling_nsd_merge.py to average.

"""
NSD noise ceiling searchlight worker (MPI).
Method: Leave-One-Subject-Out RSM correlation (Nili et al., 2014).

SLURM_ARRAY_TASK_ID = 1..16  →  whichSet_0 = 0..15
  setOf16Images = repeated3imgIDs[16*whichSet_0 : 16*(whichSet_0+1)]

For each searchlight sphere:
  1. Load 16 NSD images × 3 reps → average reps → one map per image.
  2. Compute 16×16 RSM (Pearson corrcoef) within the sphere.
  3. LOSO: lower = corr(RSM_i, mean(RSM_others))
           upper = corr(RSM_i, mean(RSM_all))

Output (per set):
  noise_ceiling_lower_nsd_set{N}.nii.gz
  noise_ceiling_upper_nsd_set{N}.nii.gz
"""

import os
print(f"conda env={os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")

import numpy as np
import sys
import nibabel as nib
import time
from mpi4py import MPI
from brainiak.searchlight.searchlight import Searchlight

current_dir = os.getcwd()
assert current_dir.endswith('localize_paper'), f"Expected localize_paper, got {current_dir}"
sys.path.append(current_dir)

from utils import save_nib, setupPaths

_, scratchResultFolder = setupPaths()
print(f"scratchResultFolder={scratchResultFolder}")

# ── All NSD imgIDs seen ≥3 times by all 8 subjects (256 total, 16 sets × 16) ──
# Verified against: ls .../nsd/brainData/data/sub_i1_imgID*_2mm.nii.gz | sort -n
REPEATED3_IMG_IDS = [
    3050,  3158,  3172,  3435,  3627,  3688,  3810,  3848,  3857,  4052,  4250,  4424,  4668,  4691,  4787,  4836,
    4893,  4931,  5035,  5543,  5584,  5603,  6432,  6445,  6490,  6522,  7121,  7367,  7655,  7841,  7860,  8007,
    8110,  8510,  8647,  8808,  9028,  9049,  9435,  9918, 10047, 10065, 10108, 10245, 10472, 11160, 11522, 11618,
   11690, 11797, 11838, 11845, 11933, 11943, 11953, 12309, 12797, 12923, 13139, 13313, 13315, 14529, 14595, 14794,
   14821, 14899, 15365, 15507, 15820, 15940, 16253, 16345, 16703, 16824, 16918, 17049, 17375, 17486, 17550, 19075,
   19182, 19200, 19934, 20055, 20081, 20207, 20266, 20634, 20651, 20821, 21193, 21195, 21219, 21602, 21951, 22264,
   22516, 22531, 22588, 22795, 22880, 22958, 22968, 22994, 23037, 23716, 23730, 24203, 24215, 24531, 25112, 25269,
   25285, 25589, 25703, 25742, 26164, 26352, 26436, 26459, 26721, 26896, 26910, 26991, 27243, 27276, 27327, 27436,
   27569, 27879, 28069, 28097, 28160, 28190, 28320, 28350, 28525, 28596, 28752, 29382, 29712, 29887, 30082, 30396,
   30408, 30673, 30857, 31029, 31234, 31660, 31748, 31802, 31838, 31937, 31965, 32054, 32233, 32626, 32773, 32892,
   32911, 33172, 33246, 33522, 33544, 34127, 34239, 34830, 34851, 35744, 35753, 35799, 35987, 36577, 36624, 36732,
   36911, 36946, 36975, 36979, 37060, 37225, 37495, 38023, 38026, 38298, 38487, 38642, 38830, 38854, 38979, 39290,
   39370, 39403, 40424, 40576, 40722, 40771, 40841, 40847, 40925, 40936, 41117, 41163, 41567, 41654, 41711, 41928,
   42167, 42172, 42215, 42225, 42564, 42698, 42782, 42852, 42913, 43160, 43165, 43429, 43446, 43620, 43676, 43819,
   43951, 44053, 44721, 44845, 44981, 45130, 45214, 45596, 45751, 45844, 45946, 46151, 46373, 46461, 46463, 47291,
   47294, 47409, 47688, 48158, 48261, 48375, 48423, 48509, 48680, 49467, 49732, 49945, 49956, 50115, 50654, 50735,
]
assert len(REPEATED3_IMG_IDS) == 256, f"Expected 256 imgIDs, got {len(REPEATED3_IMG_IDS)}"

# Array task ID → set index (0-based)
array_task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '1'))
whichSet_0    = array_task_id - 1   # 0-indexed
set_num       = array_task_id       # 1-indexed (for filenames)

SET_IMG_IDS = REPEATED3_IMG_IDS[16 * whichSet_0 : 16 * (whichSet_0 + 1)]
assert len(SET_IMG_IDS) == 16, f"Expected 16 imgIDs for set {set_num}, got {len(SET_IMG_IDS)}"
print(f"Set {set_num} imgIDs: {SET_IMG_IDS}")

NSD_SUBS = [f"sub_i{i}" for i in range(1, 9)]
N_SUBS   = len(NSD_SUBS)   # 8
N_IMAGES = 16
N_REPS   = 3
RADIUS   = 3

# NSD brain data: exists in project dir (stable) and localize_bkp scratch
NSD_DATA_ROOT = "/gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/nsd/brainData/data"

# Save NC results alongside synthesized NC (in localize_bkp)
SAVE_DIR  = scratchResultFolder.replace('/localize/', '/localize_bkp/') + "rebuttal_analyses/noise_ceiling/"
OUT_LOWER = os.path.join(SAVE_DIR, f"noise_ceiling_lower_nsd_set{set_num}.nii.gz")
OUT_UPPER = os.path.join(SAVE_DIR, f"noise_ceiling_upper_nsd_set{set_num}.nii.gz")

os.makedirs(SAVE_DIR, exist_ok=True)

if os.path.exists(OUT_LOWER):
    print(f"Already exists: {OUT_LOWER} — skipping.")
    sys.exit(0)

# ── BrainIAK callback ─────────────────────────────────────────────────────────
def NoiseCeiling(_data, sl_mask, myrad, bcvar):
    data4d = _data[0]
    triu   = np.triu_indices(N_IMAGES, k=1)
    n_vox  = data4d.shape[0] * data4d.shape[1] * data4d.shape[2]
    flat   = data4d.reshape(n_vox, N_SUBS * N_IMAGES)

    rsm_vecs = []
    for s in range(N_SUBS):
        bold    = flat[:, s * N_IMAGES:(s + 1) * N_IMAGES].T  # (16, n_vox)
        col_std = np.std(bold, axis=0)
        valid   = col_std > 0
        if np.sum(valid) < 3:
            return np.asarray([np.nan, np.nan])
        bold    = bold[:, valid]
        rsm     = np.corrcoef(bold)
        rsm_vec = rsm[triu]
        if np.any(np.isnan(rsm_vec)):
            return np.asarray([np.nan, np.nan])
        rsm_vecs.append(rsm_vec)

    rsm_vecs     = np.array(rsm_vecs)
    rsm_mean_all = np.mean(rsm_vecs, axis=0)

    r_lower_list, r_upper_list = [], []
    for i in range(N_SUBS):
        r_upper = np.corrcoef(rsm_vecs[i], rsm_mean_all)[0, 1]
        r_upper_list.append(r_upper)
        mask_others    = np.ones(N_SUBS, dtype=bool)
        mask_others[i] = False
        rsm_mean_oth   = np.mean(rsm_vecs[mask_others], axis=0)
        r_lower        = np.corrcoef(rsm_vecs[i], rsm_mean_oth)[0, 1]
        r_lower_list.append(r_lower)

    return np.asarray([np.mean(r_lower_list), np.mean(r_upper_list)])


# ── Data loading ──────────────────────────────────────────────────────────────
print(f"Loading NSD data: {N_SUBS} subjects × {N_IMAGES} images (avg {N_REPS} reps)...")
all_data   = []
affine_mat = None

for s_idx, sub in enumerate(NSD_SUBS):
    print(f"  {sub} ({s_idx+1}/{N_SUBS})...")
    sub_vols = []
    for img_id in SET_IMG_IDS:
        path = f"{NSD_DATA_ROOT}/{sub}_imgID{img_id}_2mm.nii.gz"
        img  = nib.load(path)
        data = img.get_fdata()               # (91,109,91,3)
        assert data.shape[3] == N_REPS
        sub_vols.append(np.mean(data, axis=3))  # (91,109,91)
        if affine_mat is None:
            affine_mat = img.affine
    all_data.append(np.stack(sub_vols, axis=3))  # (91,109,91,16)

data = np.concatenate(all_data, axis=3)   # (91,109,91, 8*16)
del all_data
print(f"data.shape={data.shape}")
assert data.shape == (91, 109, 91, N_SUBS * N_IMAGES)

mask = (np.abs(data[:, :, :, 0]) > 0).astype(float)
print(f"Brain voxels: {int(mask.sum())}")

# ── BrainIAK searchlight ─────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.rank
print(f"MPI rank={rank}, size={comm.size}")

sl = Searchlight(sl_rad=RADIUS, max_blk_edge=5)
sl.distribute([data], mask)
sl.broadcast(np.array([N_SUBS, N_IMAGES]))

t0        = time.time()
sl_result = sl.run_searchlight(NoiseCeiling)
elapsed   = time.time() - t0
print(f"Searchlight done in {elapsed:.0f}s")

if rank == 0:
    nc_lower = np.full((91, 109, 91), np.nan)
    nc_upper = np.full((91, 109, 91), np.nan)
    for x in range(sl_result.shape[0]):
        for y in range(sl_result.shape[1]):
            for z in range(sl_result.shape[2]):
                if sl_result[x, y, z] is not None:
                    nc_lower[x, y, z] = sl_result[x, y, z][0]
                    nc_upper[x, y, z] = sl_result[x, y, z][1]

    save_nib(nc_lower, OUT_LOWER, affine_mat=affine_mat)
    save_nib(nc_upper, OUT_UPPER, affine_mat=affine_mat)

    valid = np.isfinite(nc_lower) & (nc_lower != 0)
    print(f"\n=== NSD NC Set {set_num} ===")
    print(f"Valid voxels: {valid.sum()}")
    print(f"Lower: mean={np.nanmean(nc_lower[valid]):.4f}, max={np.nanmax(nc_lower):.4f}")
    print(f"Upper: mean={np.nanmean(nc_upper[valid]):.4f}, max={np.nanmax(nc_upper):.4f}")

print("done")

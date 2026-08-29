# Usage:
#   python3 -u rebuttal/analyses/noise_ceiling/noise_ceiling_nsd_merge.py
# Run from localize_paper root AFTER all 16 sl_nsd.sh array jobs finish.
# Averages nc_lower and nc_upper across all 16 NSD sets.

import os
import sys
import numpy as np
import nibabel as nib

current_dir = os.getcwd()
assert current_dir.endswith('localize_paper')
sys.path.append(current_dir)
from utils import save_nib, setupPaths

_, scratch = setupPaths()
NC_DIR = scratch.replace('/localize/', '/localize_bkp/') + "rebuttal_analyses/noise_ceiling"

print("Merging NSD noise ceiling across 16 sets...")
lower_maps, upper_maps = [], []
missing = []

for s in range(1, 17):
    lo = f"{NC_DIR}/noise_ceiling_lower_nsd_set{s}.nii.gz"
    up = f"{NC_DIR}/noise_ceiling_upper_nsd_set{s}.nii.gz"
    if os.path.exists(lo) and os.path.exists(up):
        lower_maps.append(nib.load(lo).get_fdata())
        upper_maps.append(nib.load(up).get_fdata())
        print(f"  Set {s:2d}: loaded")
    else:
        missing.append(s)
        print(f"  Set {s:2d}: MISSING — {lo}")

if missing:
    print(f"\nWARNING: {len(missing)} sets missing: {missing}")
    print("Averaging over available sets only.")

if not lower_maps:
    print("No sets found. Exiting.")
    sys.exit(1)

affine = nib.load(f"{NC_DIR}/noise_ceiling_lower_nsd_set1.nii.gz").affine

nc_lower_mean = np.nanmean(np.stack(lower_maps, axis=-1), axis=-1)
nc_upper_mean = np.nanmean(np.stack(upper_maps, axis=-1), axis=-1)

out_lo = f"{NC_DIR}/noise_ceiling_lower_nsd_mean16sets.nii.gz"
out_up = f"{NC_DIR}/noise_ceiling_upper_nsd_mean16sets.nii.gz"

save_nib(nc_lower_mean, out_lo, affine_mat=affine)
save_nib(nc_upper_mean, out_up, affine_mat=affine)

valid = np.isfinite(nc_lower_mean) & (nc_lower_mean != 0)
print(f"\n=== Merged NSD NC ({len(lower_maps)} sets) ===")
print(f"Valid voxels: {valid.sum()}")
print(f"Lower: mean={np.nanmean(nc_lower_mean[valid]):.4f}, max={np.nanmax(nc_lower_mean):.4f}")
print(f"Upper: mean={np.nanmean(nc_upper_mean[valid]):.4f}, max={np.nanmax(nc_upper_mean):.4f}")
print(f"Saved: {out_lo}")

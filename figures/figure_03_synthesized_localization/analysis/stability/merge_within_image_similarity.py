# Usage:
#   python3 -u figures/figure_03_synthesized_localization/analysis/stability/merge_within_image_similarity.py
# Run from localize_paper root AFTER all 30 subjects' sl.py jobs finish.
# Averages within_image_similarity_sl.nii.gz across subjects → group mean map.

import os
import sys
import numpy as np
import nibabel as nib

current_dir = os.getcwd()
assert current_dir.endswith(('localize_paper', 'cnn-layer-brain-mapping-figures'))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "figures/figure_03_synthesized_localization/analysis"))
from utils import save_nib, setupPaths, getSubs

subs = getSubs()
[subjectDir, scratchResultFolder] = setupPaths()

RADIUS   = 3
OUT_DIR  = os.path.join(scratchResultFolder, f"repStability_RSA_sl/rad{RADIUS}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "within_image_similarity_group_mean.nii.gz")

maps, affine = [], None
missing = []

for sub in subs:
    path = f"{subjectDir}/{sub}/repStability_RSA_sl/rad{RADIUS}/{sub}_within_image_similarity_sl.nii.gz"
    if os.path.exists(path):
        img = nib.load(path)
        maps.append(img.get_fdata())
        if affine is None:
            affine = img.affine
        print(f"  {sub}: loaded")
    else:
        missing.append(sub)
        print(f"  {sub}: MISSING — {path}")

if missing:
    print(f"\nWARNING: {len(missing)} subjects missing: {missing}")

assert maps, "No within-image similarity maps found. Run sl.py first."

group_mean = np.mean(np.stack(maps, axis=-1), axis=-1)
save_nib(group_mean, OUT_PATH, affine_mat=affine)

valid = np.isfinite(group_mean) & (group_mean != 0)
print(f"\n=== Within-image similarity group mean ({len(maps)} subjects) ===")
print(f"Valid voxels : {valid.sum()}")
print(f"Mean r       : {group_mean[valid].mean():.4f}")
print(f"Max r        : {group_mean[valid].max():.4f}")
print(f"Saved        : {OUT_PATH}")

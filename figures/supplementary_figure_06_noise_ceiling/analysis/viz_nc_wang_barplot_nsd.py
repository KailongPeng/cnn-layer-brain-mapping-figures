# Usage:
#   python3 -u rebuttal/analyses/viz_nc_wang_barplot_nsd.py
#   sbatch rebuttal/analyses/viz_nc_wang_barplot_nsd.sh
# Run from localize_paper root directory.

"""
NSD (natural images) noise ceiling — bar plot by Wang visual atlas ROI.

Same layout as viz_nc_wang_barplot.py (synthesized images), for direct comparison.

Inputs:
  noise_ceiling_lower_nsd_set1.nii.gz   (MNI 2mm)
  wang_atlas_mni2mm.nii.gz              (same space)

Output:
  figures/components/noise_ceiling_wang_barplot_nsd.pdf
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from utils import setupPaths

_, scratch = setupPaths()

SCRIPT_DIR    = "rebuttal/analyses"
PAPER_FIG_DIR = "figures/components"
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

NC_DIR  = scratch.replace('/localize/', '/localize_bkp/') + "rebuttal_analyses/noise_ceiling"
NC_PATH = os.path.join(NC_DIR, "noise_ceiling_lower_nsd_mean16sets.nii.gz")
if not os.path.exists(NC_PATH):
    NC_PATH = os.path.join(NC_DIR, "noise_ceiling_lower_nsd_set1.nii.gz")
    print(f"mean16sets not found, using set1: {NC_PATH}")
assert os.path.exists(NC_PATH), f"NSD NC not found: {NC_PATH}\nRun sl_nsd.sh + noise_ceiling_nsd_merge.py first."

label      = "mean16sets" if "mean16sets" in NC_PATH else "set1"
ATLAS_PATH = os.path.join(SCRIPT_DIR, "wang_atlas_mni2mm.nii.gz")
OUT_PDF    = os.path.join(PAPER_FIG_DIR, f"noise_ceiling_wang_barplot_nsd_{label}.pdf")

ROIS = [
    ('V1',    [1,  2],  [26, 27]),
    ('V2',    [3,  4],  [28, 29]),
    ('V3',    [5,  6],  [30, 31]),
    ('hV4',   [7],      [32]),
    ('VO',    [8,  9],  [33, 34]),
    ('PHC',   [10, 11], [35, 36]),
    ('LO',    [14, 15], [39, 40]),
    ('hMT',   [13],     [38]),
    ('V3A/B', [16, 17], [41, 42]),
    ('IPS0',  [18],     [43]),
    ('IPS1',  [19],     [44]),
    ('IPS2',  [20],     [45]),
    ('IPS3',  [21],     [46]),
]

print(f"Loading NSD nc_lower: {NC_PATH}")
nc_data    = nib.load(NC_PATH).get_fdata()
atlas_data = nib.load(ATLAS_PATH).get_fdata()

assert nc_data.shape == atlas_data.shape, \
    f"Shape mismatch: nc={nc_data.shape}, atlas={atlas_data.shape}"
print(f"Shape OK: {nc_data.shape}")

roi_names, roi_means, roi_sems = [], [], []

for name, right_labels, left_labels in ROIS:
    all_labels = right_labels + left_labels
    mask = np.zeros(atlas_data.shape, dtype=bool)
    for lbl in all_labels:
        mask |= (atlas_data == lbl)

    vals = nc_data[mask]
    vals = vals[np.isfinite(vals) & (vals > -0.5)]

    if len(vals) == 0:
        print(f"  WARNING: {name} — no valid voxels, skipping")
        continue

    mean = vals.mean()
    sem  = vals.std() / np.sqrt(len(vals))
    print(f"  {name:8s}  n={len(vals):4d}  mean={mean:.4f}  sem={sem:.4f}")

    roi_names.append(name)
    roi_means.append(mean)
    roi_sems.append(sem)

roi_means = np.array(roi_means)
roi_sems  = np.array(roi_sems)

fig, ax = plt.subplots(figsize=(10, 5))
x      = np.arange(len(roi_names))
colors = plt.cm.inferno(np.linspace(0.15, 0.85, len(roi_names)))

ax.bar(x, roi_means, yerr=roi_sems, capsize=4,
       color=colors, edgecolor='white', linewidth=0.5,
       error_kw=dict(elinewidth=1, ecolor='#444444', capthick=1))

valid_all  = nc_data[np.isfinite(nc_data) & (nc_data > -0.5)]
brain_mean = valid_all.mean()
ax.axhline(brain_mean, color='gray', linestyle='--', linewidth=1,
           label=f'Whole-brain mean (r={brain_mean:.3f})')

ax.set_xticks(x)
ax.set_xticklabels(roi_names, fontsize=11)
ax.set_ylabel('Noise ceiling lower bound\n(LOO Pearson r)', fontsize=12)
ax.set_xlabel('Visual area (Wang et al., 2015)', fontsize=12)
ax.set_title('NSD natural images: noise ceiling across the visual hierarchy\n'
             '(bilateral ROIs, N=8 subjects, set 1)',
             fontsize=13)
ax.legend(fontsize=10, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(bottom=0)

for i, (name, right_labels, left_labels) in enumerate(ROIS):
    if name not in roi_names:
        continue
    all_labels = right_labels + left_labels
    mask = np.zeros(atlas_data.shape, dtype=bool)
    for lbl in all_labels:
        mask |= (atlas_data == lbl)
    n = int(np.sum(mask))
    ax.text(i, -0.003, f'n={n}', ha='center', va='top', fontsize=7, color='#555555')

plt.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight', transparent=True)
plt.close()
print(f"\nSaved: {OUT_PDF}")

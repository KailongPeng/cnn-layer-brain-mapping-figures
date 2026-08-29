# Usage:
#   python3 -u rebuttal/analyses/viz_nc_wang_barplot.py
#   sbatch rebuttal/analyses/viz_nc_wang_barplot.sh
# Run from localize_paper root directory.

"""
Noise ceiling lower bound — bar plot by Wang visual atlas ROI.

Jeff's suggestion: "a complementary panel that plots maybe the noise ceiling
as a function of either the Wang visual atlas (bar plot probably)"

For each bilateral ROI (V1 → IPS, hierarchical order), extract mean nc_lower
across voxels in that ROI and plot as a bar chart.

Inputs:
  noise_ceiling_lower_synthesized.nii.gz   (MNI 2mm, shape 91×109×91)
  wang_atlas_mni2mm.nii.gz                 (same space, from download_wang_atlas.py)

Output:
  figures/components/noise_ceiling_wang_barplot.pdf
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

SCRIPT_DIR   = "rebuttal/analyses"
PAPER_FIG_DIR = "figures/components"
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

NC_PATH    = os.path.join(scratch, "rebuttal_analyses/noise_ceiling/noise_ceiling_lower_synthesized.nii.gz")
ATLAS_PATH = os.path.join(SCRIPT_DIR, "wang_atlas_mni2mm.nii.gz")
OUT_PDF    = os.path.join(PAPER_FIG_DIR, "noise_ceiling_wang_barplot.pdf")

# ── Bilateral ROI definitions (from wang_atlas_labels.txt) ───────────────────
# Labels 1–25 = right hemisphere, 26–50 = left hemisphere
# Each entry: (display_name, [right_labels], [left_labels])
ROIS = [
    ('V1',    [1,  2],       [26, 27]),
    ('V2',    [3,  4],       [28, 29]),
    ('V3',    [5,  6],       [30, 31]),
    ('hV4',   [7],           [32]),
    ('VO',    [8,  9],       [33, 34]),
    ('PHC',   [10, 11],      [35, 36]),
    ('LO',    [14, 15],      [39, 40]),
    ('hMT',   [13],          [38]),
    ('V3A/B', [16, 17],      [41, 42]),
    ('IPS0',  [18],          [43]),
    ('IPS1',  [19],          [44]),
    ('IPS2',  [20],          [45]),
    ('IPS3',  [21],          [46]),
]

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading nc_lower: {NC_PATH}")
nc_img  = nib.load(NC_PATH)
nc_data = nc_img.get_fdata()

print(f"Loading Wang atlas: {ATLAS_PATH}")
atlas_data = nib.load(ATLAS_PATH).get_fdata()

# Verify alignment
assert nc_data.shape == atlas_data.shape, \
    f"Shape mismatch: nc={nc_data.shape}, atlas={atlas_data.shape}"
print(f"Shape OK: {nc_data.shape}")

# ── Extract per-ROI mean nc_lower ─────────────────────────────────────────────
roi_names = []
roi_means = []
roi_sems  = []

for name, right_labels, left_labels in ROIS:
    all_labels = right_labels + left_labels
    mask = np.zeros(atlas_data.shape, dtype=bool)
    for lbl in all_labels:
        mask |= (atlas_data == lbl)

    vals = nc_data[mask]
    # Keep only finite positive values (exclude background/NaN voxels)
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

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(roi_names))
colors = plt.cm.inferno(np.linspace(0.15, 0.85, len(roi_names)))

bars = ax.bar(x, roi_means, yerr=roi_sems, capsize=4,
              color=colors, edgecolor='white', linewidth=0.5,
              error_kw=dict(elinewidth=1, ecolor='#444444', capthick=1))

# Reference line: whole-brain mean nc_lower
valid_all = nc_data[np.isfinite(nc_data) & (nc_data > -0.5)]
brain_mean = valid_all.mean()
ax.axhline(brain_mean, color='gray', linestyle='--', linewidth=1,
           label=f'Whole-brain mean (r={brain_mean:.3f})')

ax.set_xticks(x)
ax.set_xticklabels(roi_names, fontsize=11)
ax.set_ylabel('Noise ceiling lower bound\n(LOO Pearson r)', fontsize=12)
ax.set_xlabel('Visual area (Wang et al., 2015)', fontsize=12)
ax.set_title('Noise ceiling across the visual hierarchy\n(bilateral ROIs, N=30 subjects)',
             fontsize=13)
ax.legend(fontsize=10, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(bottom=0)

# Annotate n voxels
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

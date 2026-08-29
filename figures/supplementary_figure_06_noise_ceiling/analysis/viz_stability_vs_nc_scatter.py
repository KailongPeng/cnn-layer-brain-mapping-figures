# Usage:
#   python3 -u rebuttal/analyses/viz_stability_vs_nc_scatter.py
#   sbatch rebuttal/analyses/viz_stability_vs_nc_scatter.sh
# Run from localize_paper root directory.

"""
Scatter plot: NC lower bound (x) vs repStability t-stat (y), per voxel.

Two panels:
  Left : all brain voxels gray; stable voxels (TFCE corrp > 0.95) colored red.
  Right: same axes; stable voxels colored by Wang visual ROI membership.

Motivation: the NC lower bound is the "ideal ceiling" for within-image reliability.
repStability (within-minus-between, group t-stat) measures discriminability.
Stable voxels should cluster top-right: high NC AND high repStability.

Output: figures/components/stability_vs_nc_scatter.pdf
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, '.')
from utils import setupPaths

_, scratch = setupPaths()

PAPER_FIG_DIR = "figures/components"
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# ── File paths ────────────────────────────────────────────────────────────────
REPSTAB_DIR   = "/gpfs/milgram/project/turk-browne/projects/localize/analysis/results/repStability_RSA_sl/rad3"
TSTAT_PATH    = os.path.join(REPSTAB_DIR, "__tstat1.nii.gz")
TFCE_PATH     = os.path.join(REPSTAB_DIR, "__tfce_corrp_tstat1.nii.gz")

NC_DIR        = scratch.replace('/localize/', '/localize_bkp/') + "rebuttal_analyses/noise_ceiling"
NC_PATH       = os.path.join(NC_DIR, "noise_ceiling_lower_synthesized.nii.gz")

SCRIPT_DIR    = "rebuttal/analyses"
ATLAS_PATH    = os.path.join(SCRIPT_DIR, "wang_atlas_mni2mm.nii.gz")

OUT_PDF       = os.path.join(PAPER_FIG_DIR, "stability_vs_nc_scatter.pdf")

for p in [TSTAT_PATH, TFCE_PATH, NC_PATH, ATLAS_PATH]:
    assert os.path.exists(p), f"Missing: {p}"

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading maps...")
tstat = nib.load(TSTAT_PATH).get_fdata()
tfce  = nib.load(TFCE_PATH).get_fdata()
nc    = nib.load(NC_PATH).get_fdata()
atl   = nib.load(ATLAS_PATH).get_fdata()

print(f"  tstat shape: {tstat.shape}, nc shape: {nc.shape}")
assert tstat.shape == nc.shape == atl.shape, \
    f"Shape mismatch: tstat={tstat.shape}, nc={nc.shape}, atlas={atl.shape}"

# ── Masks ─────────────────────────────────────────────────────────────────────
# Brain voxels: both maps must be finite and non-zero
brain_mask  = (np.isfinite(tstat) & (tstat != 0) &
               np.isfinite(nc)    & (nc    != 0))
stable_mask = brain_mask & (tfce > 0.95)

x_all    = nc[brain_mask]
y_all    = tstat[brain_mask]
x_stable = nc[stable_mask]
y_stable = tstat[stable_mask]

print(f"Brain voxels : {brain_mask.sum()}")
print(f"Stable voxels: {stable_mask.sum()}  ({100*stable_mask.sum()/brain_mask.sum():.1f}%)")

# Subsample gray background for speed (keep all stable voxels)
rng   = np.random.default_rng(42)
n_bg  = min(50000, len(x_all))
idx   = rng.choice(len(x_all), size=n_bg, replace=False)

# ── Wang ROIs ─────────────────────────────────────────────────────────────────
ROIS = [
    ('V1',    [1, 2],   [26, 27]),
    ('V2',    [3, 4],   [28, 29]),
    ('V3',    [5, 6],   [30, 31]),
    ('hV4',   [7],      [32]),
    ('VO',    [8, 9],   [33, 34]),
    ('PHC',   [10, 11], [35, 36]),
    ('LO',    [14, 15], [39, 40]),
    ('hMT',   [13],     [38]),
    ('V3A/B', [16, 17], [41, 42]),
    ('IPS0',  [18],     [43]),
    ('IPS1',  [19],     [44]),
    ('IPS2',  [20],     [45]),
    ('IPS3',  [21],     [46]),
]
roi_colors = plt.cm.tab10(np.linspace(0, 0.9, len(ROIS)))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

xlim = (np.percentile(x_all, 0.5), np.percentile(x_all, 99.5))
ylim = (np.percentile(y_all, 0.5), np.percentile(y_all, 99.5))

for ax_idx, ax in enumerate(axes):
    # Gray background (subsampled)
    ax.scatter(x_all[idx], y_all[idx], s=1, c='#cccccc', alpha=0.25,
               rasterized=True, label='All brain voxels')

    if ax_idx == 0:
        # Left panel: stable voxels in solid red
        ax.scatter(x_stable, y_stable, s=3, c='#d62728', alpha=0.7,
                   rasterized=True, label=f'Stable voxels (TFCE corrp>0.95, n={stable_mask.sum():,})')
        handles = [
            mpatches.Patch(color='#cccccc', label='All brain voxels'),
            mpatches.Patch(color='#d62728',
                           label=f'Stable (TFCE corrp>0.95, n={stable_mask.sum():,})'),
        ]
        ax.legend(handles=handles, fontsize=9, frameon=False)
        ax.set_title('Stable voxels highlighted\n(TFCE corrp > 0.95)', fontsize=12)

    else:
        # Right panel: stable voxels colored by Wang ROI
        in_any_roi = np.zeros(tstat.shape, dtype=bool)
        for (name, right_labels, left_labels), color in zip(ROIS, roi_colors):
            roi_mask = np.zeros(atl.shape, dtype=bool)
            for lbl in right_labels + left_labels:
                roi_mask |= (atl == lbl)
            combined = stable_mask & roi_mask
            in_any_roi |= roi_mask
            if combined.sum() == 0:
                continue
            ax.scatter(nc[combined], tstat[combined], s=5, color=color,
                       alpha=0.75, rasterized=True, label=name)

        # Stable voxels not in any Wang ROI
        outside = stable_mask & ~in_any_roi
        if outside.sum() > 0:
            ax.scatter(nc[outside], tstat[outside], s=3, c='#d62728',
                       alpha=0.5, rasterized=True, label='Stable (no ROI)')

        ax.legend(fontsize=8, frameon=False, ncol=2, markerscale=3)
        ax.set_title('Stable voxels colored by Wang ROI\n(bilateral, rad=3)', fontsize=12)

    # Reference lines
    ax.axhline(0, color='k', lw=0.6, alpha=0.4, linestyle='--')
    ax.axvline(0, color='k', lw=0.6, alpha=0.4, linestyle='--')

    ax.set_xlabel('NC lower bound — synthesized images\n(LOO Pearson r, group mean)', fontsize=11)
    ax.set_ylabel('repStability t-stat\n(within-minus-between, group level)', fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Correlation annotation on left panel
from scipy.stats import pearsonr
r, p = pearsonr(x_all, y_all)
axes[0].text(0.97, 0.05, f'r = {r:.3f} (all voxels)', transform=axes[0].transAxes,
             ha='right', fontsize=9, color='#555555')
r2, _ = pearsonr(x_stable, y_stable)
axes[0].text(0.97, 0.10, f'r = {r2:.3f} (stable only)', transform=axes[0].transAxes,
             ha='right', fontsize=9, color='#d62728')

plt.suptitle('NC lower bound vs. representational stability (within-minus-between)\n'
             'Synthesized images, N=30 subjects, searchlight rad=3',
             fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=150, transparent=False)
plt.close()
print(f"\nSaved: {OUT_PDF}")

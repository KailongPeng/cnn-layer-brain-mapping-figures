# Usage:
#   sbatch rebuttal/analyses/viz_nc_surface_nsd.sh          # recommended (HPC)
#   nohup python3 -u rebuttal/analyses/visualize_nc_surface_nsd.py > logs/nc_surface_nsd.log 2>&1 &
# Both must be run from the localize_paper root directory.

"""
NSD (natural images) noise ceiling lower bound — surface plot in fig3 style.

Same aesthetic as visualize_nc_surface.py (synthesized images), for direct comparison.

Input:  SCRATCH_RESULT/rebuttal_analyses/noise_ceiling/noise_ceiling_lower_nsd_set1.nii.gz
Output: figures/components/noise_ceiling_lower_surface_nsd.pdf
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nilearn import datasets, plotting, surface

current_dir = os.getcwd()
assert current_dir.endswith('localize_paper'), \
    f"Run from localize_paper root, not {current_dir}"

sys.path.append(current_dir)
from utils import setupPaths

_, scratchResultFolder = setupPaths()

NC_DIR  = scratchResultFolder.replace('/localize/', '/localize_bkp/') + "rebuttal_analyses/noise_ceiling"
# Prefer mean across 16 sets; fall back to set 1 if merge hasn't run yet.
NC_PATH = os.path.join(NC_DIR, "noise_ceiling_lower_nsd_mean16sets.nii.gz")
if not os.path.exists(NC_PATH):
    NC_PATH = os.path.join(NC_DIR, "noise_ceiling_lower_nsd_set1.nii.gz")
    print(f"mean16sets not found, using set1: {NC_PATH}")
assert os.path.exists(NC_PATH), f"NSD NC not found: {NC_PATH}\nRun sl_nsd.sh + noise_ceiling_nsd_merge.py first."

PAPER_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures", "components")
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
label    = "mean16sets" if "mean16sets" in NC_PATH else "set1"
OUT_PDF  = os.path.join(PAPER_FIG_DIR, f"noise_ceiling_lower_surface_nsd_{label}.pdf")
OUT_PNG  = os.path.join(PAPER_FIG_DIR, f"noise_ceiling_lower_surface_nsd_{label}.png")


# 纯 inferno，不用 designCmap 的白色下半段——理由同 visualize_nc_surface.py:
# 那个双段色标是给对称范围(-vmax~vmax)设计的，裁负值到 0 后如果继续用它、又让
# nilearn 按实际数据范围自动重铺色标，白色下半段会吞掉一大截中低正值信号。
cmap = 'inferno'

nii  = nib.load(NC_PATH)
nc   = nii.get_fdata()

valid = nc[np.isfinite(nc) & (nc > 0)]
vmax  = float(np.percentile(valid, 99))
print(f"NSD nc_lower: mean={valid.mean():.4f}, max={valid.max():.4f}, vmax(p99)={vmax:.4f}")

# 负值只是噪声、没有可解释的意义，裁到非负后色标范围会贴着 0 起跳(同 visualize_nc_surface.py)。
nc  = np.clip(nc, 0, None)
nii = nib.Nifti1Image(nc, nii.affine, nii.header)

fsaverage = datasets.fetch_surf_fsaverage()

base_w = 0.4
base_h = 0.4
positions = {
    "lateral-left":    [0.00,                  0.23854166, base_w, base_h],
    "lateral-right":   [0.20693055555555548,    0.23854166, base_w, base_h],
    "medial-left":     [0.00,                  0.00,        base_w, base_h],
    "medial-right":    [0.20693055555555548,    0.00,        base_w, base_h],
    "posterior-left":  [0.364,                 0.11,        base_w, base_h],
    "posterior-right": [0.44599999999999995,   0.11,        base_w, base_h],
}

views = ['lateral', 'medial', 'posterior']
hemis = ['left', 'right']

fig = plt.figure(figsize=(3 * 10, 2 * 10))
fig.patch.set_facecolor('white')
fig.suptitle('NSD (natural images) noise ceiling lower bound (LOO Pearson r)',
             fontsize=24, y=1.01)
plt.axis('off')

for hemi in hemis:
    if hemi == 'left':
        texture = surface.vol_to_surf(nii, fsaverage.pial_left,  interpolation='nearest')
        bg_map  = fsaverage.sulc_left
        surf    = fsaverage.infl_left
    else:
        texture = surface.vol_to_surf(nii, fsaverage.pial_right, interpolation='nearest')
        bg_map  = fsaverage.sulc_right
        surf    = fsaverage.infl_right

    for view in views:
        key = f"{view}-{hemi}"
        pos = positions[key]
        ax  = fig.add_axes(pos, projection='3d')
        ax.set_axis_off()
        ax.tick_params(left=False, right=False, labelleft=False,
                       labelbottom=False, bottom=False)

        plotting.plot_surf_stat_map(
            surf, texture,
            cmap=cmap,
            hemi=hemi,
            view=view,
            bg_on_data=True,
            bg_map=bg_map,
            colorbar=True,
            black_bg=False,
            threshold=0.001,
            darkness=0.7,
            axes=ax,
            vmax=vmax,
            symmetric_cbar=False,   # 数据已裁到非负，色标就该是 [0, vmax]，不强制对称
        )
        print(f"  rendered {key}")

plt.subplots_adjust(wspace=0, hspace=0)
fig.savefig(OUT_PDF, transparent=True, bbox_inches='tight')
# PNG（栅格）版本供正式排版用，理由同 visualize_nc_surface.py。
fig.savefig(OUT_PNG, dpi=300, transparent=True, bbox_inches='tight')
plt.close()
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")

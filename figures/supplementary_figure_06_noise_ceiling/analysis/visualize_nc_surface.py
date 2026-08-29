# Usage:
#   sbatch rebuttal/analyses/viz_nc_surface.sh          # recommended (HPC)
#   nohup python3 -u rebuttal/analyses/visualize_nc_surface.py > logs/nc_surface.log 2>&1 &
# Both must be run from the localize_paper root directory.

"""
Noise ceiling lower bound — surface plot in fig3 style.

Replicates the display_surface_plot.py aesthetic:
  - fsaverage inflated brain (infl_left / infl_right)
  - sulcal depth background (sulc_left / sulc_right)
  - lateral + medial + posterior views, both hemispheres
  - inferno colormap (designCmap style, 1200 levels)
  - white background
  - saved as PDF

Input:  SCRATCH_RESULT/rebuttal_analyses/noise_ceiling/noise_ceiling_lower_synthesized.nii.gz
Output: SCRATCH_RESULT/rebuttal_analyses/noise_ceiling/noise_ceiling_lower_surface.pdf
        (same folder as paper_figures for convenience)
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nilearn import datasets, plotting, surface

# ── paths ─────────────────────────────────────────────────────────────────────
current_dir = os.getcwd()
assert current_dir.endswith('localize_paper'), \
    f"Run from localize_paper root, not {current_dir}"

sys.path.append(current_dir)
from utils import setupPaths

_, scratchResultFolder = setupPaths()

NC_PATH = os.path.join(
    scratchResultFolder,
    "rebuttal_analyses/noise_ceiling/noise_ceiling_lower_synthesized.nii.gz"
)

PAPER_FIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "figures", "components"
)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
OUT_PDF = os.path.join(PAPER_FIG_DIR, "noise_ceiling_lower_surface.pdf")
OUT_PNG = os.path.join(PAPER_FIG_DIR, "noise_ceiling_lower_surface.png")

# ── colormap: 纯 inferno，不用 designCmap 的白色下半段 ──────────────────────────
# designCmap(白+inferno 各占一半)是给对称色标(-vmax~vmax)设计的：0 正好落在色标
# 中点，下半段的白色专门用来盖住负值。裁负值到 0 之后如果继续用这个双段色标、又让
# nilearn 按实际数据范围(0~vmax)自动重铺色标，0 会落到色标最左端而不是中点——这样
# 下半段白色就会吞掉一大截原本可见的中低正值信号，不是去掉负色标那么简单，而是会把
# 图画错。改用纯 inferno + threshold(下面已有 threshold=0.001)做近零透明，和
# fig3a_fdr.py/fig4a_fdr.py 现有的口径一致，同一份仓库不要留两套不兼容的显示逻辑。
cmap = 'inferno'

# ── load data ─────────────────────────────────────────────────────────────────
nii = nib.load(NC_PATH)
nc = nii.get_fdata()

valid = nc[np.isfinite(nc) & (nc > 0)]
vmax = float(np.percentile(valid, 99))
print(f"nc_lower: mean={valid.mean():.4f}, max={valid.max():.4f}, vmax(p99)={vmax:.4f}")

# 负值只是噪声、没有可解释的意义(noise ceiling 只在 >=0 时才是"可靠性"的度量)。
# 裁到非负后再渲染，nilearn 的默认色标范围会贴着 0 起跳，不再显示一段无意义的负半色标。
nc = np.clip(nc, 0, None)
nii = nib.Nifti1Image(nc, nii.affine, nii.header)

# ── fsaverage surface ─────────────────────────────────────────────────────────
fsaverage = datasets.fetch_surf_fsaverage()

# ── layout: mirrors wholeBrainPositions() in display_surface_plot.py ──────────
# 6 panels: lateral-L, lateral-R, medial-L, medial-R, posterior-L, posterior-R
base_w = 0.4
base_h = 0.4
positions = {
    "lateral-left":    [0.00,                    0.23854166, base_w, base_h],
    "lateral-right":   [0.20693055555555548,      0.23854166, base_w, base_h],
    "medial-left":     [0.00,                    0.00,        base_w, base_h],
    "medial-right":    [0.20693055555555548,      0.00,        base_w, base_h],
    "posterior-left":  [0.364,                   0.11,        base_w, base_h],
    "posterior-right": [0.44599999999999995,      0.11,        base_w, base_h],
}

views    = ['lateral', 'medial', 'posterior']
hemis    = ['left', 'right']

fig = plt.figure(figsize=(3 * 10, 2 * 10))
fig.patch.set_facecolor('white')
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
            threshold=0.001,   # hide near-zero (non-brain) voxels
            darkness=0.7,
            axes=ax,
            vmax=vmax,
            symmetric_cbar=False,   # 数据已裁到非负，色标就该是 [0, vmax]，不强制对称
        )
        print(f"  rendered {key}")

plt.subplots_adjust(wspace=0, hspace=0)
fig.savefig(OUT_PDF, transparent=True, bbox_inches='tight')
# PNG（栅格）版本供正式排版用：fsaverage 两个半球的三角网格全部存成矢量路径会让
# PDF 体积暴涨，供最终拼图使用的是这张高分辨率 PNG，不是上面的矢量 PDF。
fig.savefig(OUT_PNG, dpi=300, transparent=True, bbox_inches='tight')
plt.close()
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")

import os
import sys
import time
import random
import urllib.request
import subprocess
import shutil
import argparse
import warnings

import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
import matplotlib
# 强制非交互后端：必须在 import pyplot 之前。否则在带 $DISPLAY 的 login 节点上，
# matplotlib 会选交互式 GUI 后端，拼图末尾的 plt.show()/plotting.show() 会弹窗口并阻塞
# （SSH 里窗口不出现 → 永久卡住、PDF 不保存）。Agg 下 show() 变无操作，savefig 照常。
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from pprint import pprint
from os.path import join, exists, split

import pickle5 as pickle

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")
print(sys.executable)
print(sys.version)
print(sys.version_info)
print(f"numpy version = {np.__version__}")
print(f"pandas version = {pd.__version__}")

current_dir = os.getcwd()
assert current_dir.endswith(('localize_paper', 'cnn-layer-brain-mapping-figures')), f"The directory {current_dir} does not end with 'localize_paper'"
print(f"Current directory is: {current_dir}")

sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "figures/figure_06_residual_localization/analysis"))

from utils import (
    load_obj, save_obj, custom_run, waitForEnd, getjobID_num, check_jobArray, getSubs, setupPaths, check_jobIDs,
    custom_and, save_nib
)

import nilearn
import nilearn.datasets
import nilearn.plotting
import nilearn.datasets.utils

from nilearn import datasets
from nilearn import plotting
from nilearn import surface
import numpy as np
import nibabel as nib


subs = getSubs()
subjectDir, scratchResultFolder = setupPaths()
print(f"subjectDir={subjectDir}")
print(f"scratchResultFolder={scratchResultFolder}")

import nibabel as nib


def designCmap(cmap='rainbow', colors=12):
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    if cmap in ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']:
        viridis = cm.get_cmap(cmap, colors + 1)
        newColors = viridis(np.linspace(0, 1, 2*colors + 1))
        white = np.array([1, 1, 1, 1])
        newColors_ = np.zeros((2 * colors + 1, 4))
        newColors_ = newColors
        newColors_[0:(colors + 1), :] = white
        newColors_[(colors + 1):(2 * colors + 1), :] = newColors[colors:-1]
    elif cmap == "gnuplot2":
        viridis = cm.get_cmap(cmap, colors + 1)
        newColors = viridis(np.linspace(0, 1, colors + 1))
        white = np.array([1, 1, 1, 1])
        newColors_ = np.zeros((2 * colors + 1, 4))
        newColors_[0:(colors + 1), :] = white
        newColors_[(colors + 1):(2 * colors + 1), :] = newColors[:-1]
    else:
        viridis = cm.get_cmap(cmap, colors)
        newColors = viridis(np.linspace(0, 1, colors))
        white = np.array([1, 1, 1, 1])
        newColors_ = np.zeros((2 * colors + 1, 4))
        newColors_[0:(colors + 1), :] = white
        newColors_[(colors + 1):(2 * colors + 1), :] = newColors
    newCmap = ListedColormap(newColors_)
    return newCmap




def plot_surface_repStabilityMask_pdf(resultType=None,
                                      views=None,
                                      cmap=None,
                                      vmax=None,
                                      cmapName=None,
                                      figureName=""):
    from nilearn import datasets
    from nilearn import plotting
    from nilearn import surface
    import numpy as np
    import nibabel as nib
    fsaverage = datasets.fetch_surf_fsaverage()
    layers = np.arange(1, 13)
    if resultType == "repStabilityMask-synthesized":
        file_path = f"{scratchResultFolder}/" \
                    f"repStability_RSA_sl" \
                    f"/rad3/__tfce_corrp_tstat1.nii.gz"
        nii = nib.load(file_path)
        niiData = nii.get_fdata()
    elif resultType == "repStabilityMask-NSD":
        file_path = f"{scratchResultFolder}/nsd_result_folder/result/repStability_RSA_sl/" \
                    f"/randomise/rad3/allSet//__tfce_corrp_tstat1.nii.gz"
        nii = nib.load(file_path)
        niiData = nii.get_fdata()
    elif resultType == "repStabilityMask-synthesized-tfce-masked-tstat":
        tfce_path = \
            f"{scratchResultFolder}/" \
            f"repStability_RSA_sl" \
            f"/rad3/__tfce_corrp_tstat1.nii.gz"
        tfce = nib.load(tfce_path).get_fdata()
        tfce = tfce > 0.95
        affine = nib.load(tfce_path).affine
        tstat_path = f"{scratchResultFolder}/" \
                     f"repStability_RSA_sl" \
                     f"/rad3/__tstat1.nii.gz"
        tstat = nib.load(tstat_path).get_fdata()
        niiData = tfce * tstat
        nii = nib.Nifti1Image(niiData.astype('double'), affine)
    elif resultType == "repStabilityMask-nsd-tfce-masked-tstat":
        tfce_path = \
            f"{scratchResultFolder}/nsd_result_folder/result/repStability_RSA_sl/" \
            f"/randomise/rad3/allSet/__tfce_corrp_tstat1.nii.gz"
        tfce = nib.load(tfce_path).get_fdata()
        tfce = tfce > 0.95
        affine = nib.load(tfce_path).affine
        tstat_path = f"{scratchResultFolder}/nsd_result_folder/result/repStability_RSA_sl/" \
                     f"/randomise/rad3/allSet/__tstat1.nii.gz"
        tstat = nib.load(tstat_path).get_fdata()
        niiData = tfce * tstat
        nii = nib.Nifti1Image(niiData.astype('double'), affine)
    else:
        raise Exception("resultType not defined")
    leftHemi = niiData[46:-1, :, :]
    leftHemiNumber = np.sum(leftHemi)
    rightHemi = niiData[0:46, :, :]
    rightHemiNumber = np.sum(rightHemi)

    nonZeroVoxelNumber = np.sum(niiData != 0)
    maxNiiValue = np.max(niiData)
    hemispheres = ['left', 'right']

    def wholeBrainPositions():
        base_width = 0.4
        base_height = 0.4
        positions = {
            "lateral-left": [0.00,
                             0.23854166666666665,
                             base_width, base_height],
            "lateral-right": [0.20693055555555548,
                              0.23854166666666665,
                              base_width, base_height],

            "medial-left": [0.00,
                            0.00,
                            base_width, base_height],
            "medial-right": [0.20693055555555548,
                             0.00,
                             base_width, base_height],

            "posterior-left": [0.364, 0.10999999999999999,
                               base_width, base_height],
            "posterior-right": [0.44599999999999995,
                                0.10999999999999999,
                                base_width, base_height],
        }
        return positions

    positions = wholeBrainPositions()
    fig, axes = plt.subplots(figsize=(3 * 10, 2 * 10))
    plt.axis('off')

    threshold = None
    darkness = None
    for hemi in hemispheres:
        if hemi == 'left':
            texture = surface.vol_to_surf(nii, fsaverage.pial_left, interpolation='nearest')
            bg_map = fsaverage.sulc_left
            surf = fsaverage.infl_left
        else:
            texture = surface.vol_to_surf(nii, fsaverage.pial_right, interpolation='nearest')
            bg_map = fsaverage.sulc_right
            surf = fsaverage.infl_right
        for view in views:
            pos = positions[f"{view}-{hemi}"]
            ax = fig.add_axes(pos, projection='3d')
            ax.set_axis_off()
            ax.tick_params(left=False, right=False, labelleft=False,
                           labelbottom=False, bottom=False)

            surf_map = plotting.plot_surf_stat_map(surf, texture, cmap=cmap,
                                                   hemi=hemi, view=view, bg_on_data=True, bg_map=bg_map,
                                                   bg_img='MNI152',
                                                   colorbar=True,
                                                   black_bg=False,
                                                   threshold=threshold,
                                                   darkness=darkness,
                                                   axes=ax,
                                                   vmax=vmax,
                                                   )
            print(f"resultType-{resultType}: {view} {hemi}-hemi "
                  f"voxelNumber-left{leftHemiNumber}-right{rightHemiNumber} "
                  f"maxValue-{maxNiiValue}")
    plt.subplots_adjust(wspace=0, hspace=0)
    savePath = f"{scratchResultFolder}/figures/{figureName}{resultType}-vmax{vmax}.pdf"
    if cmapName is not None:
        savePath = savePath.replace(".pdf", f"-{cmapName}.pdf")
    savePath = savePath.replace(" ", "_")
    savePath = savePath.replace(":", "_")
    fig.savefig(savePath, transparent=True)
    plotting.show()
    print(f"resultType={resultType}, views={views}")
    print(f"saved to {savePath}")


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg 邻接 p（q-value）。与 diag_outlined_fdr.py:fdr_bh 同款、
    本地对 statsmodels / BH 阈值定义逐一致验过。"""
    pvals = np.asarray(pvals, float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.minimum(1.0, pvals[order] * n / (np.arange(1, n + 1)))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = adj
    return out


def fdr_significant_mask(tstat, in_mask, df, q=0.05):
    """方案 B1：在 in_mask 内把 t → 单尾参数 p → BH-FDR，返回二值显著图（掩膜外恒 0）+ 诊断。

    数学与本地蒙特卡洛（N=30 实测平均 FDR≈0.048≤0.05）见 20_/21_ 说明与 docs/adr。
    fail-loud：形状不符 / df<1 / 掩膜空 / 掩膜内 t 非有限，一律 raise，绝不静默跳过或回退。"""
    from scipy.stats import t as _t_dist
    tstat = np.asarray(tstat, float)
    in_mask = np.asarray(in_mask)
    if tstat.shape != in_mask.shape:
        raise ValueError(f"[门禁 FDR] tstat 形状 {tstat.shape} 与稳定性掩膜 {in_mask.shape} 不一致")
    if df < 1:
        raise ValueError(f"[门禁 FDR] df={df} 非法（应为实际被试数 N-1，需 >=1）")
    idx = in_mask.astype(bool)
    m = int(idx.sum())
    if m == 0:
        raise ValueError("[门禁 FDR] 稳定性掩膜为空，没有可检验的体素")
    tv = tstat[idx]
    if not np.all(np.isfinite(tv)):
        raise ValueError(f"[门禁 FDR] 掩膜内有 {int((~np.isfinite(tv)).sum())} 个非有限 t 值")
    p = _t_dist.sf(tv, df)                       # 单尾右：P(T>=t)=1-CDF，检验均值>0
    q_adj = fdr_bh(p, alpha=q)
    sig_vox = (q_adj < q).astype(float)
    sig = np.zeros(in_mask.shape, float)
    sig[idx] = sig_vox
    return sig, {'m': m, 'df': df, 'n_sig': int(sig_vox.sum())}


def _nsd_meanacross_tstat_and_mask(scratchResultFolder, layer):
    """fig4(NSD 非残差) meanAcrossSets 的单样本 t 图 + NSD 稳定性掩膜——从现存的逐-set 逐被试图廉价重建。

    背景：原 meanAcrossSets randomise 输出已被 scratch60 清理，但每个 set 的 subs_compiled
    （该 set 下 N 个被试的图，4D）还在。原流水线做的就是"逐被试跨 16 set 求均值 → 组 randomise"，
    而 randomise 的 __tstat1 就是这些均值图的单样本 t（确定量，置换只影响 corrp）。FDR 只需 t，
    故直接重算该单样本 t，与原 meanAcrossSets t 图等价，避免重跑昂贵的 searchlight/randomise。
    fail-loud：找不到 set 目录 / 缺 subs_compiled / 非 4D / 各 set 形状不一致 / N<2 / 掩膜缺失或形状不符，一律 raise。
    返回 (tstat, N, affine, in_mask)。"""
    import glob as _glob
    import nibabel as nib
    base = (f"{scratchResultFolder}"
            f"nsd/findLayerInBrain_leaveOneSubOutReg/randomise/layer{layer}/reg/rad3/")
    set_dirs = sorted(_glob.glob(base + "whichSet*/"))
    if len(set_dirs) == 0:
        raise FileNotFoundError(f"[门禁 FDR] NSD layer{layer} 找不到任何 whichSet 目录: {base}")
    if len(set_dirs) != 16:
        # NSD 固定 16 个图片 set（实验设计参数，非分析选项）。数量不对不静默地对着不完整的
        # set 求均值——那样算出的 t 看起来完全正常、不会报错，但已经是基于不完整数据的了。
        raise ValueError(f"[门禁 FDR] NSD layer{layer} 期望 16 个 set，实际找到 {len(set_dirs)} 个: "
                         f"{[os.path.basename(d.rstrip('/')) for d in set_dirs]}")
    acc = None
    ref_shape = None
    affine = None
    for sd in set_dirs:                              # 逐 set 累加，省内存（不 stack 全部）
        sc = sd + "subs_compiled.nii.gz"
        if not os.path.exists(sc):
            raise FileNotFoundError(f"[门禁 FDR] NSD layer{layer} 缺 per-set 逐被试图: {sc}")
        nii = nib.load(sc)
        if nii.ndim != 4:
            raise ValueError(f"[门禁 FDR] NSD {sc} 不是 4D(实得 {nii.ndim}D)")
        data = nii.get_fdata()
        if ref_shape is None:
            ref_shape = data.shape
            affine = nii.affine
            acc = np.zeros(ref_shape, float)
        elif data.shape != ref_shape:
            raise ValueError(f"[门禁 FDR] NSD layer{layer} 各 set subs_compiled 形状不一致: "
                             f"{data.shape} vs {ref_shape}（{sc}）")
        acc += data
    per_subj_mean = acc / len(set_dirs)              # 逐被试跨 set 均值，(X,Y,Z,N)
    N = int(per_subj_mean.shape[3])
    if N != 8:
        # NSD 固定 8 个被试；每个 set 的 subs_compiled 由 fslmerge 拼接现存的逐被试文件
        # (Randomise.sh 里 `if [ -f "$output" ]` 会静默跳过缺失文件)，若某被试的文件缺失，
        # 拼出来的 4D 会悄悄少一个被试、N 照样 >=2 通过，但已经不是完整队列了。
        raise ValueError(f"[门禁 FDR] NSD layer{layer} 期望 8 个被试，subs_compiled 第 4 维实际 N={N}")
    m = per_subj_mean.mean(axis=3)
    s = per_subj_mean.std(axis=3, ddof=1)            # 样本标准差(ddof=1)，同 FSL 单样本 t
    with np.errstate(divide='ignore', invalid='ignore'):
        tstat = np.where(s > 0, m / (s / np.sqrt(N)), 0.0)   # 单样本 t；std=0 处置 0，避免 0/0
    # NSD 稳定性掩膜：全域搜索坐实在 project 永久盘（不在 scratch）。
    mask_path = ("/gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/"
                 "nsd/result/repStability_RSA_sl/randomise/rad3/allSet/__tfce_corrp_tstat1.nii.gz")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"[门禁 FDR] NSD 稳定性掩膜缺失: {mask_path}")
    mask_nii = nib.load(mask_path)
    stab = mask_nii.get_fdata()
    if stab.shape != tstat.shape:
        raise ValueError(f"[门禁 FDR] NSD 稳定性掩膜 {stab.shape} 与 t 图 {tstat.shape} 形状不符")
    in_mask = (stab > 0.95).astype(float)
    # affine 订正（关键 bug 修复）：subs_compiled/per-set 的 affine 是损坏的——"网格居中 + 1mm"
    #   默认值(实测 diag≈(-1,1,1)、origin=网格中心)，把本是 MNI 2mm 的内容谎报成 1mm。直接拿它
    #   vol_to_surf 会把脑压到半尺度、外侧皮层信号错位挤向内侧，正是新 fig4"只剩内侧"的真因。
    #   稳定性掩膜文件带正确的 MNI 2mm affine(diag≈(-2,2,2)、origin(90,-126,-72))，且与数据逐体素
    #   对齐(同 91×109×91 网格、非零体素 index bbox 一致)，故输出改用掩膜的 affine，渲染才空间正确。
    #   fail-loud：掩膜体素尺寸若不是 ~2mm，说明它也异常，宁可 raise 也不拿可疑 affine 蒙混。
    mask_vox = np.abs(np.diag(mask_nii.affine)[:3])
    if not np.allclose(mask_vox, 2.0, atol=0.1):
        raise ValueError(f"[门禁 FDR] NSD 稳定性掩膜体素尺寸 {mask_vox} 非 ~2mm，"
                         f"不敢用它订正 affine，请人工核查空间")
    if not np.allclose(mask_nii.affine, affine):
        print(f"[NSD重建] layer{layer}: 检出 subs_compiled affine 损坏(居中-1mm)，"
              f"已改用稳定性掩膜的正确 MNI 2mm affine 渲染")
    affine = mask_nii.affine
    print(f"[NSD重建] layer{layer}: {len(set_dirs)} 个 set → 逐被试均值 → 单样本 t, N={N}")
    return tstat, N, affine, in_mask


def _nsd_residual_meanacross_tstat_and_mask(scratchResultFolder, layer):
    """fig6(NSD 残差模型) meanAcrossSets 的单样本 t 图 + NSD 稳定性掩膜——与 `_nsd_meanacross_tstat_and_mask`
    完全同构，唯一差别是 base path(非残差=findLayerInBrain_leaveOneSubOutReg，残差=residualModel)。

    这里踩过一次坑，记录一下：一开始假设残差模型有预先算好的"逐被试跨 16 set 均值"文件
    (nsd/residualModel/meanAcrossSets/{sub}/...)，实测该目录在集群 scratch 上根本不存在——
    不是被清理，是这一步分析从没跑过(上游 randomise/searchlight_score_linearModel 都在，唯独
    meanAcrossSets 这一级连目录都没有)。改为直接从仍然存活的逐-set subs_compiled 重建：
    nsd/residualModel/randomise/layer{layer}/reg/rad3/whichSet{N}/subs_compiled.nii.gz(已在
    集群上核实存在，16 个 set、每个 set 下 8 个被试打包的 4D 图)。逐被试跨 16 set 求均值，
    再算单样本 t，与原 meanAcrossSets randomise 的 t 等价，避免重跑昂贵的 searchlight/randomise。

    稳定性掩膜与非残差共用同一张(repStability_RSA_sl 与残差/非残差无关，只是同一份 searchlight
    稳定性检验)，沿用 `_nsd_meanacross_tstat_and_mask` 里验证过的 project 永久盘路径。
    affine 同样一律用稳定性掩膜的(同 fig4 的 affine 修复教训：不信任派生数据文件自带的 affine，
    宁可统一改用已验证正确的 2mm 掩膜 affine)。
    fail-loud：找不到 set 目录 / 缺 subs_compiled / 非 4D / 各 set 形状不一致 / N<2 / 掩膜缺失
    或形状不符 / 掩膜非~2mm，一律 raise。
    返回 (tstat, N, affine, in_mask)。"""
    import glob as _glob
    import nibabel as nib
    base = (f"{scratchResultFolder}"
            f"nsd/residualModel/randomise/layer{layer}/reg/rad3/")
    set_dirs = sorted(_glob.glob(base + "whichSet*/"))
    if len(set_dirs) == 0:
        raise FileNotFoundError(f"[门禁 FDR] residual layer{layer} 找不到任何 whichSet 目录: {base}")
    if len(set_dirs) != 16:
        # 同 _nsd_meanacross_tstat_and_mask：NSD 固定 16 个 set，数量不对不静默地对着
        # 不完整的 set 求均值。
        raise ValueError(f"[门禁 FDR] residual layer{layer} 期望 16 个 set，实际找到 {len(set_dirs)} 个: "
                         f"{[os.path.basename(d.rstrip('/')) for d in set_dirs]}")
    acc = None
    ref_shape = None
    for sd in set_dirs:                              # 逐 set 累加，省内存（不 stack 全部）
        sc = sd + "subs_compiled.nii.gz"
        if not os.path.exists(sc):
            raise FileNotFoundError(f"[门禁 FDR] residual layer{layer} 缺 per-set 逐被试图: {sc}")
        nii = nib.load(sc)
        if nii.ndim != 4:
            raise ValueError(f"[门禁 FDR] residual {sc} 不是 4D(实得 {nii.ndim}D)")
        data = nii.get_fdata()
        if not np.all(np.isfinite(data)):
            raise ValueError(
                f"[门禁 FDR] residual {sc} 含 "
                f"{int(np.sum(~np.isfinite(data)))} 个非有限值"
            )
        if ref_shape is None:
            ref_shape = data.shape
            acc = np.zeros(ref_shape, float)
        elif data.shape != ref_shape:
            raise ValueError(f"[门禁 FDR] residual layer{layer} 各 set subs_compiled 形状不一致: "
                             f"{data.shape} vs {ref_shape}（{sc}）")
        acc += data
    per_subj_mean = acc / len(set_dirs)              # 逐被试跨 set 均值，(X,Y,Z,N)
    N = int(per_subj_mean.shape[3])
    if N != 8:
        # 同 _nsd_meanacross_tstat_and_mask：NSD 固定 8 个被试，不静默接受少数被试拼漏的情况。
        raise ValueError(f"[门禁 FDR] residual layer{layer} 期望 8 个被试，subs_compiled 第 4 维实际 N={N}")
    m = per_subj_mean.mean(axis=3)
    s = per_subj_mean.std(axis=3, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        tstat = np.where(s > 0, m / (s / np.sqrt(N)), 0.0)
    mask_path = ("/gpfs/milgram/project/turk-browne/projects/localize/analysis/organizedScripts/"
                 "nsd/result/repStability_RSA_sl/randomise/rad3/allSet/__tfce_corrp_tstat1.nii.gz")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"[门禁 FDR] residual 稳定性掩膜缺失: {mask_path}")
    mask_nii = nib.load(mask_path)
    stab = mask_nii.get_fdata()
    if stab.shape != tstat.shape:
        raise ValueError(f"[门禁 FDR] residual 稳定性掩膜 {stab.shape} 与 t 图 {tstat.shape} 形状不符")
    in_mask = (stab > 0.95).astype(float)
    if not np.any(in_mask):
        raise ValueError(f"[门禁 FDR] residual layer{layer} 稳定性掩膜为空")
    if not np.all(np.isfinite(tstat)):
        raise ValueError(f"[门禁 FDR] residual layer{layer} 重建 t 图含非有限值")
    mask_vox = np.abs(np.diag(mask_nii.affine)[:3])
    if not np.allclose(mask_vox, 2.0, atol=0.1):
        raise ValueError(f"[门禁 FDR] residual 稳定性掩膜体素尺寸 {mask_vox} 非 ~2mm，"
                         f"不敢用它订正 affine，请人工核查空间")
    affine = mask_nii.affine
    print(f"[residual重建] layer{layer}: {len(set_dirs)} 个 set → 逐被试均值 → 单样本 t, N={N}"
          f"(已用稳定性掩膜的正确 2mm affine)")
    return tstat, N, affine, in_mask


# 渲染缓存版本号：改了重建/渲染逻辑(如 affine 订正)就 +1，令下面 _img_path 生成的所有旧 PNG
# 缓存名作废、强制重渲。这是防"装配阶段静默读旧缓存出坏图"的门禁——affine 是数据派生、装配前
# 算不出无法入缓存名，故用这个手动版本号兜住渲染逻辑变更。v2 = NSD affine 居中-1mm→MNI 2mm 订正后。
_RENDER_CACHE_VERSION = "v2-affine2mm"


def plot_surface_saveImageReload_pdf(resultType=None,
                                     views=None, h_pad_=-70,
                                     w_pad_=-2, cmap=None, colorBar=False, vmax=None, vmin=None, colorBarColorNum=None,
                                     thresholdName=None,
                                     forceRun=False,
                                     figureName="",
                                     layers=None,
                                     renderOnly=False,
                                     copyTo=None,
                                     displayThreshold=None,
                                     bgDarkness=None,
                                     contourMaskPathFn=None,
                                     colorBarVmin=None,
                                     colorBarLabel=None,
                                     assembleOnly=False,
                                     colorBarTickFormat='%.2f'):
    # displayThreshold: 传给 nilearn 的显示阈值。FDR 图 niiData=sig×t 已是 FDR 阈过的(0=不显著)，
    #   传一个极小正值(如 0.01)+纯 inferno 色标，让 niiData=0 透明、所有显著体素(含低 t)都上色，
    #   避免 designCmap 白下半段(≈vmax/2 的隐含阈值)把广泛的中低 t 显著区藏成白色。
    # layers: 只渲染这些层（并行时每个 job 传单层，如 [5]）；None=全 12 层（默认，行为不变）。
    # renderOnly: True 时渲完各层 PNG 即返回、跳过拼图（并行 render 阶段用；拼图留给最后一个 job）。
    # assembleOnly: True 时只允许读取完整且可解码的 PNG 缓存并拼图；任何缓存缺失/损坏立即
    #   raise，绝不静默回退成串行重渲染。只用于 array 全部完成后的严格装配阶段。
    # copyTo: 拼图存到 scratch 后，再拷一份到这个（仓库内、git 跟踪的）路径，避免仓库副本变旧。
    #         renderOnly 提前返回、不触发拷贝；只有装配/串行全跑会拷。
    # bgDarkness: 传给 nilearn 正常渲染分支的 darkness（背景沟回明暗）。None=旧行为(默认较暗)，
    #   不影响任何现有调用方。纯 inferno + threshold 的 FDR 图里，不显著体素透明、露出较暗的解剖
    #   底，非空层因此偏暗；而空层走的 plot_empty_brain_ 分支已固定 darkness=0.3(偏浅)，两者撞出
    #   "空层浅、非空层暗"的不一致。fig3a_fdr/fig4a_fdr 传 bgDarkness=0.3 让正常分支也调到 0.3，
    #   与空层分支一致，整体回到老图那种干净的浅背景，又保留纯 inferno 让中等 t 可见。
    # contourMaskPathFn: 可选，callable(layer:int) -> path|None。给定时，对该层额外读取这个
    #   NIfTI(与 niiData 同一体素空间/affine 的二值显著性掩膜)，投到同一张皮层表面上，在底图上
    #   叠画黑色显著性轮廓(nilearn plot_surf_contours)。None(默认)=不叠轮廓，行为与旧版完全一致，
    #   不影响 fig3a/fig3b/fig4/fig6ab 等现有调用方。用于 rebuttal "无阈值底图 + FDR 黑框" 图。
    if cmap is None:
        cmap = 'bwr'
    if assembleOnly and (forceRun or renderOnly):
        raise ValueError("assembleOnly=True cannot be combined with forceRun or renderOnly")

    def remove_whiteBorder(array):
        array_grey = np.mean(array, axis=2)
        b1 = np.where(np.mean(array_grey, axis=0) != 255)[0][0]
        b2 = np.where(np.mean(array_grey, axis=0) != 255)[0][-1]
        b3 = np.where(np.mean(array_grey, axis=1) != 255)[0][0]
        b4 = np.where(np.mean(array_grey, axis=1) != 255)[0][-1]
        return array[b3:b4, b1:b2, :]

    def cleanFolder(folder_path):
        if not os.listdir(folder_path):
            print('Folder is empty')
        else:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    if colorBarColorNum is not None:
        savePath = f"{scratchResultFolder}figures/{figureName}{resultType}-{views[0]}-vmax{vmax}-colorBarColorNum{colorBarColorNum}.pdf"
    else:
        savePath = f"{scratchResultFolder}/figures/{figureName}{resultType}-{views[0]}-vmax{vmax}.pdf"
    savePath = savePath.replace(" ", "_")
    savePath = savePath.replace(":", "_")
    # figures/ 输出目录必须先存在：colorbar_pdf_path 与最终拼图都写这里，scratch60 被清理后
    # 该目录会消失，缺失即在 plt.savefig 处 FileNotFoundError。显式补齐，同 temp_arrangementFolder。
    os.makedirs(os.path.dirname(savePath), exist_ok=True)


    temp_arrangementFolder = f"{scratchResultFolder}/temp/temp_arrangement/"
    os.makedirs(temp_arrangementFolder, exist_ok=True)
    fsaverage = datasets.fetch_surf_fsaverage()
    hemispheres = ['left', 'right']
    # 缓存门禁：PNG 缓存名带"渲染版本号 + 关键渲染参数"，代码/参数一变即 cache-miss、强制重渲，
    #   杜绝装配阶段静默读旧缓存出坏图(affine 修复、vmax/背景/阈值变更等曾因缓存名不含参数漏读旧图)。
    #   三处路径(装配加速守卫 / render / assemble)全走这一个 helper，杜绝多处漂移又生新坑。
    def _sanitize(s):
        for a, b in ((" ", "_"), (":", "_"), ("/", "_"), ("(", ""), (")", "")):
            s = str(s).replace(a, b)
        return s
    # contourMaskPathFn=None(全部现有调用方)时这里不加任何后缀，_param_tag 与旧版逐字节相同，
    # 不会让 fig3a/fig3b/fig4/fig6ab 已有的 PNG 缓存失效；只有传了 contourMaskPathFn 才加标记，
    # 确保"叠轮廓"和"不叠轮廓"两版不会互相读错缓存。
    _contour_tag = "-contourTrue" if contourMaskPathFn is not None else ""
    _param_tag = (f"{_RENDER_CACHE_VERSION}-fn{_sanitize(figureName)}-vmax{vmax}"
                  f"-bg{bgDarkness}-th{displayThreshold}{_contour_tag}")

    def _img_path(_layer, _view, _hemi):
        return (f"{temp_arrangementFolder}/resultType-{_sanitize(resultType)}"
                f"-{_param_tag}-layer-{_layer}-{_view}-{_hemi}-hemi.png")
    valueRange = {}
    if layers is None:
        layers = np.arange(1, 13)
    layers = list(layers)
    if assembleOnly:
        from PIL import Image as _CacheImage
        cache_issues = []
        for layer in layers:
            for hemi in hemispheres:
                for view in views:
                    cache_path = _img_path(layer, view, hemi)
                    if not os.path.exists(cache_path):
                        cache_issues.append(f"missing: {cache_path}")
                        continue
                    if os.path.getsize(cache_path) < 1024:
                        cache_issues.append(f"too small/corrupt: {cache_path}")
                        continue
                    try:
                        with _CacheImage.open(cache_path) as cached_image:
                            cached_image.verify()
                    except Exception as exc:
                        cache_issues.append(f"unreadable: {cache_path} ({exc})")
        if cache_issues:
            details = "\n  ".join(cache_issues)
            raise FileNotFoundError(
                "Strict assembly refused: layer render cache is incomplete or corrupt:\n  "
                + details
            )
        print(f"[assembleOnly] validated {len(layers) * len(hemispheres) * len(views)} PNG caches")
    for layer in layers:
        # 装配加速：forceRun=False(装配/复用阶段)且本层所需的每个 (hemi×view) PNG 都已缓存，
        #   就跳过整层——它下面的重建(某些分支昂贵，如 NSD 每层 load 16 个 set)与 vol_to_surf
        #   产物只用于渲染，本层既已缓存无需重算。缓存不全或 forceRun=True 时照常回落，行为不变。
        if not forceRun:
            _cached = all(
                os.path.exists(_img_path(layer, view, hemi))
                for hemi in hemispheres for view in views)
            if _cached:
                print(f"layer-{layer}: 所需 PNG 全部已缓存，跳过重建/渲染(装配加速)")
                continue
        niiData = None
        nii = None
        if resultType == "synthesized":
            file_path = f"{scratchResultFolder}" \
                        f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/layer{layer}/" \
                        f"reg/rad3/repStabilityMasked__tfce__layer{layer}.nii.gz"
            nii = nib.load(file_path)
            niiData = nii.get_fdata()
        elif resultType == "NSD nonresidual":
            file_path = f"{scratchResultFolder}" \
                        f"nsd/findLayerInBrain_leaveOneSubOutReg/randomise//layer{layer}/" \
                        f"reg/rad3/allSets/allSetsSummed_RepStabilityMasked_layer{layer}.nii.gz"
            nii = nib.load(file_path)
            desired_affine = nib.load(
                "/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz").affine
            nii = nib.Nifti1Image(nii.get_fdata(), desired_affine)
            print(f"nii.affine={nii.affine}")
            niiData = nii.get_fdata()
        elif resultType == "NSD residual":
            file_path = f"{scratchResultFolder}" \
                        f"nsd/residualModel/allSetsSum/allSetsSum_layer{layer}.nii.gz"
            nii = nib.load(file_path)
            desired_affine = nib.load(
                "/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz").affine
            nii = nib.Nifti1Image(nii.get_fdata(), desired_affine)
            print(f"nii.affine={nii.affine}")
            niiData = nii.get_fdata()
        elif resultType == "NSD nonresidual mean+randomise":
            file_path = f"{scratchResultFolder}" \
                        f"nsd/findLayerInBrain_leaveOneSubOutReg/meanAcrossSets/randomise/" \
                        f"rad3/reg/layer{layer}/repStabilityMasked_layer{layer}__tfce_corrp_tstat1.nii.gz"
            nii = nib.load(file_path)
            niiData = nii.get_fdata()
        elif resultType == "NSD residual mean+randomise":
            file_path = f"{scratchResultFolder}" \
                        f"nsd/residualModel/meanAcrossSets/randomise/" \
                        f"rad3/reg/layer{layer}/repStabilityMasked_layer{layer}__tfce_corrp_tstat1.nii.gz"
            nii = nib.load(file_path)
            niiData = nii.get_fdata()
        elif resultType == "synthesized: repStabilityMasked TFCE masked tstat":
            repStabilityMasked_tfce_path = \
                f"{scratchResultFolder}" \
                f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/layer{layer}/" \
                f"reg/rad3/repStabilityMasked__tfce__layer{layer}.nii.gz"
            repStabilityMasked_tfce = nib.load(repStabilityMasked_tfce_path).get_fdata()
            affine = nib.load(repStabilityMasked_tfce_path).affine
            tstat_path = f"{scratchResultFolder}" \
                         f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/layer{layer}/" \
                         f"reg/rad3/__tstat1.nii.gz"
            tstat = nib.load(tstat_path).get_fdata()
            niiData = repStabilityMasked_tfce * tstat
            nii = nib.Nifti1Image(niiData.astype('double'), affine)

            repStabilityMasked_tstat_path = (f"{scratchResultFolder}/findLayerInBrain_leaveOneSubOutReg/shuffleNorm/"
                                             f"randomise/layer{layer}/reg/rad3/repStabilityMasked_layer{layer}_tstat1.nii.gz")
            save_nib(niiData, repStabilityMasked_tstat_path, affine_mat=affine)

        elif resultType == "synthesized: repStabilityMasked FDR masked tstat":
            # 方案 B1（参数 t→p → BH-FDR）：照搬上面 synthesized 的 TFCE 分支，只把"TFCE 显著"
            # 换成"FDR 显著"。统计量(t)、掩膜(稳定性)、逐层各自校正全不变，见 20_/21_ 与 docs/adr。
            base = (f"{scratchResultFolder}"
                    f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/layer{layer}/reg/rad3/")
            tstat_path = base + "__tstat1.nii.gz"
            subs_path = base + "subs_compiled.nii.gz"
            # 稳定性掩膜：与 TFCE 主图同一张，取自主图作图代码 fig3a/visualize/visualize.py:73-76——
            # repStability_RSA_sl 的 TFCE 校正结果，>0.95（即校正 p<0.05）二值化。固定路径、
            # 12 层共用、无 thresholdName（thresholdName 是 Sfig4 簇质量敏感性分析专用，与主图无关）。
            mask_path = f"{scratchResultFolder}repStability_RSA_sl/rad3/__tfce_corrp_tstat1.nii.gz"
            for pth, what in [(tstat_path, "t 图"), (subs_path, "合并被试体积(定 N/df)"),
                              (mask_path, "稳定性掩膜")]:
                if not os.path.exists(pth):
                    raise FileNotFoundError(f"[门禁 FDR] layer{layer} 缺输入 {what}: {pth}")
            tstat_nii = nib.load(tstat_path)
            affine = tstat_nii.affine
            tstat = tstat_nii.get_fdata()
            subs_nii = nib.load(subs_path)
            if subs_nii.ndim != 4:
                raise ValueError(f"[门禁 FDR] subs_compiled 不是 4D(实得 {subs_nii.ndim}D)，无法定 N")
            N = int(subs_nii.shape[3])
            if N < 2:
                raise ValueError(f"[门禁 FDR] 合并被试数 N={N} 过小(需 >=2)")
            df = N - 1
            stab = nib.load(mask_path).get_fdata()
            if stab.shape != tstat.shape:
                raise ValueError(f"[门禁 FDR] 稳定性掩膜 {stab.shape} 与 t 图 {tstat.shape} 形状不符")
            in_mask = (stab > 0.95).astype(float)    # 稳定性显著(TFCE corrp>0.95 即校正 p<0.05)=1，同 visualize.py
            sig, info = fdr_significant_mask(tstat, in_mask, df=df, q=0.05)
            print(f"[FDR] layer{layer}: N={N} df={df} mask内={info['m']}体素 FDR显著={info['n_sig']}体素")
            niiData = sig * tstat
            nii = nib.Nifti1Image(niiData.astype('double'), affine)

        elif resultType == "NSD nonresidual mean+randomise: repStabilityMasked TFCE masked tstat":
            repStabilityMasked_tfce_path = \
                f"{scratchResultFolder}" \
                f"nsd/findLayerInBrain_leaveOneSubOutReg/meanAcrossSets/randomise/" \
                f"rad3/reg/layer{layer}/repStabilityMasked_layer{layer}__tfce_corrp_tstat1.nii.gz"
            repStabilityMasked_tfce = nib.load(repStabilityMasked_tfce_path).get_fdata()
            tstat_path = f"{scratchResultFolder}" \
                         f"nsd/findLayerInBrain_leaveOneSubOutReg/meanAcrossSets/randomise/" \
                         f"rad3/reg/layer{layer}/layer{layer}__tstat1.nii.gz"
            affine = nib.load(repStabilityMasked_tfce_path).affine
            tstat = nib.load(tstat_path).get_fdata()
            niiData = repStabilityMasked_tfce * tstat
            nii = nib.Nifti1Image(niiData.astype('double'), affine)
        elif resultType == "NSD nonresidual mean+randomise: repStabilityMasked FDR masked tstat":
            # fig4A（NSD 自然图 panel A）：原 meanAcrossSets randomise 输出已被 scratch 清理，
            # 但每个 set 的逐被试图(subs_compiled)还在，据此廉价重建 meanAcrossSets 的单样本 t
            # （与原流水线等价，详见 _nsd_meanacross_tstat_and_mask）。稳定性掩膜取 project 永久盘。
            tstat, N, affine, in_mask = _nsd_meanacross_tstat_and_mask(scratchResultFolder, layer)
            df = N - 1
            sig, info = fdr_significant_mask(tstat, in_mask, df=df, q=0.05)
            print(f"[FDR NSD] layer{layer}: N={N} df={df} mask内={info['m']}体素 FDR显著={info['n_sig']}体素")
            niiData = sig * tstat
            nii = nib.Nifti1Image(niiData.astype('double'), affine)
        elif resultType == "NSD nonresidual mean+randomise: repStabilityMasked RAW tstat (no sig)":
            # 诊断用(fig4 外侧信号复核)：与上面 FDR 分支同一份重建的单样本 t、同一张稳定性掩膜，
            # 唯一差别是不乘 FDR 显著掩膜(sig)而乘稳定性掩膜(in_mask)——只在稳定性掩膜内画原始 t，
            # 不加任何显著门限。用来判"外侧(lateral)到底有没有信号、有多强"：
            #   原始 t 外侧就空 → 数据本无外侧，老图外侧是显示假象(H3)；
            #   外侧 t 中等却被 FDR 砍掉 → 数据有、只是没过 FDR 阈值(H2，阈值严格度问题)。
            tstat, N, affine, in_mask = _nsd_meanacross_tstat_and_mask(scratchResultFolder, layer)
            # fail-loud，与 FDR 分支(靠 fdr_significant_mask 的 m==0 / 非有限 raise)对齐：
            # 掩膜为空 → 别静默画全零图；掩膜内有非有限 t → 别静默带进渲染。
            m = int(in_mask.sum())
            if m == 0:
                raise ValueError(f"[门禁 RAW] NSD layer{layer} 稳定性掩膜为空，没有可画的体素")
            if not np.all(np.isfinite(tstat[in_mask.astype(bool)])):
                raise ValueError(f"[门禁 RAW] NSD layer{layer} 掩膜内有非有限 t 值(重建异常)")
            niiData = in_mask * tstat
            # 注：RAW 是带符号 t(未阈值、含负值/去激活)，故半球求和的空脑判据(sum==0)不会误触发。
            print(f"[RAW NSD] layer{layer}: N={N} mask内={m}体素 "
                  f"原始t画于稳定性掩膜内(无显著门限，含负 t)")
            nii = nib.Nifti1Image(niiData.astype('double'), affine)
        elif resultType == "NSD residual mean+randomise: repStabilityMasked TFCE masked tstat":
            repStabilityMasked_tfce_path = \
                f"{scratchResultFolder}" \
                f"nsd/residualModel/meanAcrossSets/randomise/" \
                f"rad3/reg/layer{layer}/repStabilityMasked_layer{layer}__tfce_corrp_tstat1.nii.gz"
            repStabilityMasked_tfce = nib.load(repStabilityMasked_tfce_path).get_fdata()
            tstat_path = f"{scratchResultFolder}" \
                         f"nsd/residualModel/meanAcrossSets/randomise/" \
                         f"rad3/reg/layer{layer}/layer{layer}__tstat1.nii.gz"
            affine = nib.load(repStabilityMasked_tfce_path).affine
            tstat = nib.load(tstat_path).get_fdata()
            niiData = repStabilityMasked_tfce * tstat
            nii = nib.Nifti1Image(niiData.astype('double'), affine)
        elif resultType == "NSD residual mean+randomise: repStabilityMasked FDR masked tstat":
            # fig6A(NSD 残差模型 panel A)：与 fig4A 的 FDR 分支同源，只换成残差模型的重建
            # (`_nsd_residual_meanacross_tstat_and_mask`)。量、FDR 校正、稳定性掩膜口径都对齐 fig4A，
            # 只是把非残差的层表征换成残差模型的(回归掉共享跨层方差后剩下的层特异表征)。
            tstat, N, affine, in_mask = _nsd_residual_meanacross_tstat_and_mask(scratchResultFolder, layer)
            df = N - 1
            sig, info = fdr_significant_mask(tstat, in_mask, df=df, q=0.05)
            print(f"[FDR residual] layer{layer}: N={N} df={df} mask内={info['m']}体素 FDR显著={info['n_sig']}体素")
            niiData = sig * tstat
            nii = nib.Nifti1Image(niiData.astype('double'), affine)
        elif resultType == f"synthesized-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat":
            clusterm = nib.load(f"{scratchResultFolder}"
                                f"findLayerInBrain_leaveOneSubOutReg/testClustermThreshold/randomise/layer{layer}"
                                f"/reg/rad3/threshold_{thresholdName}/__clusterm_corrp_tstat1.nii.gz").get_fdata()
            clusterm[clusterm < 0.95] = 0
            clusterm[clusterm >= 0.95] = 1

            repStabilityMask = nib.load(f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/results/"
                                        f"repStability_RSA_sl/rad3/thresholdName{thresholdName}/"
                                        f"__clusterm_corrp_tstat1.nii.gz").get_fdata()
            repStabilityMask[repStabilityMask < 0.95] = 0
            repStabilityMask[repStabilityMask >= 0.95] = 1

            tstat = nib.load(f"{scratchResultFolder}"
                             f"findLayerInBrain_leaveOneSubOutReg/testClustermThreshold/randomise/layer{layer}"
                             f"/reg/rad3/threshold_{thresholdName}/__tstat1.nii.gz").get_fdata()

            affine = nib.load(f"{scratchResultFolder}"
                              f"findLayerInBrain_leaveOneSubOutReg/testClustermThreshold/randomise/layer{layer}"
                              f"/reg/rad3/threshold_{thresholdName}/__tstat1.nii.gz").affine
            niiData = clusterm * repStabilityMask * tstat
            nii = nib.Nifti1Image(niiData.astype('double'), affine)
        elif resultType == f"nsd-nonresidual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat":

                radius = 3
                method = 'reg'
                clustermPath = f"{scratchResultFolder}/nsd/" \
                               f"findLayerInBrain_leaveOneSubOutReg/testClustermThreshold/randomise/" \
                               f"rad{radius}/{method}/layer{layer}/threshold_{thresholdName}/"
                fileName = f"{clustermPath}/layer{layer}_threshold_{thresholdName}__clusterm_corrp_tstat1.nii.gz"
                clusterm = nib.load(fileName).get_fdata()
                print(f"loading {fileName}")
                clusterm[clusterm < 0.95] = 0
                clusterm[clusterm >= 0.95] = 1

                repStabilityMask = nib.load(
                    f"{scratchResultFolder}/nsd/repStability_RSA_sl_clusterm/"
                    f"nonresidual/rad{radius}/thresholdName{thresholdName}/"
                    f"thresholdName{thresholdName}__clusterm_corrp_tstat1.nii.gz").get_fdata()
                repStabilityMask[repStabilityMask < 0.95] = 0
                repStabilityMask[repStabilityMask >= 0.95] = 1

                tstatFile = nib.load(f"{clustermPath}/layer{layer}_threshold_{thresholdName}__tstat1.nii.gz")
                tstat = tstatFile.get_fdata()

                affine = nib.load("/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz").affine

                niiData = clusterm * repStabilityMask * tstat
                nii = nib.Nifti1Image(niiData.astype('double'), affine)
        elif resultType == f"nsd-residual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat":

                radius = 3
                method = 'reg'
                clustermPath = f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/nsd/" \
                               f"residual/testClustermThreshold/randomise/" \
                               f"rad{radius}/{method}/layer{layer}/threshold_{thresholdName}/"
                clusterm = nib.load(f"{clustermPath}/"
                                    f"layer{layer}_threshold_{thresholdName}__clusterm_corrp_tstat1.nii.gz").get_fdata()
                clusterm[clusterm < 0.95] = 0
                clusterm[clusterm >= 0.95] = 1

                repStabilityMask = nib.load(
                    f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/nsd/repStability_RSA_sl_clusterm/"
                    f"nonresidual/rad{radius}/thresholdName{thresholdName}/"
                    f"thresholdName{thresholdName}__clusterm_corrp_tstat1.nii.gz").get_fdata()
                repStabilityMask[repStabilityMask < 0.95] = 0
                repStabilityMask[repStabilityMask >= 0.95] = 1

                tstatFile = nib.load(f"{clustermPath}/layer{layer}_threshold_{thresholdName}__tstat1.nii.gz")
                tstat = tstatFile.get_fdata()

                affine = nib.load("/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz").affine

                niiData = clusterm * repStabilityMask * tstat
                nii = nib.Nifti1Image(niiData.astype('double'), affine)

        elif resultType == "synthesized: group_z unthresholded (FDR outline)":
            # rebuttal 图(viz_group_z_outlined.py)：无阈值连续 group z-score 底图，配合
            # contourMaskPathFn 叠黑色 FDR 显著轮廓。与 fig3a 的 FDR 分支不同，这里不乘任何
            # 显著性掩膜——niiData 就是 analysis_10 存的 Stouffer group_z，正负都保留。
            gz_path = (f"{scratchResultFolder}rebuttal_analyses/noise_ceiling_viz/"
                      f"group_z_layer{layer}_synthesized.nii.gz")
            if not os.path.exists(gz_path):
                raise FileNotFoundError(f"[门禁 outlined] layer{layer} 缺 group_z: {gz_path}")
            nii = nib.load(gz_path)
            niiData = nii.get_fdata()
            # niiData 带符号(z 可正可负)，下面"空半球"判据(hemisphere sum==0)理论上会被正负抵消
            # 误触发；同 RAW tstat 分支的判断，真实 z-score 数据里整半球精确求和为 0 概率可忽略。

        elif resultType == "synthesized: tstat unthresholded (FDR outline)":
            # S14 synthesized panel aligned with final Figure 3: show the
            # unthresholded one-sample t statistic underlying Figure 3, while
            # contourMaskPathFn supplies the same t-based BH-FDR outline.
            tstat_path = (f"{scratchResultFolder}"
                          f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/"
                          f"layer{layer}/reg/rad3/__tstat1.nii.gz")
            if not os.path.exists(tstat_path):
                raise FileNotFoundError(
                    f"[门禁 outlined-t] layer{layer} 缺 unthresholded t 图: {tstat_path}")
            nii = nib.load(tstat_path)
            niiData = nii.get_fdata()

        elif resultType == "synthesized: raw R2 unthresholded (FDR outline)":
            # rebuttal 图(viz_raw_R2_surface_outlined.py)：无阈值 raw cross-validated R² 底图
            # （不是 z-score），配合 contourMaskPathFn 叠黑色 FDR 显著轮廓——用同一套显著性判据
            # (viz_group_z_outlined.py 已存好的每层 FDR 掩膜)，直观展示"就算在 z-score 显著的
            # 体素里，raw R² 依然是负的"(R1P2)。niiData 就是 viz_raw_R2_surface_outlined.py
            # 存的 mean_R2 = normalized_R2 * nc_lower(与 viz_raw_R2_negative.py 同一算法)。
            r2_path = (f"{scratchResultFolder}rebuttal_analyses/noise_ceiling_viz/"
                      f"mean_R2_layer{layer}_synthesized.nii.gz")
            if not os.path.exists(r2_path):
                raise FileNotFoundError(f"[门禁 raw-R2-outlined] layer{layer} 缺 mean_R2: {r2_path}")
            nii = nib.load(r2_path)
            niiData = nii.get_fdata()

        elif resultType == "NSD nonresidual mean+randomise: group_z unthresholded (FDR outline)":
            # rebuttal 图(viz_group_z_outlined_nsd.py)：NSD 版"无阈值连续底图 + FDR 黑色轮廓"，
            # 与 Figure 4 配套(呼应 R1P2/R1P4 "we need the unthresholded map for the NSD version,
            # akin to S14")。tstat/in_mask 复用 fig4A FDR 分支同一份重建(_nsd_meanacross_tstat_
            # and_mask)，这里不乘 in_mask——niiData 是全脑单样本 t，正负都保留，与 synthesized 版
            # 的"不乘显著性掩膜"是同一个设计。_nsd_meanacross_tstat_and_mask 内部已对输入文件数量/
            # 维度/被试数做过门禁；这里额外补一道"稳定性掩膜非空 + 掩膜内 t 有限"的检查，跟同一个
            # 文件里的 "RAW tstat (no sig)" 分支(line ~646)保持同等严格度，不因为改成全脑显示就
            # 松掉这道检查。
            tstat, _N, affine, in_mask = _nsd_meanacross_tstat_and_mask(scratchResultFolder, layer)
            m = int(in_mask.sum())
            if m == 0:
                raise ValueError(f"[门禁 outlined-NSD] layer{layer} 稳定性掩膜为空，没有可画的体素")
            if not np.all(np.isfinite(tstat[in_mask.astype(bool)])):
                raise ValueError(f"[门禁 outlined-NSD] layer{layer} 掩膜内有非有限 t 值(重建异常)")
            niiData = tstat
            nii = nib.Nifti1Image(niiData.astype('double'), affine)

        elif resultType == "NSD residual mean+randomise: tstat unthresholded (FDR outline)":
            # Figure S18B: show the complete signed one-sample t map underlying
            # Figure 6.  The representational-stability mask is used to define
            # the separately supplied FDR contour, not to threshold this base map.
            tstat, _N, affine, in_mask = _nsd_residual_meanacross_tstat_and_mask(
                scratchResultFolder, layer)
            m = int(in_mask.sum())
            if m == 0:
                raise ValueError(
                    f"[门禁 outlined-residual] layer{layer} 稳定性掩膜为空，没有可检验的体素")
            if not np.all(np.isfinite(tstat[in_mask.astype(bool)])):
                raise ValueError(
                    f"[门禁 outlined-residual] layer{layer} 掩膜内有非有限 t 值")
            niiData = tstat
            nii = nib.Nifti1Image(niiData.astype('double'), affine)

        elif resultType == "NSD nonresidual: raw R2 unthresholded (FDR outline)":
            # R1P2 follow-up: the true, unnormalized cross-validated R² for NSD.
            # viz_raw_R2_surface_outlined_nsd.py computes a complete 8-subject ×
            # 16-set mean for each layer and stores it with the corrected MNI 2-mm
            # affine. The FDR contour is supplied separately via contourMaskPathFn.
            r2_path = (f"{scratchResultFolder}rebuttal_analyses/noise_ceiling_viz/"
                       f"mean_R2_layer{layer}_nsd_nonresidual.nii.gz")
            if not os.path.exists(r2_path):
                raise FileNotFoundError(f"[门禁 raw-R2-NSD] layer{layer} 缺 mean raw R²: {r2_path}")
            nii = nib.load(r2_path)
            niiData = nii.get_fdata()

        elif resultType == "NSD residual: raw R2 unthresholded (FDR outline)":
            # Figure S18A: true residual-model cross-validated R², averaged over
            # the complete 8-subject x 16-set design.  The FDR contour is loaded
            # independently and does not threshold the raw-R² base map.
            r2_path = (f"{scratchResultFolder}rebuttal_analyses/noise_ceiling_viz/"
                       f"mean_R2_layer{layer}_nsd_residual.nii.gz")
            if not os.path.exists(r2_path):
                raise FileNotFoundError(
                    f"[门禁 raw-R2-residual] layer{layer} 缺 mean raw R²: {r2_path}")
            nii = nib.load(r2_path)
            niiData = nii.get_fdata()

        else:
            raise Exception

        valueRange[f"layer-{layer}"] = [np.min(niiData), np.max(niiData)]
        print(f"\nlayer-{layer}", end=' ')
        print(f"niiData max: {np.max(niiData)}", end=' ')
        print(f"niiData min: {np.min(niiData)}", end=' ')
        print(f"niiData mean: {np.mean(niiData)}", end=' ')
        print(f"niiData std: {np.std(niiData)}", end=' ')

        leftHemi = niiData[46:-1, :, :]
        leftHemiNumber = np.sum(leftHemi)
        rightHemi = niiData[0:46, :, :]
        rightHemiNumber = np.sum(rightHemi)

        nonZeroVoxelNumber = np.sum(niiData != 0)
        maxNiiValue = np.max(niiData)
        first_subplot = True
        threshold = displayThreshold      # None=旧行为；FDR 图传 0.01 让所有显著体素都显示
        darkness = bgDarkness             # None=旧行为(较暗)；fig3a_fdr/fig4a_fdr 传 0.3 使与空层分支一致的浅背景
        # contourMaskPathFn: 该层的显著性掩膜(与 niiData 同 affine)投到皮层表面上，供下面
        # 叠黑框；每层每半球只需算一次，views 循环里复用。缺文件直接 raise，不静默出无框图。
        contour_texture = None
        if contourMaskPathFn is not None:
            contour_path = contourMaskPathFn(layer)
            if contour_path is not None:
                if not os.path.exists(contour_path):
                    raise FileNotFoundError(f"[contourMaskPathFn] layer{layer} 掩膜文件不存在: {contour_path}")
                contour_nii = nib.load(contour_path)
                contour_data = contour_nii.get_fdata()
                if contour_data.shape != niiData.shape:
                    raise ValueError(
                        f"[contourMaskPathFn] layer{layer} 掩膜 shape {contour_data.shape} "
                        f"与底图 {niiData.shape} 不一致: {contour_path}")
                if not np.allclose(contour_nii.affine, nii.affine, atol=1e-4):
                    raise ValueError(
                        f"[contourMaskPathFn] layer{layer} 掩膜 affine 与底图不一致: {contour_path}")
                if not np.all(np.isfinite(contour_data)):
                    raise ValueError(
                        f"[contourMaskPathFn] layer{layer} 掩膜含非有限值: {contour_path}")
                unique_contour = np.unique(contour_data)
                if not np.all(np.isin(unique_contour, [0, 1])):
                    raise ValueError(
                        f"[contourMaskPathFn] layer{layer} 掩膜不是二值 0/1: "
                        f"values={unique_contour[:10]}, path={contour_path}")
        for hemi in hemispheres:
            if hemi == 'left':
                texture = surface.vol_to_surf(nii, fsaverage.pial_left,
                                              interpolation='nearest')
                bg_map = fsaverage.sulc_left
                surf = fsaverage.infl_left
                if contourMaskPathFn is not None and contour_path is not None:
                    contour_texture = surface.vol_to_surf(contour_nii, fsaverage.pial_left,
                                                          interpolation='nearest')
            else:
                texture = surface.vol_to_surf(nii, fsaverage.pial_right,
                                              interpolation='nearest')
                bg_map = fsaverage.sulc_right
                surf = fsaverage.infl_right
                if contourMaskPathFn is not None and contour_path is not None:
                    contour_texture = surface.vol_to_surf(contour_nii, fsaverage.pial_right,
                                                          interpolation='nearest')
            for view in views:
                imgFPath = _img_path(layer, view, hemi)
                if (not os.path.exists(imgFPath)) or forceRun:
                    if (leftHemiNumber == 0 and hemi == 'left') or (rightHemiNumber == 0 and hemi == 'right'):
                        def plot_empty_brain_(hemi=None):
                            from nilearn import datasets
                            from nilearn import plotting
                            from nilearn import surface
                            import numpy as np
                            import nibabel as nib
                            fsaverage = datasets.fetch_surf_fsaverage()
                            nii = nib.load(
                                "/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz")
                            niiData = nii.get_fdata()
                            nii = nib.Nifti1Image(niiData, nii.affine)
                            leftHemi = niiData[0:46, :, :]
                            rightHemi = niiData[46:-1, :, :]
                            maxNiiValue = np.max(niiData)
                            hemispheres = ['left', 'right']

                            if hemi == 'left':
                                texture = surface.vol_to_surf(nii, fsaverage.pial_left, interpolation='nearest')
                                bg_map = fsaverage.sulc_left
                                surf = fsaverage.infl_left
                            else:
                                texture = surface.vol_to_surf(nii, fsaverage.pial_right,
                                                              interpolation='nearest')
                                bg_map = fsaverage.sulc_right
                                surf = fsaverage.infl_right
                            for view in views:
                                surf_map = plotting.plot_surf_stat_map(surf, texture, cmap=cmap,
                                                                       hemi=hemi, view=view, bg_on_data=False,
                                                                       bg_map=bg_map,
                                                                       bg_img='MNI152',
                                                                       black_bg=True,
                                                                       colorbar=False,
                                                                       threshold=1000000,
                                                                       darkness=0.3,
                                                                       )

                        plot_empty_brain_(hemi=hemi)
                    else:
                        surf_map = plotting.plot_surf_stat_map(surf, texture, cmap=cmap,
                                                               hemi=hemi, view=view, bg_on_data=True,
                                                               bg_map=bg_map,
                                                               bg_img='MNI152',
                                                               colorbar=False,
                                                               black_bg=False,
                                                               threshold=threshold,
                                                               darkness=darkness,
                                                               vmax=vmax
                                                               )
                        if contour_texture is not None and (contour_texture > 0.5).any():
                            plotting.plot_surf_contours(
                                surf, (contour_texture > 0.5).astype(int),
                                axes=surf_map.axes[0], figure=surf_map,
                                levels=[1], colors=['black'])
                    plt.savefig(imgFPath, dpi=600)

                    plt.close()
                    if first_subplot:
                        surf_map = plotting.plot_surf_stat_map(surf, texture, cmap=cmap,
                                                               hemi=hemi, view=view, bg_on_data=True,
                                                               bg_map=bg_map,
                                                               bg_img='MNI152',
                                                               colorbar=True,
                                                               black_bg=False,
                                                               threshold=threshold,
                                                               darkness=darkness,
                                                               vmax=vmax
                                                               )
                        colorbar_pdf_path =f"{savePath.split('.pdf')[0]}_colorbar.pdf"
                        plt.savefig(colorbar_pdf_path, transparent=True)

                        plt.close()

                        first_subplot=False

                    print(f"resultType-{resultType}: layer-{layer} {view} {hemi}-hemi "
                          f"voxelNumber-left{leftHemiNumber}-right{rightHemiNumber} "
                          f"maxValue-{maxNiiValue}")

    # 并行 render 阶段：只渲染 PNG、不拼图（各层 PNG 独立缓存，拼图交给最后的装配 job）。
    if renderOnly:
        return valueRange
    from PIL import Image
    arrangement = "3-8"
    if arrangement == "6-4":
        fig, axes = plt.subplots(nrows=6, ncols=4, subplot_kw={'projection': '3d'}, figsize=(20, 20))
        w_pad = -2
        h_pad = -4
    elif arrangement == "3-8":
        fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(40, 40))
        if views[0] == 'posterior':
            w_pad = w_pad_
            h_pad = h_pad_
        else:
            w_pad = -3
            h_pad = -200
    else:
        raise Exception
    # 装配版式（仅真实的 3-8 排布）：右侧留出整合 colorbar 的竖条、顶部留出 "A"/"Layer"/层号 的横带。
    # tight_layout 在设标题之前调用 → 不为标题预留空间；层号作为标题落在各行上方的既有间隙里，
    # 顶部横带只需把整个网格向下压一点即可容纳 "A"/"Layer"。历史 6-4 排布保持原有满幅版式。
    if arrangement == "3-8":
        layout_rect = (0, 0, 0.9, 0.93)
    else:
        layout_rect = (0, 0, 1, 1)
    plt.tight_layout(pad=0, w_pad=w_pad, h_pad=h_pad, rect=layout_rect)

    for layer in range(1, 13):
        for hemi in hemispheres:
            if arrangement == "6-4":
                if layer - 1 > 5.5:
                    figureAxis_x = layer - 1 - 6
                    if hemi == 'left':
                        figureAxis_y = 2
                    else:
                        figureAxis_y = 3
                else:
                    figureAxis_x = layer - 1
                    if hemi == 'left':
                        figureAxis_y = 0
                    else:
                        figureAxis_y = 1
            elif arrangement == "3-8":
                def testArangement():
                    for layer in range(1, 13):
                        figureAxis_x = np.floor((layer - 1) / 4)
                        for hemi in range(2):
                            figureAxis_y = int(2 * (layer - 1 - np.floor((layer - 1) / 4) * 4) + hemi)
                            print(figureAxis_x, figureAxis_y)

                figureAxis_x = int(np.floor((layer - 1) / 4))
                if hemi == 'left':
                    figureAxis_y = int(2 * (layer - 1 - np.floor((layer - 1) / 4) * 4) + 0)
                else:
                    figureAxis_y = int(2 * (layer - 1 - np.floor((layer - 1) / 4) * 4) + 1)
            else:
                raise Exception
            for view in views:
                imgFPath = _img_path(layer, view, hemi)
                array = np.array(Image.open(imgFPath))
                image_noBorder = remove_whiteBorder(array)

                axes[figureAxis_x, figureAxis_y].imshow(image_noBorder)
                # 层号：每个"左右半球对"上方只放一个居中的纯数字（不再每格一个 "layer-N"）。
                # 标题挂在左半球那一格、x=1.0 —— 两格共享的边界正是这一对的水平中心，黑色纯数字。
                if hemi == 'left':
                    axes[figureAxis_x, figureAxis_y].set_title(
                        f'{layer}', x=1.0, fontsize=48, color='black')
                axes[figureAxis_x, figureAxis_y].set_xticks([])
                axes[figureAxis_x, figureAxis_y].set_yticks([])
                axes[figureAxis_x, figureAxis_y].axis('off')
    # ---- 出版级装饰（面板 A）：面板标号 "A"、斜体 "Layer"、右侧整合 colorbar ----
    # 只对真实的 3-8 排布加；历史 6-4 分支不动。位置/字号按原发表图目测标定，见 report。
    if arrangement == "3-8":
        # 面板标号 "A"：图左上角，大号纯字。
        fig.text(0.012, 0.982, 'A', fontsize=80, color='black', ha='left', va='center')
        # "Layer"：斜体，置于网格左上方（在层号 "1" 的左上侧）。
        fig.text(0.05, 0.955, 'Layer', fontsize=52, color='black',
                 style='italic', ha='left', va='center')
        # 整合 colorbar：右侧单条竖向、纵向大致居中。由函数参数 cmap / vmax 驱动，绝不写死 8——
        # 其它调用方传的 vmax 不同（22/7/6…）。刻度用 np.linspace(0, vmax, 5) 以便通用。
        import matplotlib.cm as _cm
        from matplotlib.colors import Normalize as _Normalize
        # colorbar 的下端：历史默认是 0，但 nilearn 的 plot_surf_stat_map 只接 vmax、
        # 内部一律按对称区间 [-vmax, +vmax] 渲染（它是统计图专用函数，0 固定落在色标中点）。
        # 于是画出来的 colorbar 和真正的表面配色对不上：读者看到的近白区其实是 0，colorbar
        # 却标成 vmax/2。数据全为负时这个错位最致命（raw R² 图实测 positive=0.0%，
        # 按标称的 [0, vmax] 渲染应当整脑钳到最深蓝，实际却大片近白，可反证渲染是对称的）。
        # 默认值保持 0 不动，避免动到已定稿的历史图；需要正确对称色标的调用方显式传
        # colorBarVmin=-vmax。标签同理：写死的 't statistic value' 对画 group-z / raw R²
        # 的图是错的，用 colorBarLabel 覆盖。
        _cb_vmin = 0 if colorBarVmin is None else colorBarVmin
        _cb_label = 't statistic value' if colorBarLabel is None else colorBarLabel
        from matplotlib.ticker import FormatStrFormatter as _FormatStrFormatter
        _sm = _cm.ScalarMappable(cmap=cmap, norm=_Normalize(vmin=_cb_vmin, vmax=vmax))
        _sm.set_array([])
        # x 从 0.925 挪到 0.912：改成对称区间之后刻度多了负号、字符更长，轴标签被挤出画布
        # 裁掉了（实测 group-z 图和 NSD 图的内容都顶到第 3999 列）。历史图默认保留两位小数，
        # 因为 np.linspace 出来的 4.642 / 2.321 这种三位小数最占宽。本地按 figsize=(40,40)、
        # dpi=100 复现过这段布局做网格搜索：0.912 + 两位小数在三张图上右侧最少还剩 34px，
        # 左侧到脑面板还有 60px，两头都不挤，且字号一个没动。
        cbar_ax = fig.add_axes([0.912, 0.35, 0.018, 0.30])
        cbar = fig.colorbar(_sm, cax=cbar_ax, ticks=np.linspace(_cb_vmin, vmax, 5))
        cbar.ax.yaxis.set_major_formatter(_FormatStrFormatter(colorBarTickFormat))
        cbar.ax.tick_params(labelsize=44)
        # 刻度数字在竖条右侧（竖向 colorbar 的默认位置），标签也放右侧、自下而上读。
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
        # labelpad=28 会把标签推到画布外被裁掉（三张 FDR 轮廓图实测都只剩几个像素），
        # 收窄到 6 让它落回画布内。量级更小的调用方可通过 colorBarTickFormat 显式增加
        # 精度，避免不同刻度被格式化成同一个 0.00。
        cbar.set_label(_cb_label, fontsize=52, rotation=90, labelpad=6)

    _ = plt.show()

    fig.savefig(savePath, transparent=True)
    print(f"savePath={savePath}")
    if copyTo is not None:
        os.makedirs(os.path.dirname(copyTo), exist_ok=True)
        shutil.copy(savePath, copyTo)
        print(f"[copyTo] 已同步到 git 跟踪位置: {copyTo}")
    plotting.show()
    print(f"resultType={resultType}, views={views}")
    print(f"valueRange={valueRange}")
    return valueRange




def plotMaximalLayer_pdf(resultType=None, views=None, cmap=None, colorBar=True, thresholdName=None, figureName="",
                         copyTo=None, bgDarkness=None):
    # copyTo: 存到 scratch 后再拷一份到这个（仓库内、git 跟踪的）路径，避免仓库副本变旧。
    # bgDarkness: 传给 nilearn 的 darkness（背景沟回明暗）。None=旧行为(较暗)，不影响现有调用方；
    #   fig3b_fdr/fig4b_fdr 传 0.3，让 panel B 背景与调浅后的 panel A 一致（同 plot_surface_saveImageReload_pdf）。
    from nilearn import datasets
    from nilearn import plotting
    from nilearn import surface
    import numpy as np
    import nibabel as nib
    fsaverage = datasets.fetch_surf_fsaverage()
    if cmap is None:
        cmap = 'bwr'
    nii = None
    niiData = None
    if resultType == "synthesized: maximally fit layer":
        tstat_allLayer = np.zeros((12, 91, 109, 91))
        affine = None
        for layer in range(1, 13):
            repStabilityMasked_tfce_path = \
                f"{scratchResultFolder}/" \
                f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/layer{layer}/" \
                f"reg/rad3/repStabilityMasked__tfce__layer{layer}.nii.gz"
            repStabilityMasked_tfce = nib.load(repStabilityMasked_tfce_path).get_fdata()
            affine = nib.load(repStabilityMasked_tfce_path).affine
            tstat_path = f"{scratchResultFolder}/" \
                         f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/layer{layer}/" \
                         f"reg/rad3/__tstat1.nii.gz"
            tstat = nib.load(tstat_path).get_fdata()
            niiData = repStabilityMasked_tfce * tstat
            tstat_allLayer[layer - 1, :, :, :] = niiData
        maxValue = np.nanmax(tstat_allLayer, axis=0)
        tstat_maxLayerID = np.argmax(tstat_allLayer, axis=0) + 1
        tstat_maxLayerID[maxValue == 0] = 0
        print(f"tstat_maxLayerID.shape={tstat_maxLayerID.shape}")
        print(f"np.unique(tstat_maxLayerID)={np.unique(tstat_maxLayerID)}")
        nii = nib.Nifti1Image(tstat_maxLayerID.astype('double'), affine)
        niiData = tstat_maxLayerID.astype('double')
    elif resultType == "synthesized: maximally fit layer FDR":
        # 方案 B1 的 panel B：照搬上面 "synthesized: maximally fit layer" 的 argmax 逻辑，
        # 只把每层的"TFCE 显著掩膜 × t"换成"FDR 显著掩膜 × t"，与 panel A 的 FDR 分支同源
        # （同 t、同稳定性掩膜、逐层各自 t→参数 p→BH q<0.05）。见 20_/21_ 与 docs/adr。
        tstat_allLayer = np.zeros((12, 91, 109, 91))
        affine = None
        for layer in range(1, 13):
            base = (f"{scratchResultFolder}"
                    f"findLayerInBrain_leaveOneSubOutReg/shuffleNorm/randomise/layer{layer}/reg/rad3/")
            tstat_path = base + "__tstat1.nii.gz"
            subs_path = base + "subs_compiled.nii.gz"      # 4D 合并被试体积，用来定 N/df
            mask_path = f"{scratchResultFolder}repStability_RSA_sl/rad3/__tfce_corrp_tstat1.nii.gz"
            for pth, what in [(tstat_path, "t 图"), (subs_path, "合并被试体积(定 N/df)"),
                              (mask_path, "稳定性掩膜")]:
                if not os.path.exists(pth):
                    raise FileNotFoundError(f"[门禁 FDR] panelB layer{layer} 缺输入 {what}: {pth}")
            tstat_nii = nib.load(tstat_path)
            affine = tstat_nii.affine
            tstat = tstat_nii.get_fdata()
            subs_nii = nib.load(subs_path)
            if subs_nii.ndim != 4:
                raise ValueError(f"[门禁 FDR] panelB subs_compiled 不是 4D(实得 {subs_nii.ndim}D)，无法定 N")
            N = int(subs_nii.shape[3])
            if N < 2:
                raise ValueError(f"[门禁 FDR] panelB 合并被试数 N={N} 过小(需 >=2)")
            df = N - 1
            stab = nib.load(mask_path).get_fdata()
            if stab.shape != tstat.shape:
                raise ValueError(f"[门禁 FDR] panelB 稳定性掩膜 {stab.shape} 与 t 图 {tstat.shape} 形状不符")
            in_mask = (stab > 0.95).astype(float)
            sig, info = fdr_significant_mask(tstat, in_mask, df=df, q=0.05)
            print(f"[FDR panelB] layer{layer}: N={N} df={df} mask内={info['m']}体素 FDR显著={info['n_sig']}体素")
            niiData = sig * tstat
            tstat_allLayer[layer - 1, :, :, :] = niiData
        maxValue = np.nanmax(tstat_allLayer, axis=0)
        tstat_maxLayerID = np.argmax(tstat_allLayer, axis=0) + 1
        tstat_maxLayerID[maxValue == 0] = 0
        print(f"tstat_maxLayerID.shape={tstat_maxLayerID.shape}")
        print(f"np.unique(tstat_maxLayerID)={np.unique(tstat_maxLayerID)}")
        nii = nib.Nifti1Image(tstat_maxLayerID.astype('double'), affine)
        niiData = tstat_maxLayerID.astype('double')
    elif resultType == "NSD nonresidual: maximally fit layer":
        tstat_allLayer = np.zeros((12, 91, 109, 91))
        affine = None
        for layer in range(1, 13):
            repStabilityMasked_tfce_path = \
                f"{scratchResultFolder}/" \
                f"nsd/findLayerInBrain_leaveOneSubOutReg/meanAcrossSets/randomise/" \
                f"rad3/reg/layer{layer}/repStabilityMasked_layer{layer}__tfce_corrp_tstat1.nii.gz"
            repStabilityMasked_tfce = nib.load(repStabilityMasked_tfce_path).get_fdata()
            tstat_path = f"{scratchResultFolder}/" \
                         f"nsd/findLayerInBrain_leaveOneSubOutReg/meanAcrossSets/randomise/" \
                         f"rad3/reg/layer{layer}/layer{layer}__tstat1.nii.gz"
            affine = nib.load(repStabilityMasked_tfce_path).affine
            tstat = nib.load(tstat_path).get_fdata()
            niiData = repStabilityMasked_tfce * tstat
            tstat_allLayer[layer - 1, :, :, :] = niiData
        maxValue = np.nanmax(tstat_allLayer, axis=0)
        tstat_maxLayerID = np.argmax(tstat_allLayer, axis=0) + 1
        tstat_maxLayerID[maxValue == 0] = 0
        print(f"tstat_maxLayerID.shape={tstat_maxLayerID.shape}")
        print(f"np.unique(tstat_maxLayerID)={np.unique(tstat_maxLayerID)}")
        nii = nib.Nifti1Image(tstat_maxLayerID.astype('double'), affine)
        niiData = tstat_maxLayerID.astype('double')
    elif resultType == "NSD nonresidual: maximally fit layer FDR":
        # fig4B（NSD panel B）的 FDR：同 fig4A，从现存的逐-set subs_compiled 廉价重建 meanAcrossSets
        # 的单样本 t（原 meanAcrossSets randomise 输出已被 scratch 清理）。
        tstat_allLayer = np.zeros((12, 91, 109, 91))
        affine = None
        for layer in range(1, 13):
            tstat, N, affine, in_mask = _nsd_meanacross_tstat_and_mask(scratchResultFolder, layer)
            df = N - 1
            sig, info = fdr_significant_mask(tstat, in_mask, df=df, q=0.05)
            print(f"[FDR panelB NSD] layer{layer}: N={N} df={df} mask内={info['m']}体素 FDR显著={info['n_sig']}体素")
            niiData = sig * tstat
            tstat_allLayer[layer - 1, :, :, :] = niiData
        maxValue = np.nanmax(tstat_allLayer, axis=0)
        tstat_maxLayerID = np.argmax(tstat_allLayer, axis=0) + 1
        tstat_maxLayerID[maxValue == 0] = 0
        print(f"tstat_maxLayerID.shape={tstat_maxLayerID.shape}")
        print(f"np.unique(tstat_maxLayerID)={np.unique(tstat_maxLayerID)}")
        nii = nib.Nifti1Image(tstat_maxLayerID.astype('double'), affine)
        niiData = tstat_maxLayerID.astype('double')
    elif resultType == "NSD residual: maximally fit layer":
        tstat_allLayer = np.zeros((12, 91, 109, 91))
        affine = None
        for layer in range(1, 13):
            repStabilityMasked_tfce_path = \
                f"{scratchResultFolder}/" \
                f"nsd/residualModel/meanAcrossSets/randomise/" \
                f"rad3/reg/layer{layer}/repStabilityMasked_layer{layer}__tfce_corrp_tstat1.nii.gz"
            repStabilityMasked_tfce = nib.load(repStabilityMasked_tfce_path).get_fdata()
            tstat_path = f"{scratchResultFolder}/" \
                         f"nsd/residualModel/meanAcrossSets/randomise/" \
                         f"rad3/reg/layer{layer}/layer{layer}__tstat1.nii.gz"
            affine = nib.load(repStabilityMasked_tfce_path).affine
            tstat = nib.load(tstat_path).get_fdata()
            niiData = repStabilityMasked_tfce * tstat
            tstat_allLayer[layer - 1, :, :, :] = niiData

        maxValue = np.nanmax(tstat_allLayer, axis=0)
        tstat_maxLayerID = np.argmax(tstat_allLayer, axis=0) + 1
        tstat_maxLayerID[maxValue == 0] = 0
        print(f"tstat_maxLayerID.shape={tstat_maxLayerID.shape}")
        print(f"np.unique(tstat_maxLayerID)={np.unique(tstat_maxLayerID)}")
        nii = nib.Nifti1Image(tstat_maxLayerID.astype('double'), affine)
        niiData = tstat_maxLayerID.astype('double')
    elif resultType == "NSD residual: maximally fit layer FDR":
        # fig6B(NSD 残差模型 panel B)：与 fig4B 的 FDR 分支同源，重建换成
        # `_nsd_residual_meanacross_tstat_and_mask`，argmax/背景清零/渲染逻辑完全不动。
        tstat_allLayer = np.zeros((12, 91, 109, 91))
        affine = None
        for layer in range(1, 13):
            tstat, N, affine, in_mask = _nsd_residual_meanacross_tstat_and_mask(scratchResultFolder, layer)
            df = N - 1
            sig, info = fdr_significant_mask(tstat, in_mask, df=df, q=0.05)
            print(f"[FDR panelB residual] layer{layer}: N={N} df={df} mask内={info['m']}体素 "
                  f"FDR显著={info['n_sig']}体素")
            niiData = sig * tstat
            tstat_allLayer[layer - 1, :, :, :] = niiData
        maxValue = np.nanmax(tstat_allLayer, axis=0)
        tstat_maxLayerID = np.argmax(tstat_allLayer, axis=0) + 1
        tstat_maxLayerID[maxValue == 0] = 0
        print(f"tstat_maxLayerID.shape={tstat_maxLayerID.shape}")
        print(f"np.unique(tstat_maxLayerID)={np.unique(tstat_maxLayerID)}")
        nii = nib.Nifti1Image(tstat_maxLayerID.astype('double'), affine)
        niiData = tstat_maxLayerID.astype('double')
    elif resultType == f"synthesized-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat":
            tstat_allLayer = np.zeros((12, 91, 109, 91))
            affine = None
            for layer in range(1, 13):
                clusterm = nib.load(f"{scratchResultFolder}"
                                    f"findLayerInBrain_leaveOneSubOutReg/testClustermThreshold/randomise/layer{layer}"
                                    f"/reg/rad3/threshold_{thresholdName}/__clusterm_corrp_tstat1.nii.gz").get_fdata()
                clusterm[clusterm < 0.95] = 0
                clusterm[clusterm >= 0.95] = 1

                repStabilityMask = nib.load(f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/results/"
                                            f"repStability_RSA_sl/rad3/thresholdName{thresholdName}/"
                                            f"__clusterm_corrp_tstat1.nii.gz").get_fdata()
                repStabilityMask[repStabilityMask < 0.95] = 0
                repStabilityMask[repStabilityMask >= 0.95] = 1

                tstat = nib.load(f"{scratchResultFolder}"
                                 f"findLayerInBrain_leaveOneSubOutReg/testClustermThreshold/randomise/layer{layer}"
                                 f"/reg/rad3/threshold_{thresholdName}/__tstat1.nii.gz").get_fdata()

                affine = nib.load(f"{scratchResultFolder}"
                                  f"findLayerInBrain_leaveOneSubOutReg/testClustermThreshold/randomise/layer{layer}"
                                  f"/reg/rad3/threshold_{thresholdName}/__tstat1.nii.gz").affine

                niiData = clusterm * repStabilityMask * tstat
                tstat_allLayer[layer - 1, :, :, :] = niiData
            maxValue = np.nanmax(tstat_allLayer, axis=0)
            tstat_maxLayerID = np.argmax(tstat_allLayer, axis=0) + 1
            tstat_maxLayerID[maxValue == 0] = 0
            print(f"tstat_maxLayerID.shape={tstat_maxLayerID.shape}")
            print(f"np.unique(tstat_maxLayerID)={np.unique(tstat_maxLayerID)}")
            nii = nib.Nifti1Image(tstat_maxLayerID.astype('double'), affine)
            niiData = tstat_maxLayerID.astype('double')
    elif resultType == f"nsd-nonresidual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat":
            tstat_allLayer = np.zeros((12, 91, 109, 91))
            affine = None
            for layer in range(1, 13):
                radius = 3
                method = 'reg'
                clustermPath = f"{scratchResultFolder}/nsd/" \
                               f"findLayerInBrain_leaveOneSubOutReg/testClustermThreshold/randomise/" \
                               f"rad{radius}/{method}/layer{layer}/threshold_{thresholdName}/"
                fileName = f"{clustermPath}/layer{layer}_threshold_{thresholdName}__clusterm_corrp_tstat1.nii.gz"
                clusterm = nib.load(fileName).get_fdata()
                print(f"loading {fileName}")

                clusterm[clusterm < 0.95] = 0
                clusterm[clusterm >= 0.95] = 1

                repStabilityMask = nib.load(
                    f"{scratchResultFolder}/nsd/repStability_RSA_sl_clusterm/"
                    f"nonresidual/rad{radius}/thresholdName{thresholdName}/"
                    f"thresholdName{thresholdName}__clusterm_corrp_tstat1.nii.gz").get_fdata()
                repStabilityMask[repStabilityMask < 0.95] = 0
                repStabilityMask[repStabilityMask >= 0.95] = 1

                tstatFile = nib.load(f"{clustermPath}/layer{layer}_threshold_{thresholdName}__tstat1.nii.gz")

                tstat = tstatFile.get_fdata()
                affine = nib.load("/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz").affine
                niiData = clusterm * repStabilityMask * tstat
                tstat_allLayer[layer - 1, :, :, :] = niiData
            maxValue = np.nanmax(tstat_allLayer, axis=0)
            tstat_maxLayerID = np.argmax(tstat_allLayer, axis=0) + 1
            tstat_maxLayerID[maxValue == 0] = 0
            print(f"tstat_maxLayerID.shape={tstat_maxLayerID.shape}")
            print(f"np.unique(tstat_maxLayerID)={np.unique(tstat_maxLayerID)}")
            nii = nib.Nifti1Image(tstat_maxLayerID.astype('double'), affine)
            niiData = tstat_maxLayerID.astype('double')
    elif resultType == f"nsd-residual-repStabilityMasked-clusterm-threshold_{thresholdName}-masked-tstat":
                tstat_allLayer = np.zeros((12, 91, 109, 91))
                affine = None
                for layer in range(1, 13):
                    radius = 3
                    method = 'reg'
                    clustermPath = f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/nsd/" \
                                   f"residual/testClustermThreshold/randomise/" \
                                   f"rad{radius}/{method}/layer{layer}/threshold_{thresholdName}/"
                    clusterm = nib.load(f"{clustermPath}/"
                                        f"layer{layer}_threshold_{thresholdName}__clusterm_corrp_tstat1.nii.gz").get_fdata()
                    clusterm[clusterm < 0.95] = 0
                    clusterm[clusterm >= 0.95] = 1

                    repStabilityMask = nib.load(
                        f"/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/nsd/repStability_RSA_sl_clusterm/"
                        f"nonresidual/rad{radius}/thresholdName{thresholdName}/"
                        f"thresholdName{thresholdName}__clusterm_corrp_tstat1.nii.gz").get_fdata()
                    repStabilityMask[repStabilityMask < 0.95] = 0
                    repStabilityMask[repStabilityMask >= 0.95] = 1

                    tstatFile = nib.load(f"{clustermPath}/layer{layer}_threshold_{thresholdName}__tstat1.nii.gz")

                    tstat = tstatFile.get_fdata()
                    affine = nib.load(
                        "/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz").affine
                    niiData = clusterm * repStabilityMask * tstat
                    tstat_allLayer[layer - 1, :, :, :] = niiData
                maxValue = np.nanmax(tstat_allLayer,
                                     axis=0)
                tstat_maxLayerID = np.argmax(tstat_allLayer, axis=0) + 1
                tstat_maxLayerID[maxValue == 0] = 0
                print(f"tstat_maxLayerID.shape={tstat_maxLayerID.shape}")
                print(f"np.unique(tstat_maxLayerID)={np.unique(tstat_maxLayerID)}")
                nii = nib.Nifti1Image(tstat_maxLayerID.astype('double'), affine)
                niiData = tstat_maxLayerID.astype('double')
    else:
        raise Exception

    leftHemi = niiData[46:-1, :, :]
    leftHemiNumber = np.sum(leftHemi)
    rightHemi = niiData[0:46, :, :]
    rightHemiNumber = np.sum(rightHemi)
    nonZeroVoxelNumber = np.sum(niiData != 0)
    maxNiiValue = np.max(niiData)
    hemispheres = ['left', 'right']


    def wholeBrainPositions():
        base_width = 0.4
        base_height = 0.4
        positions = {
            "lateral-left": [0.00,
                             0.23854166666666665,
                             base_width, base_height],
            "lateral-right": [0.20693055555555548,
                              0.23854166666666665,
                              base_width, base_height],

            "medial-left": [0.00,
                            0.00,
                            base_width, base_height],
            "medial-right": [0.20693055555555548,
                             0.00,
                             base_width, base_height],

            "posterior-left": [0.364, 0.10999999999999999,
                               base_width, base_height],
            "posterior-right": [0.44599999999999995,
                                0.10999999999999999,
                                base_width, base_height],
        }
        return positions



    positions = wholeBrainPositions()
    fig, axes = plt.subplots(figsize=(3 * 10, 2 * 10))
    plt.axis('off')

    threshold = None
    darkness = bgDarkness             # None=旧行为(较暗)；fig3b_fdr/fig4b_fdr 传 0.3 使与 panel A 一致的浅背景
    for view in views:
        for hemi in hemispheres:
            if hemi == 'left':
                texture = surface.vol_to_surf(nii, fsaverage.pial_left, interpolation='nearest')
                bg_map = fsaverage.sulc_left
                surf = fsaverage.infl_left
            else:
                texture = surface.vol_to_surf(nii, fsaverage.pial_right, interpolation='nearest')
                bg_map = fsaverage.sulc_right
                surf = fsaverage.infl_right
            pos = positions[f"{view}-{hemi}"]
            ax = fig.add_axes(pos, projection='3d')
            ax.set_axis_off()
            ax.tick_params(left=False, right=False, labelleft=False,
                           labelbottom=False, bottom=False)

            surf_map = plotting.plot_surf_stat_map(surf, texture, cmap=cmap,
                                                   hemi=hemi, view=view, bg_on_data=True, bg_map=bg_map,
                                                   bg_img='MNI152',
                                                   colorbar=False,  # 关掉每个子脑各自的 colorbar；循环后统一加一条整合 colorbar
                                                   black_bg=False,
                                                   threshold=threshold,
                                                   darkness=darkness,
                                                   axes=ax
                                                   )
            print(f"resultType-{resultType}: {view} {hemi}-hemi "
                  f"voxelNumber-left{leftHemiNumber}-right{rightHemiNumber} "
                  f"maxValue-{maxNiiValue}")

    # ---- 出版级装饰（面板 B）：面板标号 "B" + 右侧单条整合 colorbar，复刻原发表图 ----
    # 循环里每个子脑的 colorbar 已关掉（colorbar=False），这里统一加一条竖向 colorbar。
    # 数值口径：voxel 值是"最大 t 所在层号"，0=无显著层（白底），1..12=各模型层（gist_rainbow）。
    # 颜色映射必须和脑面一致：nilearn plot_surf_stat_map 默认对称 colorbar（vmin=-vmax），
    # 而本函数的 cmap 是 designCmap 造的 25 色表（前 13 白 + 后 12 彩）。于是层号 v 经
    # Normalize(-12,12) 落到：0→白、1→红 … 12→品红——正是脑面看到的配色。
    # 所以这里用 Normalize(-12, 12)（不能用 0..12：那会让 colorbar 半截发白、且层号→颜色与脑面不符），
    # 再把可视区间裁到 [0,12] 只显示有意义的一半；刻度 0/3/6/9/12。12 = 模型层数，对所有调用方通用。
    import matplotlib.cm as _cm
    from matplotlib.colors import Normalize as _Normalize
    # 面板标号 "B"：图左上角，与上排 lateral 脑顶大致齐平（脑体只占下半幅，不放到顶端空白里）。
    fig.text(0.012, 0.60, 'B', fontsize=80, color='black', ha='left', va='center')
    _sm = _cm.ScalarMappable(cmap=cmap, norm=_Normalize(vmin=-12, vmax=12))
    _sm.set_array([])
    cbar_ax = fig.add_axes([0.85, 0.13, 0.02, 0.40])  # 右侧、纵向大致居中在脑体（脑体 y≈0.07–0.57）
    cbar = fig.colorbar(_sm, cax=cbar_ax, ticks=[0, 3, 6, 9, 12])
    cbar.ax.set_ylim(0, 12)  # 只显示 0..12（脑区实际取值区间），下半截 -12..0 的白不画出
    cbar.ax.tick_params(labelsize=44)
    # 刻度数字与标签都放竖条右侧、自下而上读。
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.set_label('Maximum layer representation', fontsize=52, rotation=90, labelpad=28)

    savePath = f"{scratchResultFolder}/figures/{figureName}{resultType}.pdf"
    savePath = savePath.replace(" ", "_")
    savePath = savePath.replace(":", "_")
    fig.savefig(savePath, transparent=True)
    if copyTo is not None:
        os.makedirs(os.path.dirname(copyTo), exist_ok=True)
        shutil.copy(savePath, copyTo)
        print(f"[copyTo] 已同步到 git 跟踪位置: {copyTo}")
    plotting.show()
    print(f"resultType={resultType}, views={views}")
    print(f"saved at {savePath}")


def plot_empty_brain(views=None):
    from nilearn import datasets
    from nilearn import plotting
    from nilearn import surface
    import numpy as np
    import nibabel as nib
    fsaverage = datasets.fetch_surf_fsaverage()
    nii = nib.load(
        "/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz")
    niiData = nii.get_fdata()
    nii = nib.Nifti1Image(niiData, nii.affine)
    leftHemi = niiData[0:46, :, :]
    leftHemiNumber = np.sum(leftHemi)
    rightHemi = niiData[46:-1, :, :]
    rightHemiNumber = np.sum(rightHemi)

    nonZeroVoxelNumber = np.sum(niiData != 0)
    maxNiiValue = np.max(niiData)
    hemispheres = ['left', 'right']
    cmap = 'bwr'
    fig, axes = plt.subplots(1, 2, figsize=(len(views) * 10, 10), subplot_kw={'projection': '3d'})
    for currHemi, hemi in enumerate(hemispheres):
        if hemi == 'left':
            texture = surface.vol_to_surf(nii, fsaverage.pial_left, interpolation='nearest')
            bg_map = fsaverage.sulc_left
            surf = fsaverage.infl_left
        else:
            texture = surface.vol_to_surf(nii, fsaverage.pial_right, interpolation='nearest')
            bg_map = fsaverage.sulc_right
            surf = fsaverage.infl_right
        for view in views:
            surf_map = plotting.plot_surf_stat_map(surf, texture, cmap=cmap,
                                                   hemi=hemi, view=view, bg_on_data=False,
                                                   bg_map=bg_map,
                                                   bg_img='MNI152',
                                                   black_bg=True,
                                                   colorbar=False,
                                                   threshold=1000000,
                                                   darkness=0.3,
                                                   axes=axes[currHemi]
                                                   )

    plt.subplots_adjust(wspace=0, hspace=0)
    savePath = f"{scratchResultFolder}/figures/emptyBrain.pdf"
    fig.savefig(savePath, transparent=True)
    plotting.show()

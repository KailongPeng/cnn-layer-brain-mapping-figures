"""
生成 "无阈值连续底图 + 黑色显著轮廓(FDR)" 论文图的 NSD 版，配合正文 Figure 4。

回应 Jeff 07.80 反馈（R1P2/R1P4 + 07.80_jeff_overall.md）：
  "For the NSD version of the analysis (i.e. figure 4) we need the
   unthresholded map, akin to S14."
main_text.md 里 Figure 4 caption 原有一句 "See Fig SXX for unthresholded maps"，
在 localize_main.tex 当前版本里已经整句被删掉（不是占位符没填数字，是句子本身
不见了），这张图做出来后要把那句话连同真实编号一起补回 Figure 4 caption。

跟 viz_group_z_outlined.py（合成图版）的关系：同一个设计，只是数据源换成 NSD。
  - 合成图版底图 = analysis_10 存的 Stouffer group_z（逐体素直接可读）。
  - NSD 没有现成存好的等价 group_z 文件（原 meanAcrossSets randomise 输出已被
    scratch60 清理），但 fig4A 的 FDR 分支(display_surface_plot.py 的
    "NSD nonresidual mean+randomise: repStabilityMasked FDR masked tstat" 分支)
    已经实现了从现存的逐-set 逐被试图廉价重建单样本 t 的逻辑
    (_nsd_meanacross_tstat_and_mask)，本脚本直接复用这个重建，不重新发明。
  - 底图换成这个重建出来的全脑单样本 t（不乘任何显著性掩膜，正负都保留）；
    显著性判据也复用同一个 FDR 校正(fdr_significant_mask)，与 Figure 4 正文用的
    是同一套统计，不是另起一套口径。

视角：只画 posterior，跟 S14/Figure 4 一致。

输出：
  paper_figures/paper_figure_outlined_nsd.pdf（+ md_png 预览）

用法（Milgram，localize_paper 根目录，需先 source 好装 nilearn 的 conda 环境）：
  python3 -u rebuttal/analyses/viz_group_z_outlined_nsd.py
"""
import os
import sys
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(__file__))
from config import REBUTTAL_OUTPUT, N_LAYERS

sys.path.append("display_surface_plot")
from display_surface_plot import (
    plot_surface_saveImageReload_pdf,
    _nsd_meanacross_tstat_and_mask,
    fdr_significant_mask,
    scratchResultFolder,   # 模块内部 setupPaths() 算好的路径，_nsd_meanacross_tstat_and_mask
                            # 要跟 display_surface_plot.py 内部其它 NSD 分支用同一个，不能自己另定义。
)

# ── 路径 ─────────────────────────────────────────────────────────────────────
NC_VIZ_DIR    = os.path.join(REBUTTAL_OUTPUT, "noise_ceiling_viz")
PAPER_FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "components")
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

SHOW_LAYERS = list(range(1, N_LAYERS + 1))
FDR_ALPHA   = 0.05   # 与 fig4A FDR 分支一致

OUT_PDF = os.path.join(PAPER_FIG_DIR, "paper_figure_outlined_nsd.pdf")


def contour_mask_path(layer):
    return os.path.join(NC_VIZ_DIR, f"fdr_sig_mask_layer{layer}_nsd_nonresidual.nii.gz")


def save_fdr_mask_niis():
    """逐层重建 tstat + 稳定性掩膜，算 FDR 显著性，存成 NIfTI 供
    display_surface_plot.py 的 contourMaskPathFn 读取。返回每层的 tstat（顺带复用，
    避免重建两次）。"""
    tstats = []
    for layer in SHOW_LAYERS:
        tstat, N, affine, in_mask = _nsd_meanacross_tstat_and_mask(scratchResultFolder, layer)
        df = N - 1
        sig, info = fdr_significant_mask(tstat, in_mask, df=df, q=FDR_ALPHA)
        print(f"  Layer {layer:2d}: N={N} df={df} mask内={info['m']}体素 "
              f"FDR显著={info['n_sig']}体素")
        nib.save(nib.Nifti1Image(sig.astype(np.uint8), affine), contour_mask_path(layer))
        tstats.append(tstat)
    return tstats


def save_md_png_preview(pdf_path):
    """同 viz_group_z_outlined.py：把 PDF 转一份预览 PNG 到 md_png/，缺 pdftoppm 只警告不崩。"""
    import subprocess
    import shutil
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    md_png_dir = os.path.join(os.path.dirname(pdf_path), "..", "previews")
    os.makedirs(md_png_dir, exist_ok=True)
    prefix = os.path.join(md_png_dir, stem)
    try:
        subprocess.run(["pdftoppm", "-png", "-r", "150", pdf_path, prefix], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  WARNING: pdftoppm 预览转换失败({e})，md_png/{stem}.png 需要手动生成")
        return
    rendered = f"{prefix}-1.png"
    final = f"{prefix}.png"
    if os.path.exists(rendered):
        shutil.move(rendered, final)
        print(f"  Saved preview (repo, git-tracked): {final}")
    else:
        print(f"  WARNING: 没找到 {rendered}，md_png/{stem}.png 需要手动生成")


def main():
    print("=== 生成 NSD 版 无阈值单样本 t + 黑色 FDR 显著轮廓 论文图（posterior）===")

    print("\n-- 逐层重建 t / 存 FDR 掩膜 --")
    tstats = save_fdr_mask_niis()

    # 色标范围：t 统计量的量级跟 z-score/R² 都不一样，同样用跨 12 层绝对值 99 分位定。
    all_vals = np.concatenate([t[np.isfinite(t) & (t != 0)].ravel() for t in tstats])
    vmax = float(np.percentile(np.abs(all_vals), 99))
    print(f"\n  RdBu_r symmetric range: [-{vmax:.3f}, {vmax:.3f}]")

    plot_surface_saveImageReload_pdf(
        resultType="NSD nonresidual mean+randomise: group_z unthresholded (FDR outline)",
        cmap='RdBu_r',
        vmax=vmax,
        displayThreshold=None,
        bgDarkness=0.3,
        contourMaskPathFn=contour_mask_path,
        # colorbar 修正（2026-08-08）：nilearn 的 plot_surf_stat_map 只接 vmax，内部一律按
        # 对称区间 [-vmax, +vmax] 渲染、0 落在色标中点（白）；而图上那根 colorbar 原先是另外
        # 用 Normalize(vmin=0, vmax=vmax) 画的，两者差半个量程。这张图上标着 4.08 的浅色区，
        # 真实的 t 其实是 0，会被当成强显著来解读。轴标签 't statistic value' 对这张图本来
        # 就是对的（底图确实是重建出来的单样本 t），这里显式传一遍，不再依赖函数里的默认值。
        # 排查记录见 07.81_raw_R2.md 第四节。
        colorBarVmin=-vmax,
        colorBarLabel='t statistic value',
        # 缓存复用：逐层半球 PNG 的缓存键(_param_tag)不含 colorbar 参数，colorbar 是在装配
        # 阶段才画的，所以只需重新装配。缓存被清了会自动回落到完整渲染，没有坏处。
        forceRun=False,
        views=['posterior'],
        figureName="paper_figure_outlined_nsd_posterior_",
        copyTo=OUT_PDF,
    )

    save_md_png_preview(OUT_PDF)

    print("\n=== 完成 ===")
    print(f"  posterior: {OUT_PDF}")


if __name__ == '__main__':
    main()

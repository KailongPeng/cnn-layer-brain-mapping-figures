"""
生成 "无阈值 raw cross-validated R² 全脑表面图 + 黑色 FDR 显著轮廓" 论文图。

回应 Jeff 07.80 反馈（R1P2 / 07.80_jeff_overall.md）：
  "For the synth images, we show a bar plot of the r squared values. We need
   the whole brain plots, in identical views as S14."
即：S14 现有的 raw R² 部分只有直方图/柱状图(B/C panel)，Jeff 要的是跟 S14 panel A
（paper_figure_outlined.pdf，无阈值 t 统计量 + FDR 黑框，posterior 视角）同样风格的
表面渲染，只是把底图换成 raw R²。这是一张独立的新增补充图，不是改 S14。

底图 = mean_R2 = normalized_R2 * nc_lower（与 viz_raw_R2_negative.py 完全同一算法，
公式见该脚本注释：analysis_10 存的是 normalized_R2 = mean_R2 / nc_lower）。

黑框 = FDR 显著轮廓：直接复用 viz_tstat_outlined_synthesized.py 已经算好、存好的
Figure 3 单样本 t + BH-FDR 掩膜。这套显著性不是基于 R² 本身——刻意如此：
用同一批"论文里说显著"的体素做轮廓，叠在 raw R² 底图上，直接回答 R1P2 的关切
"就算在显著的体素里，raw R² 依然是负的"。

视角：只画 posterior，跟 S14 panel A 一致（S14 没有 lateral，按 Jeff 07.80 的条件
"如果 S14 有 lateral 才需要在这张新图里也加 lateral"，这里也不加）。

依赖：
  - 需要先跑过 viz_tstat_outlined_synthesized.py（存好 t-based FDR 掩膜），
    本脚本直接 import 它的 contour_mask_path()，不重新算 FDR。
  - normalized_R2_layer{N}_synthesized.nii.gz / noise_ceiling_lower_synthesized.nii.gz
    同 viz_raw_R2_negative.py。

输出：
  paper_figures/raw_R2_surface_outlined.pdf（+ md_png 预览）

用法（Milgram，localize_paper 根目录，需先 source 好装 nilearn 的 conda 环境，且要先
跑过 viz_group_z_outlined.py 才有显著性掩膜可用）：
  python3 -u rebuttal/analyses/viz_raw_R2_surface_outlined.py
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(__file__))
from config import REBUTTAL_OUTPUT, N_LAYERS

sys.path.append("display_surface_plot")
from display_surface_plot import plot_surface_saveImageReload_pdf

from viz_tstat_outlined_synthesized import (
    contour_mask_path,
    validate_fdr_mask,
    validate_pdf_output,
)

# ── 路径 ─────────────────────────────────────────────────────────────────────
NC_VIZ_DIR    = os.path.join(REBUTTAL_OUTPUT, "noise_ceiling_viz")
NC_PATH       = os.path.join(REBUTTAL_OUTPUT, "noise_ceiling",
                             "noise_ceiling_lower_synthesized.nii.gz")
PAPER_FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "components")
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

OUT_PDF = os.path.join(PAPER_FIG_DIR, "raw_R2_surface_outlined.pdf")

# 人工指定色标量程。None = 沿用下面那套自动规则（会打印候选表）。
# 想换量程时改这里、重跑即可，不要再往自动规则里塞条件，那条路已经绕过一次弯了，
# 排查记录见 07.81_raw_R2.md。
#
# 定为 3（2026-08-08，Kailong 拍板）。自动规则选出的 97.175 是坏的：全脑 |raw R²| 中位数
# 只有 0.98，而 nilearn 按 [-vmax, +vmax] 对称渲染、白色是 0，于是典型体素只走了到深蓝的
# 1% 路程、渲染成近白。读者看到一个发白的脑会以为 R²≈0，而这张图要说的恰恰是"处处为负"，
# 意思正好反了。vmax=3 让中位体素落在色标 32.8% 处（明显的蓝），代价是 28.9% 的体素饱和；
# 被饱和掉的是重尾（p90=21、p95=44、p99=379、max=54625），来自留一被试方差趋近 0 的病态
# 体素，压成最深蓝仍然读作"非常负"，不损害这张图的结论。
# **caption 里要写明色标做了截断**，否则会被质疑"为什么砍掉三成数据"。
VMAX_OVERRIDE = 3


def mean_R2_path(layer):
    return os.path.join(NC_VIZ_DIR, f"mean_R2_layer{layer}_synthesized.nii.gz")


def validate_mean_R2(layer):
    path = mean_R2_path(layer)
    if not os.path.exists(NC_PATH):
        raise FileNotFoundError(f"Missing synthesized noise ceiling: {NC_PATH}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Layer {layer}: missing mean raw-R² map: {path}")
    nc_nii = nib.load(NC_PATH)
    r2_nii = nib.load(path)
    if r2_nii.shape != nc_nii.shape:
        raise ValueError(
            f"Layer {layer}: mean raw-R² shape {r2_nii.shape} != noise ceiling {nc_nii.shape}"
        )
    if not np.allclose(r2_nii.affine, nc_nii.affine, atol=1e-4):
        raise ValueError(f"Layer {layer}: mean raw-R² affine does not match noise ceiling")
    data = r2_nii.get_fdata()
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Layer {layer}: mean raw-R² map contains non-finite values")
    nc_data = nc_nii.get_fdata()
    brain = np.isfinite(nc_data) & (nc_data > 0)
    values = data[brain]
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Layer {layer}: mean raw-R² map has no finite in-mask values")
    n_positive = int(np.sum(values > 0))
    if n_positive:
        raise ValueError(
            f"Layer {layer}: expected uniformly non-positive group raw R², "
            f"found {n_positive} positive in-mask voxels"
        )


def save_mean_R2_niis(layers=None):
    """算出每层的 mean_R2 并存成 NIfTI，供 display_surface_plot.py 按 resultType
    'synthesized: raw R2 unthresholded (FDR outline)' 直接加载。fail-loud：源文件
    缺失 / 脑掩膜为空 / 算出来的 mean_R2 全非有限，一律 raise，不静默跳过或存空图。"""
    if not os.path.exists(NC_PATH):
        raise FileNotFoundError(f"[门禁 raw-R2-surface] 缺 noise ceiling: {NC_PATH}")
    nc_nii = nib.load(NC_PATH)
    nc_lower = nc_nii.get_fdata()
    affine = nc_nii.affine

    brain_mask = np.isfinite(nc_lower) & (nc_lower > 0)
    if brain_mask.sum() == 0:
        raise ValueError("[门禁 raw-R2-surface] nc_lower>0 的脑掩膜为空，源文件可能不对")

    if layers is None:
        layers = range(1, N_LAYERS + 1)
    for layer in layers:
        nr_path = os.path.join(NC_VIZ_DIR,
                               f"normalized_R2_layer{layer}_synthesized.nii.gz")
        if not os.path.exists(nr_path):
            raise FileNotFoundError(
                f"[门禁 raw-R2-surface] layer{layer} 缺 normalized_R2: {nr_path}")
        nr = nib.load(nr_path).get_fdata()
        if nr.shape != nc_lower.shape:
            raise ValueError(f"[门禁 raw-R2-surface] layer{layer} normalized_R2 形状 "
                             f"{nr.shape} 与 nc_lower {nc_lower.shape} 不一致")
        mean_R2 = nr * nc_lower
        vals = mean_R2[brain_mask & np.isfinite(mean_R2)]
        if vals.size == 0:
            raise ValueError(f"[门禁 raw-R2-surface] layer{layer} 脑掩膜内 mean_R2 全非有限")
        out_path = mean_R2_path(layer)
        nib.save(nib.Nifti1Image(mean_R2.astype('double'), affine), out_path)
        validate_mean_R2(layer)
        print(f"  Layer {layer:2d}: saved {out_path}  "
              f"(mean={vals.mean():.4f} median={np.median(vals):.4f} "
              f"positive={100*(vals>0).mean():.1f}%)")


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
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--layer", type=int, choices=range(1, N_LAYERS + 1), default=None,
        help="Render one layer's PNG cache (Slurm-array stage).",
    )
    group.add_argument(
        "--assemble", action="store_true",
        help="Assemble the final PDF from all layer caches.",
    )
    args = parser.parse_args()

    print("=== 生成 无阈值 raw R² + 黑色 FDR 显著轮廓 论文图（皮层表面渲染，posterior）===")

    if args.layer is not None:
        if not os.path.exists(mean_R2_path(args.layer)):
            save_mean_R2_niis([args.layer])
        validate_mean_R2(args.layer)
        validate_fdr_mask(args.layer)
        return _render(float(VMAX_OVERRIDE), layer=args.layer)

    if args.assemble:
        for layer in range(1, N_LAYERS + 1):
            validate_mean_R2(layer)
            validate_fdr_mask(layer)
        return _render(float(VMAX_OVERRIDE), assemble=True)

    print("\n-- 存每层 mean_R2 --")
    save_mean_R2_niis()

    # 检查显著性掩膜是否已存在（viz_tstat_outlined_synthesized.py 的产物），没有就明确报错，
    # 不要静默用一张空白/全零的轮廓。
    missing = [l for l in range(1, N_LAYERS + 1) if not os.path.exists(contour_mask_path(l))]
    if missing:
        raise FileNotFoundError(
            f"缺 FDR 显著性掩膜（layer {missing}），请先跑一遍 "
            f"viz_tstat_outlined_synthesized.py 把它们存出来，再跑本脚本。")

    # 色标范围：不能用全脑体素的分位数定（已实测：p50=0.98/p90=21/p95=44/p99=379/
    # p99.9=7639/max=54625，从 p90 起就已经被拖走，不是"只有个别离群点"）。这条厚尾
    # 是 cross-validated R² 本身的性质——SS_tot（留一被试的方差）趋近于 0 的体素能把
    # R² 打到远超 [-1, 1] 的量级，R² 定义上没有下界，这些体素的值本身没算错；
    # analysis_10 当初存 normalized_R2 时用的也是同一个 `nc_lower > 0`、没有下限的
    # brain_mask（见 analysis_10_noise_ceiling_viz.py L92/101），所以这条尾巴从源头
    # 就在，median≈-0.982 之所以看着正常，是因为中位数本来就对厚尾稳健，不代表全脑
    # 分位数也稳健。
    #
    # 这张图真正要回答的是"FDR 显著（黑框内）的体素，raw R² 有多负"——黑框是按
    # Figure 3 的单样本 t + FDR 选出来的可靠信号体素，跟 nc_lower→0
    # 的病态体素没有必然关系。所以色标范围改成只用每层黑框内体素的分布来定，而不是
    # 整个 nc_lower>0 的脑掩膜；全脑分位数仍然打印出来存档，但不参与定 vmax。
    nc_lower = nib.load(NC_PATH).get_fdata()
    brain = np.isfinite(nc_lower) & (nc_lower > 0)
    all_vals = []
    sig_vals = []
    for layer in range(1, N_LAYERS + 1):
        r2 = nib.load(mean_R2_path(layer)).get_fdata()
        finite_in_brain = brain & np.isfinite(r2)
        all_vals.append(r2[finite_in_brain])
        sig_mask = nib.load(contour_mask_path(layer)).get_fdata() > 0.5
        sig_vals.append(r2[finite_in_brain & sig_mask])
    all_vals = np.concatenate(all_vals)
    sig_vals = np.concatenate(sig_vals)

    check_percentiles = [50, 90, 95, 99, 99.9, 100]
    pvals = np.percentile(np.abs(all_vals), check_percentiles)
    print("\n  全脑(nc_lower>0) |mean_R2| 分位数（跨 12 层）：" + ", ".join(
        f"p{p}={v:.3f}" for p, v in zip(check_percentiles, pvals)) +
        "  [仅存档，不用于定色标]")

    if sig_vals.size == 0:
        raise ValueError(
            "[门禁 raw-R2-outlier] 12 层加起来一个 FDR 显著体素都没有，没法用显著体素"
            "定色标范围；contour_mask_path 对应的文件可能都是全零，先去检查 "
            "viz_group_z_outlined.py 存的显著性掩膜。")
    spvals = np.percentile(np.abs(sig_vals), check_percentiles)
    print(f"  仅 FDR 显著体素(黑框内, n={sig_vals.size:,}) |mean_R2| 分位数：" + ", ".join(
        f"p{p}={v:.3f}" for p, v in zip(check_percentiles, spvals)))

    # ── 饱和比例表：定 vmax 之前先把"每个候选量程会牺牲多少体素"摆出来 ───────────
    # 2026-08-06 复盘（见 07.81_raw_R2.md）：上一版按 nc_lower 下限自动选出的
    # vmax=97.175 是坏的。全脑 |raw R²| 中位数只有 0.98，量程 97 意味着中位体素只偏离
    # 色标中心 1%，在 RdBu_r 上就是白色；12 层之间那点差异被压到 8bit 渲染里只剩
    # 1.1 个灰阶（本地实测，diag_s16_panel_pixels.py），肉眼当然看不出层间差别。
    # 所以这里先打表，让"量程 vs 饱和"这个取舍是看得见的，而不是埋在一个自动规则里。
    print("\n  候选 vmax 的取舍（全脑 12 层合并）：")
    print(f"     {'vmax':>10}{'饱和体素占比':>14}{'中位体素占色标':>16}")
    for cand in (1, 2, 3, 5, 8, 15, 30, 60, 97.175):
        sat = 100 * (np.abs(all_vals) > cand).mean()
        print(f"     {cand:>10.3f}{sat:>13.2f}%"
              f"{100 * np.median(np.abs(all_vals)) / cand:>15.1f}%")

    if VMAX_OVERRIDE is not None:
        vmax = float(VMAX_OVERRIDE)
        print(f"\n  使用人工指定 VMAX_OVERRIDE = {vmax}")
        return _render(vmax)

    median_mag = float(np.median(np.abs(sig_vals)))
    p99_sig = float(np.percentile(np.abs(sig_vals), 99))
    if p99_sig > 20 * max(median_mag, 1e-6):
        # 第一轮修复（只挑 FDR 显著体素）已经证实不够：实测显著体素子集里 99 分位
        # (283.25) 依然是中位数(5.76)的 49 倍。说明"显著"(group_z，不依赖 nc_lower)
        # 跟"nc_lower 够大、R² 不会被 SS_tot→0 打爆"是两件独立的事，不能互相担保。
        # 【2026-08-06 更正：下面这条理由是错的，保留原文只为留下排查轨迹。】原文认为
        # "nc_lower 越接近 0，held-out 方差越接近 0，R² 越容易被打到极端值"，据此拿
        # nc_lower 下限当抓手压尾巴。但 mean_R2 = normalized_R2 × nc_lower，量级是随
        # nc_lower 同向走的，不是反向：实测加了 nc_lower>=0.15 之后，显著体素的中位
        # |R²| 反而从 5.76 升到约 13.3。所以这条支路选错了工具，最后是靠把保留比例门槛
        # 从 30% 松到 5% 才"通过"的，选出来的 vmax=97.175 相对于全脑中位数 0.98 大了两个
        # 数量级，正是 S16 十二层看着雷同的主因之一。完整复盘见 07.81_raw_R2.md，
        # 定量程请改用文件顶部的 VMAX_OVERRIDE。下面对一组候选
        # 下限做扫描，每个下限打印："显著体素子集里再过滤 nc_lower>=floor 之后剩多少
        # 体素、还剩多大比例、|mean_R2| 的 99 分位/中位数之比"，自动挑一个能把比值压到
        # 10 倍以内的最小下限。
        #
        # 保留比例的下限放得很松（5%），只是防止边界情况(比如只剩几个体素)，不是真的
        # 要求"保留大部分体素"：这个 floor 只用来决定 vmax(色标范围)，不影响实际渲染
        # 出的体素范围本身，被下限排除的体素照样会被画出来，只是颜色会饱和/裁到色标
        # 最深的一端(标准做法)。实测过 30% 的门槛太严：扫过 0~0.5 全部候选，能把比值
        # 压到 10 倍以内的下限(0.15，7.3x)保留比例只有 21.9%，卡在 30% 门槛之外，才
        # 把门槛松到 5%。
        print(f"\n  [门禁触发] 显著体素子集的 99 分位/中位数比值={p99_sig / max(median_mag, 1e-9):.0f}"
              f"，超过 20 倍上限。改成对 nc_lower 加下限，扫一遍候选下限看效果：")
        candidate_floors = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        sig_nc = []
        for layer in range(1, N_LAYERS + 1):
            r2 = nib.load(mean_R2_path(layer)).get_fdata()
            finite_in_brain = brain & np.isfinite(r2)
            sig_mask = nib.load(contour_mask_path(layer)).get_fdata() > 0.5
            keep = finite_in_brain & sig_mask
            sig_nc.append(nc_lower[keep])
        sig_nc = np.concatenate(sig_nc)
        chosen_floor = None
        for floor in candidate_floors:
            keep = sig_nc >= floor
            n_keep = int(keep.sum())
            frac_keep = n_keep / sig_vals.size
            if n_keep < 2:
                print(f"    nc_lower>={floor}: 剩 {n_keep} 个体素，太少，跳过")
                continue
            v = np.abs(sig_vals[keep])
            med = float(np.median(v))
            p99f = float(np.percentile(v, 99))
            ratio = p99f / max(med, 1e-9)
            print(f"    nc_lower>={floor}: 剩 {n_keep:,}/{sig_vals.size:,} "
                  f"({100*frac_keep:.1f}%)，median={med:.4f}，p99={p99f:.4f}，"
                  f"p99/median={ratio:.1f}x")
            if chosen_floor is None and ratio <= 10 and frac_keep >= 0.05:
                chosen_floor = floor
        if chosen_floor is None:
            raise ValueError(
                "[门禁 raw-R2-outlier] 扫过所有候选 nc_lower 下限，没有一个能同时把"
                "显著体素子集的 99分位/中位数 压到 10 倍以内、且保留至少 5% 的显著"
                "体素。看上面打出来的完整表格，人工挑一个下限（或者换别的稳健统计量，"
                "比如直接对 mean_R2 做固定上限裁剪），别再往这个函数里加自动逻辑了。")
        print(f"  自动选定 nc_lower 下限 = {chosen_floor}")
        keep = sig_nc >= chosen_floor
        sig_vals = sig_vals[keep]
        median_mag = float(np.median(np.abs(sig_vals)))
        p99_sig = float(np.percentile(np.abs(sig_vals), 99))
    return _render(p99_sig)


def _render(vmax, layer=None, assemble=False):
    # nilearn 的 plot_surf_stat_map 只接 vmax，内部一律按对称区间 [-vmax, +vmax] 渲染，
    # 0 落在色标中点(白)。所以 colorbar 也必须传 -vmax，否则画出来的刻度跟表面配色差了
    # 半个量程：旧版 colorbar 标 0.00~97.17，而这张图的数据 100% 为负，读者按 colorbar
    # 读到的"近白 ≈ 48.6"其实是 R²≈0。标签同理，写死的 't statistic value' 对这张画
    # raw R² 的图是错的。
    print(f"\n  RdBu_r symmetric range: [-{vmax:.3f}, {vmax:.3f}]（colorbar 同区间）")

    common = dict(
        resultType="synthesized: raw R2 unthresholded (FDR outline)",
        cmap='RdBu_r',
        vmax=vmax,
        displayThreshold=None,     # 无阈值：全脑都显色
        bgDarkness=0.3,
        contourMaskPathFn=contour_mask_path,
        views=['posterior'],
        # Use a new cache namespace: old PNGs contain Stouffer-based outlines.
        figureName="raw_R2_surface_outlined_tstatFDR_posterior_",
        copyTo=OUT_PDF,
        colorBarVmin=-vmax,
        colorBarLabel='raw cross-validated $R^2$',
    )

    if assemble:
        plot_surface_saveImageReload_pdf(
            **common, forceRun=False, assembleOnly=True
        )
    elif layer is not None:
        plot_surface_saveImageReload_pdf(
            **common, forceRun=True, layers=[layer], renderOnly=True
        )
        print(f"Saved PNG cache for layer {layer}")
        return
    else:
        plot_surface_saveImageReload_pdf(**common, forceRun=True)

    validate_pdf_output(OUT_PDF)
    save_md_png_preview(OUT_PDF)

    print("\n=== 完成 ===")
    print(f"  posterior: {OUT_PDF}")


if __name__ == '__main__':
    main()

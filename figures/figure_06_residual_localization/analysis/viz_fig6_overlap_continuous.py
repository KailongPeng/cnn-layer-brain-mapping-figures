# Usage:
#   python3 -u rebuttal/BZ/rebuttal_analyses/viz_fig6_overlap_continuous.py
# Run from localize_paper root.

"""
Fig 6 C/D/E 的连续度量重画（方案B per-set + 方案② residual-vs-rawNSD）。

Fig 6 是残差法的定位图，5 个面板：
  A/B  脑表面图（残差法，TFCE→FDR），走 nilearn 那条独立线，**不在本脚本**。
  C    残差 per-set 层间相关矩阵（overlap_pearson_nsd_residual_perset.npy）。
  D/E  残差观测线 vs 原始-NSD per-set 零分布（和 Fig 5 D/E 用同一条 raw-NSD null）。

方案②：不与合成图正面比（R2P3 说那不成立），而是比"残差 vs 未残差化自然图"——
即残差化是否把层间相关显著压低。残差观测值（≈ −0.014 / −0.108）落在 raw-NSD 零分布
（中心 0.245 / 0.375）全部之下，p < 0.0002。

复用 viz_fig5 的 panel_matrix / panel_hist / _pair_schematic_rgba / rebuild_null /
load_matrix / off_diag_mean / adjacent_mean，只在这里接线。fail-loud：输入缺失、
null 重建漂移都 raise，不静默出图。
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_fig5_overlap_continuous import (
    CONT_DIR, PAPER_FIG_DIR, _require, load_matrix, rebuild_null,
    off_diag_mean, adjacent_mean, panel_matrix, panel_hist,
    FONT_TICK, FONT_PANEL, FONT_COLORBAR,
)

# Fig 6 D/E 的 null = 原始-NSD 的 per-set bootstrap，与 Fig 5 D/E 用的**同一份**。
RAW_NSD_JSON = "bootstrap_nsd_pearson_per-dataset.json"
OBS_LABEL = 'Residualized NSD'

# Figure 6 的 C-E 是三栏，整张 Figure 6 又会统一缩放到与 Figure 5 相同的论文栏宽。
# 本脚本固定输出 19 inch（1368 pt）宽，组合时缩放到约 2827 pt；Figure 5 本身是
# 869 pt 宽。按最终物理字号反推，1368/2827 与 869/2857 的比值要求源字号乘约
# 1.59，因此取 1.60。只比较源代码里的 fontsize 会漏掉这一步组合缩放。
FIG6_FONT_SCALE = 1.60


def load_raw_nsd_bootstrap():
    path = _require(os.path.join(CONT_DIR, RAW_NSD_JSON),
                    "原始-NSD bootstrap（作 Fig 6 D/E 的 null，与 Fig 5 同一份）")
    with open(path) as f:
        return json.load(f)


def p_report(boot, observed):
    """与 analysis_04b 同口径：p = mean(boot < observed)，单尾左。残差远低于 raw-NSD → p 触底。"""
    p = float(np.mean(boot < observed))
    res = 1.0 / len(boot)
    if p < res:
        return f"< {res:.4f}"
    if p > 1 - res:
        return f"> {1 - res:.4f}"
    return f"= {p:.4f}"


def main():
    # ── 6C：残差 per-set 相关矩阵 ────────────────────────────────────────────────
    m_res = load_matrix('nsd_residual_perset')
    obs_off = off_diag_mean(m_res)
    obs_adj = adjacent_mean(m_res)

    # ── 6D/6E 的 null：原始-NSD per-set bootstrap（Fig 5 同一条）──────────────────
    raw = load_raw_nsd_bootstrap()
    n_resamples = raw.get('n_resamples', 5000)
    seed = raw.get('seed', 0)
    boot_off = rebuild_null(raw['set_level_offdiag'], n_resamples, seed)
    boot_adj = rebuild_null(raw['set_level_adjacent'], n_resamples, seed)

    # fail-loud：重建的 raw-NSD null 均值必须与 JSON 记录一致（同 viz_fig5 的漂移门），
    # 否则说明作图用的 seed/口径与分析时不一致，不能作图。
    for label, boot, rec in [('off', boot_off, raw['results']['off_diagonal']),
                             ('adj', boot_adj, raw['results']['adjacent'])]:
        drift = abs(np.mean(boot) - rec['boot_mean'])
        if drift > 0.02:
            raise RuntimeError(
                f"[口径漂移] Fig6 {label} 重建 raw-NSD null 均值 {np.mean(boot):.4f} 与 JSON "
                f"{rec['boot_mean']:.4f} 差 {drift:.4f}；请对齐 seed/n_resamples/set 级值。")
        print(f"[核对] Fig6 {label} raw-NSD null 均值 重建={np.mean(boot):.4f} vs JSON={rec['boot_mean']:.4f} OK")

    p_off = p_report(boot_off, obs_off)
    p_adj = p_report(boot_adj, obs_adj)
    print(f"\n6C 残差矩阵: off-diag={obs_off:.4f}, adjacent={obs_adj:.4f}")
    print(f"6D 全体层对: 残差 {obs_off:.4f} vs raw-NSD null 中心 {np.mean(boot_off):.4f}  p {p_off}")
    print(f"6E 相邻层对: 残差 {obs_adj:.4f} vs raw-NSD null 中心 {np.mean(boot_adj):.4f}  p {p_adj}")

    # ── 画 C/D/E 三联 ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 5.6))
    # 固定绝对位置，给 C 的 colorbar label 与 D/E 的 y-axis labels 留出明确间距。
    # GridSpec 在字号变化时会重新分配间距，曾导致 Pearson r 与 Count 互相覆盖。
    axes = [fig.add_axes([0.055, 0.20, 0.205, 0.70]),
            fig.add_axes([0.405, 0.20, 0.245, 0.70]),
            fig.add_axes([0.735, 0.20, 0.245, 0.70])]
    cbar_ax = fig.add_axes([0.280, 0.20, 0.015, 0.70])

    im_c = panel_matrix(axes[0], m_res, font_scale=FIG6_FONT_SCALE)
    cbar = fig.colorbar(im_c, cax=cbar_ax)
    cbar.set_label('Pearson r', fontsize=FONT_COLORBAR * FIG6_FONT_SCALE)
    cbar.ax.tick_params(labelsize=FONT_TICK * FIG6_FONT_SCALE)
    panel_hist(axes[1], boot_off, obs_off, p_off, '', 'all',
               obs_label=OBS_LABEL, font_scale=FIG6_FONT_SCALE,
               legend_anchor=(0.10, 1.0),
               schematic_bounds=(0.70, 0.40, 0.27, 0.30))
    panel_hist(axes[2], boot_adj, obs_adj, p_adj, '', 'adjacent',
               obs_label=OBS_LABEL, font_scale=FIG6_FONT_SCALE,
               legend_anchor=(0.10, 1.0),
               schematic_bounds=(0.77, 0.40, 0.20, 0.30))
    # 字母用 figure 坐标锚定在固定的页内上边距。这里不能沿用 Figure 5 的
    # axes-relative y=1.045：C-E 使用固定页面尺寸，外置文字会被页面顶部裁掉。
    for ax, label in zip(axes, 'CDE'):
        fig.text(ax.get_position().x0 - 0.025, 0.985, label,
                 fontsize=FONT_PANEL * FIG6_FONT_SCALE,
                 fontweight='normal', ha='left', va='top')
    # 不加页面级 suptitle：旧版这里印过"Figure 6C-E ... R1-P5"这行内部复核标签，会带着
    # 审稿人编号漏进最终发表图。面板 C/D/E 已各自带字母标签，和 fig3/fig4 的合成页一样，
    # 标题信息交给 tex 图注，页面本身不需要总标题。

    out_pdf = os.path.join(PAPER_FIG_DIR, "fig6_overlap_continuous_CDE.pdf")
    out_png = os.path.join(PAPER_FIG_DIR, "md_png", "fig6_overlap_continuous_CDE.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    # 固定页面尺寸而不使用 bbox_inches='tight'。组合脚本据此计算缩放，避免文字变化
    # 让 C-E 的 PDF 宽度漂移，进而再次改变最终视觉字号。
    fig.savefig(out_pdf, transparent=True)
    fig.savefig(out_png, dpi=150)
    plt.close()
    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_png}")

    # 单面板各出一份，方便论文最终排版单独取用
    for tag, fn in [('C', lambda ax: panel_matrix(ax, m_res)),
                    ('D', lambda ax: panel_hist(ax, boot_off, obs_off, p_off, '', 'all',
                                                obs_label=OBS_LABEL)),
                    ('E', lambda ax: panel_hist(ax, boot_adj, obs_adj, p_adj, '', 'adjacent',
                                                obs_label=OBS_LABEL))]:
        f1, a1 = plt.subplots(figsize=(6, 5))
        fn(a1)
        p1 = os.path.join(PAPER_FIG_DIR, f"fig6{tag}_continuous.pdf")
        f1.savefig(p1, bbox_inches='tight', transparent=True)
        plt.close()
        print(f"Saved: {p1}")

    print("\n=== 填进正文的数字（Fig 6，方案B+②）===")
    print(f"  6C 残差矩阵 off-diag={obs_off:.4f} / adjacent={obs_adj:.4f}")
    print(f"  6D (all pairs) : 残差 {obs_off:.3f} vs raw-NSD {np.mean(boot_off):.3f}; p {p_off}")
    print(f"  6E (adjacent)  : 残差 {obs_adj:.3f} vs raw-NSD {np.mean(boot_adj):.3f}; p {p_adj}")


if __name__ == '__main__':
    main()

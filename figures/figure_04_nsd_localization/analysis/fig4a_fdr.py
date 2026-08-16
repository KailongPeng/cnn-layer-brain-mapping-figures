# 方案 B1：用 FDR（参数 t→p → Benjamini-Hochberg）出 Fig 4A（NSD 自然图）的后视表面图，
# 取代原来的 TFCE。回应 R1-P4，与 fig3a_fdr.py 完全同源，只换成 NSD 的数据分支。
#
# 只改校正：统计量仍是那些图的单样本 t，稳定性掩膜仍是 NSD 主图那张
# （project 永久盘的 repStability_RSA_sl/randomise/rad3/allSet/__tfce_corrp_tstat1.nii.gz > 0.95），
# 逐层各自校正。
#
# vmax=14：诊断 diag_fig4_layer_separability.py 实测，NSD 每层 t 的 max 高达 22~66，
# 而 vmax=8 把 t>8 的一大片全饱和成同一档黄，把逐层真实的强度差异（连续 t 跨层
# off-diag Pearson=0.53，数据是可分的）压平了——这正是 fig4 看起来 12 层雷同的主因。
# vmax=22（旧 TFCE 值）又反过来把广泛的中等 t（t≈2~6）弄成暗紫/黑、看着像稀疏小簇。
# 14 取自 FDR-masked 的 99 分位附近（三值 8/auto/22 对比图见 paper_figures/
# diag_fig4_fdrmasked_t_montage_*.png）：高 t 分级不饱和、中等 t 又不发暗，两头兼顾。
# 注：fig4 用 14、fig3 仍用 8，两图色标不再共用——fig3（t 偏低、选择性）与 fig4（t 到 66）
# 本就该用各自合适的量程，各自 colorbar 标清即可，不为“可比”牺牲 fig4 的“可分”。见 lastEmail #23。
#
# 三种跑法（同 fig3a_fdr.py）：
#   串行全跑（慢）：           python3 fig4/fig4a_fdr.py
#   并行 render + 装配：
#       sbatch fig4/fig4a_fdr_array.sh          # 每个任务 --layer $SLURM_ARRAY_TASK_ID
#       python3 fig4/fig4a_fdr.py --assemble    # 12 张 PNG 都在了，读缓存拼图（秒级）
#
# 任何输入文件缺失都会在 display_surface_plot 的 [门禁 FDR] 带路径直接 raise，不静默出错图。
import sys
import argparse

sys.path.append("display_surface_plot")
from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap


def main():
    ap = argparse.ArgumentParser(
        description="Fig 4A NSD 图的 FDR（参数 t→p→BH）后视表面图，取代 TFCE")
    ap.add_argument("--layer", type=int, default=None,
                    help="只渲染这一层的 PNG（并行 render 阶段用，1-12）；不给=全 12 层")
    ap.add_argument("--assemble", action="store_true",
                    help="装配阶段：读所有已缓存的层 PNG 拼成最终图，不重渲染")
    args = ap.parse_args()

    # 纯 inferno（不用 designCmap 的白下半段）+ displayThreshold=0.01：niiData=sig×t 已是 FDR 阈过的，
    # 让 0 透明、所有显著体素(含 t∈[2,4) 的广泛区)都上色，避免白下半段把近一半显著区藏成白色。
    common = dict(
        resultType="NSD nonresidual mean+randomise: repStabilityMasked FDR masked tstat",
        views=['posterior'],
        cmap='inferno',
        colorBar=True,
        vmax=14,
        figureName="nsd_FDR",
        displayThreshold=0.01,
        # 背景调浅到 0.3，与空层 plot_empty_brain_ 分支(固定 0.3)一致，整体回到老图干净浅背景。
        # NSD 无空层，此处主要为与 fig3 观感统一（用户选的方案 2）。
        bgDarkness=0.3,
        # 装配后自动同步到仓库 git 跟踪位置，免得手动 cp、免得仓库副本变旧。
        copyTo="rebuttal/BZ/paper_figures/fig4A_nsd_FDR_posterior.pdf",
    )

    if args.assemble:
        plot_surface_saveImageReload_pdf(**common, forceRun=False)
    elif args.layer is not None:
        plot_surface_saveImageReload_pdf(**common, forceRun=True,
                                         layers=[args.layer], renderOnly=True)
    else:
        plot_surface_saveImageReload_pdf(**common, forceRun=True)


if __name__ == '__main__':
    main()

# 方案 B1：用 FDR(参数 t→p → Benjamini-Hochberg)出 Fig 6A(NSD 残差模型)的后视表面图，
# 取代原来的 TFCE。与 fig4a_fdr.py 完全同源，只换成残差模型的重建
# (`_nsd_residual_meanacross_tstat_and_mask`：回归掉共享跨层方差后剩下的层特异表征)。
#
# vmax=14 是从 fig4a_fdr.py 沿用的起始值，不是残差模型自己的实测结果——残差模型把共享方差
# 回归掉后 t 的量级可能和非残差差很多，正式出图前务必先跑 fig6ab/diag_fig6_vmax.py 核实
# t 的 max/p99 落在哪个区间，vmax 明显不对就按诊断结果改这里的值再出图。
#
# 三种跑法(同 fig4a_fdr.py)：
#   串行全跑(慢)：           python3 fig6ab/fig6a_fdr.py
#   并行 render + 装配：
#       sbatch fig6ab/fig6a_fdr_array.sh          # 每个任务 --layer $SLURM_ARRAY_TASK_ID
#       python3 fig6ab/fig6a_fdr.py --assemble    # 12 张 PNG 都在了，读缓存拼图(秒级)
#
# 任何输入文件缺失都会在 display_surface_plot 的 [门禁 FDR] 带路径直接 raise，不静默出错图。
import sys
import argparse

sys.path.append("display_surface_plot")
from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap


def main():
    ap = argparse.ArgumentParser(
        description="Fig 6A NSD 残差模型图的 FDR(参数 t→p→BH)后视表面图，取代 TFCE")
    ap.add_argument("--layer", type=int, default=None,
                    help="只渲染这一层的 PNG(并行 render 阶段用，1-12)；不给=全 12 层")
    ap.add_argument("--assemble", action="store_true",
                    help="装配阶段：读所有已缓存的层 PNG 拼成最终图，不重渲染")
    ap.add_argument("--vmax", type=float, default=14,
                    help="色标上界；正式出图前先用 diag_fig6_vmax.py 核实这个值(默认沿用 fig4 的 14)")
    args = ap.parse_args()

    # 纯 inferno(不用 designCmap 的白下半段)+ displayThreshold=0.01：niiData=sig×t 已是 FDR 阈过的，
    # 让 0 透明、所有显著体素都上色；bgDarkness=0.3 与 fig3/fig4 统一的浅背景口径一致。
    common = dict(
        resultType="NSD residual mean+randomise: repStabilityMasked FDR masked tstat",
        views=['posterior'],
        cmap='inferno',
        colorBar=True,
        vmax=args.vmax,
        figureName="nsd_residual_FDR",
        displayThreshold=0.01,
        bgDarkness=0.3,
        copyTo="rebuttal/BZ/paper_figures/fig6A_nsd_residual_FDR_posterior.pdf",
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

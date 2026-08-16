# 方案 B1：用 FDR（参数 t→p → Benjamini-Hochberg）出 Fig 3A（合成图）的后视表面图，
# 取代原来的 TFCE。回应 R1-P4（TFCE 把 layer 1 弱信号连成巨簇、蔓延进非视觉皮层）。
#
# 只改校正：统计量仍是那 30 张图的单样本 t，稳定性掩膜仍是主图那张
# （repStability_RSA_sl/rad3/__tfce_corrp_tstat1.nii.gz > 0.95，同 visualize.py），逐层各自校正。
# FDR 的数学与本地蒙特卡洛验证见 lastEmail/21_ 与 docs/adr/0001。
#
# 三种跑法：
#   串行全跑（慢，~10 分钟，行为同以前）：
#       python3 fig3a/fig3a_fdr.py
#   并行：先用 job array 每层各渲一张 PNG（12 个任务并发），再装配：
#       sbatch fig3a/fig3a_fdr_array.sh          # 每个任务 --layer $SLURM_ARRAY_TASK_ID
#       python3 fig3a/fig3a_fdr.py --assemble    # 12 张 PNG 都在了，读缓存拼图（秒级）
#
# 任何输入文件缺失都会在 display_surface_plot 的 [门禁 FDR] 带路径直接 raise，不静默出错图。
import sys
import argparse

sys.path.append("display_surface_plot")
from display_surface_plot import plot_surface_saveImageReload_pdf, designCmap


def main():
    ap = argparse.ArgumentParser(
        description="Fig 3A 合成图的 FDR（参数 t→p→BH）后视表面图，取代 TFCE")
    ap.add_argument("--layer", type=int, default=None,
                    help="只渲染这一层的 PNG（并行 render 阶段用，1-12）；不给=全 12 层")
    ap.add_argument("--assemble", action="store_true",
                    help="装配阶段：读所有已缓存的层 PNG 拼成最终图，不重渲染")
    args = ap.parse_args()

    # 纯 inferno + displayThreshold=0.01：niiData=sig×t 已是 FDR 阈过的，让 0 透明、所有显著体素
    # (含低 t)都上色，避免 designCmap 白下半段(≈vmax/2 隐含阈值)把中低 t 显著区藏成白色。
    # 与 fig4a 同一显示口径，保证 fig3(选择性) 与 fig4(广泛) 可直接对比。
    common = dict(
        resultType="synthesized: repStabilityMasked FDR masked tstat",
        views=['posterior'],
        cmap='inferno',
        colorBar=True,
        vmax=8,
        figureName="synthesized_FDR",
        displayThreshold=0.01,
        # 背景调浅到 0.3，与空层 plot_empty_brain_ 分支(固定 0.3)一致，消掉"空层浅、非空层暗"的
        # 不一致，整体回到老图干净浅背景，又保留纯 inferno 让中等 t 可见（用户选的方案 2）。
        bgDarkness=0.3,
        # 装配后自动同步到仓库 git 跟踪位置，免得手动 cp、免得仓库副本变旧。
        copyTo="rebuttal/BZ/paper_figures/fig3A_synthesized_FDR_posterior.pdf",
    )

    if args.assemble:
        # 装配：全 12 层、forceRun=False（PNG 已缓存则跳过渲染）、正常拼图。
        plot_surface_saveImageReload_pdf(**common, forceRun=False)
    elif args.layer is not None:
        # 并行 render：只渲染这一层的 PNG，渲完即返回、跳过拼图。
        plot_surface_saveImageReload_pdf(**common, forceRun=True,
                                         layers=[args.layer], renderOnly=True)
    else:
        # 默认：串行全跑（渲染 12 层 + 拼图），行为同以前。
        plot_surface_saveImageReload_pdf(**common, forceRun=True)


if __name__ == '__main__':
    main()

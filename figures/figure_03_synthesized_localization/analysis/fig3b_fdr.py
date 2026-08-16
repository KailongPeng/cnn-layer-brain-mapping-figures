# 方案 B1：Fig 3B（合成图 consolidated "maximally fit layer" 胜者通吃图）改用 FDR 门限，取代 TFCE。
# 回应 R1-P4：panel B 的 lateral/medial 能看到前后轴，layer 1 的地盘在前部有没有退掉一目了然。
#
# 与 fig3a_fdr.py 同源的 FDR：每层 t→参数 p→BH（q<0.05）、在稳定性掩膜内、逐层各自校正，
# 只把 panel B argmax 的每层输入从"TFCE 显著×t"换成"FDR 显著×t"，其余（argmax、背景清零、
# 三视角渲染）全不动。FDR 数学与本地蒙特卡洛验证见 lastEmail/21_ 与 docs/adr/0001。
#
# 跑法（三视角一次出一张 PDF，串行；panel B 只有 6 个子脑，不像 panel A 那样并行）：
#     source env_setup_glmsingle.sh
#     python3 -u fig3b/fig3b_fdr.py
#
# 任何输入文件缺失都会在 display_surface_plot 的 [门禁 FDR] 带路径直接 raise，不静默出错图。
import sys

sys.path.append("display_surface_plot")
from display_surface_plot import plotMaximalLayer_pdf, designCmap

# 和 TFCE 版 fig3b() 一样的 12 色 rainbow 色标，保证除"校正方法"外观感一致、可直接对比。
continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)
plotMaximalLayer_pdf(resultType="synthesized: maximally fit layer FDR",
                     views=['lateral', 'medial', 'posterior'],
                     cmap=continuous_Cmap,
                     # 背景调浅到 0.3，与 panel A(fig3a_fdr)一致的浅背景。
                     bgDarkness=0.3,
                     # 存 scratch 后自动同步到仓库 git 跟踪位置。
                     copyTo="rebuttal/BZ/paper_figures/fig3B_synthesized_FDR_3view.pdf")
print("done")

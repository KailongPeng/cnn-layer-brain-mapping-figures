# 方案 B1：Fig 6B(NSD 残差模型 consolidated "maximally fit layer" 胜者通吃图)改用 FDR 门限，
# 取代 TFCE。与 fig4b_fdr.py 完全同源，只换成残差模型的数据分支
# (NSD 残差模型的重建 t + 与非残差共用的稳定性掩膜，逐层 t→参数 p→BH)。
# 三视角(lateral/medial/posterior) 一次出一张 PDF，串行(panel B 无 PNG 缓存，天然无过期风险)。
#
#     source env_setup_glmsingle.sh
#     python3 fig6ab/fig6b_fdr.py
#
# 任何输入文件缺失都会在 display_surface_plot 的 [门禁 FDR] 带路径直接 raise，不静默出错图。
import sys

sys.path.append("display_surface_plot")
from display_surface_plot import plotMaximalLayer_pdf, designCmap

continuous_Cmap = designCmap(cmap='gist_rainbow', colors=12)
plotMaximalLayer_pdf(resultType="NSD residual: maximally fit layer FDR",
                     views=['lateral', 'medial', 'posterior'],
                     cmap=continuous_Cmap,
                     # 背景调浅到 0.3，与 fig6A(fig6a_fdr)一致的浅背景。
                     bgDarkness=0.3,
                     # 存 scratch 后自动同步到仓库 git 跟踪位置。
                     copyTo="rebuttal/BZ/paper_figures/fig6B_nsd_residual_FDR_3view.pdf")
print("done")

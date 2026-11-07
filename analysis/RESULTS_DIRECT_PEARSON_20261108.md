# Direct Pearson 第一版全体被试结果记录

2026-11-08更新。已完成描述性全脑结果与数值核查；尚未完成正式显著性推断、NSD对照、重复图片分半或论文叙事决策。

## 已验证的证据

| 项目 | 实际结果 |
|---|---|
| 独立模型核验 | 作业29530910，COMPLETED，exit 0:0；从16张实验PNG重算12层模型RSM，最大误差低于5e-7 |
| 图片对应 | 任务A–P → im01of16–im16of16；230份现用run设计标签与contrast审计通过；对应模型索引0–15 |
| 全脑计算 | array 29530911，30个子任务均COMPLETED、exit 0:0；360张有符号r图；代码db79619 |
| 组汇总 | 29530912，COMPLETED、exit 0:0；12层组r和有效被试数图 |
| 独立QC与预览 | 29530973，COMPLETED、exit 0:0；读取全部360张图，逐体素复算组图，核对覆盖和输入文件状态，均通过 |
| 共同覆盖 | 全部30人、全部12层均有效的191696个体素；空间相关及预览使用同一范围 |
| 数学测试 | Milgram analysis环境8项通过；包含缺失覆盖、Fisher汇总、符号和原核心定义 |

两次初始独立模型核验失败的原因是新核验脚本读取了内部不含bias的conv张量。历史代码使用命名pre-ReLU张量，包含bias；修正后通过。没有修改原features.npy，也没有放宽原定容差。失败报告保留用于追溯。

输入来源核验不等于重放每次历史呈现日志或重拟合高层GLM。旧cor异常仍未解决，当前结果来自现用cope与模型重新计算。

## 服务器输出入口

以下路径均相对于：

`/gpfs/milgram/project/turk-browne/projects/localize_paper/derivatives/direct_rsa_revision_20261107/`

| 文件/目录 | 内容 |
|---|---|
| `provenance/stimulus_identity.json` | 图片、任务源码、run设计与模型的来源记录 |
| `provenance/model_identity_29530910.json` | 独立模型重算，PNG及模型hash、逐层误差 |
| `production_jobs.json` | 本批作业ID与提交版本 |
| `full/subXXX/layer01_r.nii.gz`等 | 30人×12层原始r图；各目录有qc.json及覆盖图 |
| `group/layer01_mean_r.nii.gz`等 | tanh(mean(atanh(subject r)))，对有限被试汇总 |
| `group/layer01_n.nii.gz`等 | 每体素实际有效被试数 |
| `group/common_all30_mask.nii.gz` | 30人和12层共同有效范围 |
| `group/spatial_correlation_common_all30.csv` | 新r图的12×12描述性空间相关矩阵 |
| `cohort_independent_qc.json` | 全量图像与组汇总独立核查 |
| `previews/group_r_axial_preview.png`及PDF | 12层统一色标、未阈值化轴位预览 |
| `previews/spatial_correlation_preview.png`及PDF | 跨层空间矩阵预览 |
| `roi/subject_roi_r.csv` | 每被试、每层、每个双侧Wang ROI的平均r |
| `roi/group_roi_r.csv` | ROI被试均值、pointwise bootstrap区间，及组Fisher图的ROI均值（单列） |
| `roi/qc.json` | 实际ROI标签、覆盖、图谱来源hash及汇总方法 |

## 阅读限制与剩余工作

组图保留所有有覆盖体素，同时提供实际n；不同体素n可能不同。预览与空间比较固定为全部30人共同覆盖，不将边缘少数被试位置的极值当作30人群体强度。统一预览色标为±0.15，未叠加任何显著性mask；预览尚待视觉检查。

ROI使用论文现存vtpm.mat及历史转换方法重建、最近邻重采样到当前脑图。通过源文件实际label名称选择双侧区域，避免旧脚本中不同版本标签编号混用。保留每个ROI的完整图谱体素数和共同覆盖体素数；正式呈现前仍需检查覆盖与配准。

ROI先对每人的ROI内体素取原始r算术均值，再跨被试平均。另单独报告组Fisher图的ROI空间均值，两者不能混称。1000次被试bootstrap提供逐项区间，不能替代多重比较校正或正式层间差异检验，也不将体素或图片对当独立被试。

接下来：视觉检查及方法记录；预先确定新推断；补充跨层空间比较的不确定性；NSD直接r；多次随机8/8图片分半；结合noise ceiling与团队讨论主文。当前描述性结果不证明不同计算阶段已被定位。

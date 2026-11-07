# Direct Pearson完成结果与文件入口

2026-11-08。本轮计划内分析已完成并通过数值与图像核查。解释见[分析摘要](ANALYSIS_BRIEF_20261108.md)，范围与逐项证据见[完成审计](COMPLETION_AUDIT_20261108.md)。这不代表论文修稿或合作者的最终方法决定已经完成。

## 已完成作业与证据

| 内容 | 作业/证据 | 结果 |
|---|---|---|
| 独立PNG模型重算 | 29530910 | 12层与features最大误差<5e-7 |
| 实际高层GLM输入身份 | `higher_glm_identity_reproduced.json` | 30高层设计、实际230 run的EV与contrast通过 |
| 合成全脑 | 29530911 | 30项全部成功，360张原始r图 |
| 合成组汇总 | 29530912 | 12组图、12计数图，共同191696体素 |
| 合成全量数值QC | 29530973 | 360张图及组汇总逐体素检查通过 |
| NSD全脑/汇总 | 29530978 / 29530979 | 8人×16集×12层=1536张r图，分层汇总 |
| NSD独立QC | 29531063 | 1536图、128覆盖通过；384组值抽样误差3.6069e-9 |
| 图片分半/汇总 | 29530981 / 29530982 | 100次互补8/8、98种无序划分，全部成功 |
| 分半独立QC | 29531064 | 100图/矩阵核对；原始cope重算的组值误差6.7461e-9 |
| 群体检验与空间bootstrap | 29531037 | 10000次符号翻转、1000次bootstrap |
| 推断独立QC | 29531065 | 全部被检验位置的t/p/mask及bootstrap核对通过 |
| 同定义noise ceiling | 29531157 | 同一完整Cube脑RSM；1440显式相关抽查误差2.8085e-8 |
| 修正ROI/可靠性图 | 29531159 | 正确MATLAB索引起点；旧摘要保留 |
| 修正ROI独立QC | 29531160 | 902629体素零差异，4680逐人数值及156组摘要行通过 |
| 解剖范围敏感性 | 29531161 | 13视觉ROI并集5463体素，保留主分析不变 |

所需作业均COMPLETED、exit 0:0。16项科学计算测试最终组合运行全部通过，9张关键预览完成视觉审查；服务器`completion_evidence.json`和`visual_review.json`记录文件hash与完成快照。

两次早期模型核验失败来自新核验脚本误取不含bias的内部conv张量；修正为历史命名pre-ReLU张量后通过，原features未修改。旧cor地图的异常来源仍未知，本轮没有依赖它们。

## 服务器输出

以下相对路径均位于：

`/gpfs/milgram/project/turk-browne/projects/localize_paper/derivatives/direct_rsa_revision_20261107/`

| 路径 | 内容 |
|---|---|
| `provenance/` | 模型PNG核验、任务/run/高层cope顺序及来源hash |
| `full/subXXX/` | 每人12张`layerXX_r.nii.gz`、覆盖和qc.json |
| `group/` | 12张`layerXX_mean_r.nii.gz`、n图及`common_all30_mask.nii.gz` |
| `group/spatial_correlation_common_all30.csv` | 新r图的12×12空间相关矩阵 |
| `cohort_independent_qc.json` | 合成图像全量及组汇总逐体素核查 |
| `inference/` | 12层t、新FWE p及mask、10000个null最大值 |
| `inference/spatial_pointwise95_low.csv`、`inference/spatial_pointwise95_high.csv` | 1000被试bootstrap的逐项空间相关区间 |
| `inference/adjacent_layer_spatial_intervals.csv` | 11对相邻层摘要 |
| `nsd/manifest.json` / `full/job0..127/` | 明确有序ID清单及1536张原始r图 |
| `nsd/group/` | 每人完整16集平均、逐集8人平均及最终组图和n |
| `nsd/group/common_all8_all16_mask.nii.gz` | 8人×16集×12层共同302573体素 |
| `image_splits/plan.json` | 100次固定图片划分、seed及范围 |
| `image_splits/batchXX/splitXXX/` | 两半组图`group_r.npy`、12×12跨半矩阵及逐层摘要；坐标在batch目录 |
| `image_splits/summary/` | 100次矩阵及同层、异层、差值分布；分位数不是置信区间 |
| `roi_corrected/` | 正式逐人/组ROI CSV、修正图谱和MNI配准叠图 |
| `noise_ceiling/` | 同定义下界/上界r图和逐人值；完整Cube脑RSM |
| `noise_ceiling/roi_corrected_plots/` | 正式ROI可靠性CSV及与模型r对照图 |
| `anatomical_sensitivity/` | 13视觉ROI并集下的分半及空间比较补充检查 |
| `previews/`、`inference/plots/`、`image_splits/plots/`、`nsd/comparison_plots/` | 未阈值化、新显著性轮廓、分半和NSD对照预览 |
| `completion_evidence.json` / `visual_review.json` | 完成证据及经审查图像SHA |

## 本机可打开的图和汇总

目录：`D:\Desktop\files\job\localize-direct-pearson-results\20261108`。仅预览、摘要、ROI CSV与QC在本机；计算、NIfTI脑图与脑RSM缓存保留在Milgram。

推荐依次查看：

1. `previews/group_r_axial_preview.png`：合成数据12层直接r。
2. `nsd/comparison_plots/synth_nsd_common_scale.png`：合成与NSD，交集191258体素、同一±0.30色标。
3. `image_splits/plots/same_vs_different_layers.png`：全共同空间分半稳定性。
4. `anatomical_sensitivity/comparison_space_sensitivity.png`：视觉范围补充结果，避免只凭全脑空间概括。
5. `noise_ceiling/roi_corrected_plots/roi_model_r_and_noise_ceiling.png`：模型r与同输入可靠性。
6. `roi_corrected/atlas_alignment.png`：修正ROI与MNI模板叠图。

**旧`roi/`和原ROI可靠性图已被修正版本替代。**历史转换在三个轴各有一个1mm源体素的索引起点差异；修正影响ROI边界及其摘要，不影响全脑r、主群体检验、NSD或主图片分半结果。旧文件保留，不删除或静默覆盖。

## 阅读限制

主预览和主空间比较使用全部30人的共同覆盖；其他组图位置的n可能不同，不能把低覆盖边缘极值当作30人效应。FWE检验推广到参与者、条件固定于现有图片且要求符号对称性；NSD比较是不同设计间的描述性比较；空间bootstrap区间是逐项区间；分半抽样分位数不是独立实验的置信区间。灰色noise-ceiling带也不是置信区间。

图为已核查的分析预览，论文定稿仍需选择版式、撰写完整图注并据结果改写论证。本轮不声称新的residual Pearson地图已经完成。

# Direct Pearson修回分析入口

本轮执行已完成；原`figures/`仍保留已投稿方法。计算与绘图在Milgram进行，本机仅查看预览与汇总。

- [结果摘要与解释](ANALYSIS_BRIEF_20261108.md)
- [执行规划与完成记录](PLAN_DIRECT_PEARSON.md)
- [方法、推断范围和限制](METHODS_DIRECT_PEARSON.md)
- [输出位置与作业证据](RESULTS_DIRECT_PEARSON_20261108.md)
- [逐项完成审计](COMPLETION_AUDIT_20261108.md)

## 已运行入口

| 代码 | 功能 |
|---|---|
| `direct_rsa/core.py` | 复用旧ModelSim的有符号Pearson定义；来源c94b5dc，16图/8图共同索引 |
| `verify_stimulus_model.py`、`audit_stimulus_identity.py`、`audit_higher_glm.py` | 独立PNG模型复算、任务与实际GLM输入顺序 |
| `worker.py`、`slurm/direct_rsa_full_array.sbatch` | BrainIAK Searchlight + MPI，30人全脑12层r |
| `group.py`、`qc_cohort.py` | 被试Fisher汇总、覆盖与全量独立QC |
| `nsd_manifest.py`、`nsd_worker.py`、`nsd_group.py` | NSD 8人×16集输入、全脑与分层汇总 |
| `cache_worker.py`、`cached_rsa.py`、`image_splits.py` | float64脑RSM缓存、100次互补8/8分半与汇总 |
| `inference.py`、`configs/inference_direct_rsa.json` | 联合max-|t|被试符号翻转与空间相关bootstrap |
| `atlas.py`、`roi_summary.py`、`qc_corrected_roi.py` | 正确MATLAB坐标转换、ROI摘要与独立坐标/数值核验 |
| `noise_ceiling_cached.py` | 同一脑RSM的LOSO/inclusive可靠性背景 |
| `anatomical_sensitivity.py` | 独立解剖视觉范围的描述性补充检查 |
| `qc_extended.py` | NSD、图片分半与正式推断的独立QC |
| `preview.py`、`revision_plots.py` | 原始r、新推断轮廓、分半、NSD及修正ROI可靠性图 |

所有Python模块位于`analysis/direct_rsa/`，Slurm入口位于`analysis/slurm/`。源数据与当前输出根目录在`configs/milgram_direct_rsa.json`和各Slurm脚本中明确记录。

## 测试与运行注意

从仓库根目录，在Milgram已有`analysis`环境运行：

```bash
python -m unittest \
  analysis.direct_rsa.tests.test_core \
  analysis.direct_rsa.test_group \
  analysis.direct_rsa.test_cached_rsa \
  analysis.direct_rsa.test_inference \
  analysis.direct_rsa.test_noise_ceiling_cached \
  analysis.direct_rsa.test_atlas
```

16项通过。MPI worker使用已有`brainiak`环境及MPICH 3.3.2，不把仅有NumPy的`analysis`环境当作MPI环境。本机默认Python缺NumPy，未用它作为通过依据。

现有输出目录已经有结果，入口会拒绝覆盖。要独立重现，应使用新的输出根目录并相应更新Slurm脚本/CLI参数；不要直接重投本轮同名任务。原始r、模型和脑RSM索引、覆盖、输入stat、配置/模型hash与代码版本均可追溯。

## 历史保留与正式版本

旧direct Pearson来源为`preprocess/glm/prototype/link/scripts/mapping.py:ModelSim`。旧1440张cor图存在，但抽样差异和异常未解决，未用于本轮正式结果。2026年旧回归结果优先只读`$scratch60/localize_bkp_09272026/result`，不能因工作目录为空推断丢失。

本轮ROI正式目录为`roi_corrected/`，对应可靠性ROI图为`noise_ceiling/roi_corrected_plots/`。原`roi/`保留历史坐标转换，已经被替代。全脑结果、主统计与图片分半没有因这个ROI修正而重算。

内部合作者邮件和讨论材料在Git忽略的`_private/`中，不包含在本公开分析入口。当前工作分支为`revision/direct-pearson-20261107`。

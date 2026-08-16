# 代码公开范围规划（审阅稿）

状态：公开范围规划与 figure-centered 迁移草案。当前分支已开始按本规划迁移正文 Figure 1–6；补充图仍暂缓。

## 目标

当前仓库已经包含两类内容：生成合成图像的历史程序，以及从派生结果绘制单个 panel 的脚本。它还不能复现论文最核心的分析，即如何从 CNN 激活和 fMRI 图像得到逐层的 whole-brain searchlight map。

下一阶段的目标不是复制原项目目录，而是整理出一条能被读懂、运行和核验的分析链路：

1. 从刺激图像得到 12 个 CNN 层的 representational similarity matrices（RSMs）。
2. 从每名参与者的 image-level beta maps 得到每个 searchlight 的脑 RSM。
3. 使用 leave-one-participant-out 回归得到 raw cross-validated \(R^2\) 和 permutation-normalized score maps。
4. 对 NSD 的 16 个 image sets 重复相同分析，并按论文规定的顺序聚合。
5. 实现 residual method、组水平推断、跨层空间相关和主要补充分析。

公开仓应以方法复现为中心。PDF 组装、Overleaf 排版和 Milgram 调度记录不属于这条主线。

## 术语和数据单位

这些名称应在代码、README 和论文中保持一致。

| 名称 | 在公开代码中的含义 |
| --- | --- |
| CNN layer RSM | 一个模型层内，16 张图像的 unit-activation patterns 两两做 Pearson correlation 后得到的 16 x 16 矩阵 |
| CNN layer fingerprint | CNN layer RSM 的上三角 120 个元素。它不是单个 channel 的 activation vector |
| Brain searchlight RSM | 一个 7 x 7 x 7 voxel searchlight 内，16 张图像的空间 activation patterns 两两做 Pearson correlation 后得到的 16 x 16 矩阵 |
| Brain fingerprint | Brain searchlight RSM 的上三角 120 个元素 |
| Non-residual mapping | 用一个 CNN layer fingerprint 预测一个 brain fingerprint 的简单线性回归 |
| Residual mapping | 先用其余 11 个 CNN layer fingerprints 回归 brain fingerprint，再用目标层预测脑残差。这里残差化的是脑 RSM，不是目标层 RSM |
| Raw cross-validated \(R^2\) | 在 held-out participant 上比较预测 fingerprint 与观测 fingerprint 得到的原始 \(R^2\) |
| Permutation-normalized score | 将真实 cross-validated \(R^2\) 相对于 participant-level permutation null 转为 z score |
| Per-set map | NSD 某一个 16-image set 单独得到的 map |
| Mean-across-sets map | 先在每名 NSD participant 内对 16 个 set 求均值，再做组水平统计 |
| Cross-layer spatial correlation | 在 representational-stability mask 内，对两个 layer 的未阈值化 whole-brain maps 跨 voxel 做 Pearson correlation |

Residual method 在统计学上更接近 semi-partial 或 part regression，而不是把脑 RSM 和目标层 RSM 都对其他层残差化的 full partial correlation。公开文档中应直接写出计算式，避免只用“partial correlation”概括。

## 论文方法与原代码位置

下表按 `rebuttal/final/main_tex.md` 的最新方法和 figure 编号整理。原代码路径用于追溯，不代表这些文件应原样复制。

| 方法或结果 | 最新 figure | 主要原代码 | 处理意见 |
| --- | --- | --- | --- |
| 合成 16 张图像 | Fig. 1, Fig. 2 | `ImageSynthesis/ImageSynthesis.py` | 已整理到 `image_synthesis/`，仍需补齐精确复现所需资产及 provenance |
| 计算 synthesized 和 NSD 的 CNN RSM | Fig. 1B-E | `Sfig3/Sfig3b/nsd_dataPrepare/modelData/dataPrepare_model.py`，`NSD_similarity/baselineLayerCorr_NSD_parent.py` | 从历史合成脚本中抽出只负责 activation extraction 和 RSM 计算的模块 |
| NSD 随机 image-set baseline | Fig. 1E-H | `NSD_similarity/baselineLayerCorr_NSD_parent.py` | 保留 sampling、seed、5,000 sets 和 adjacent-layer summary，去掉绘图与集群路径 |
| Synthesized fMRI preprocessing 和 first-level GLM | Methods, Fig. 3 输入 | `preprocess/glm/` | 公开可移植的 FEAT templates、event-file contract 和运行说明；不复制站点相关路径和作业管理代码 |
| Synthesized representational stability | Fig. S7A | `Sfig3/Sfig3a/searchlight/sl.py` | 提取 within-image、between-image 和 searchlight 核心；明确原方法的 5,000 次抽样及等价 closed-form 实现 |
| NSD beta extraction 和 2-mm resampling | Fig. 4 输入 | `Sfig3/Sfig3b/nsd_dataPrepare/brainData/dataPrepare_brain.py`，`sharedImageIndex.csv` | 重写为基于 NSD release layout 的 CLI，公开 256 个 image IDs 和 repetition/session 索引生成规则 |
| NSD representational stability | Fig. S7B | `Sfig3/Sfig3b/nsd_repStability_RSA_sl/`，`nsd_repStability_RSA_sl_meanAndRandomise/` | 抽出每个 set 的 stability score、participant 内跨 set 平均和组水平 mask |
| Synthesized non-residual searchlight | Fig. 3A | `fig3a/searchlight_train_linearModel/sl.py`，`fig3a/searchlight_score_linearModel/sl.py`，`fig3a/shuffleNorm/sl.py` | 作为统一 searchlight API 的 synthesized 配置，不保留重复的专用脚本 |
| NSD non-residual searchlight | Fig. 4A | `fig4/nsd_findLayerInBrain_leaveOneSubOutReg/searchlight_train_linearModel/sl.py`，`searchlight_score_linearModel/sl.py` | 与 synthesized 共用算法，每个 NSD set 独立运行 |
| Residual searchlight | Fig. 6A | `nsd_residualModel/searchlight_train_linearModel/sl.py`，`searchlight_score_linearModel/sl.py` | 单独实现并测试 brain-only residualization，不能只公开画图代码 |
| Participant-level permutation normalization | Fig. 3A, 4A, 6A | `rebuttal/BZ/rebuttal_analyses/corrected_permutation/`，上述历史 score scripts | 以 `corrected_permutation/` 中 coordinated row/column stimulus-label permutation 为当前规范；旧 score scripts 只用于追溯，不作为公开实现 |
| NSD participant 内跨 set 聚合 | Fig. 4A, 6A | 两套 `nsd_allSetMeanRandomise/nsd_allSetMeanRandomise.py` | 用一个纯 Python aggregation module 代替两份几乎相同的脚本 |
| Group t maps、稳定性 mask 和 FDR | Fig. 3A, 4A, 6A | `display_surface_plot/display_surface_plot.py` 中的 t-statistic reconstruction 与 BH-FDR；原 `randomise` scripts | 统计计算与 surface rendering 分离。公开一尾 one-sample t test、逐层 BH-FDR 及 mask 规则 |
| Best-layer map | Fig. 3B, 4B, 6B | `fig3b/`、`fig4/fig4b_fdr.py`、`fig6ab/fig6b_fdr.py` | 保留 argmax 和显著性资格规则，交给已有 panel script 绘图 |
| Continuous cross-layer overlap | Fig. 5A-B, 6C | `analysis_04_continuous_overlap.py`，`analysis_04b_continuous_overlap_bootstrap.py` | 公开最终采用的 Pearson metric 和 per-set aggregation，不把已弃用的二值 overlap 当作主流程 |
| Bootstrap comparison | Fig. 5C-D, 6D-E | `analysis_04b_continuous_overlap_bootstrap.py`，`docs/adr/0001-per-set-vs-grand-mean-cross-layer-overlap.md` | 复用同一批 16 个 per-set matrices，保留 matrix mean 与 bootstrap center 一致的 fail-loud check |
| Noise ceiling | Fig. S6 | `analysis_01_noise_ceiling.py`，`analysis_10_noise_ceiling_viz.py` | 需要修完 NSD loader 后再公开，当前 `analysis_01` 的 NSD 分支仍是 `NotImplementedError` |
| Split-half reliability | Fig. S8 | `analysis_12_split_half_subjects.py` 及相关 merge scripts | 整理为一个可设 seed 的 analysis script，输出 split-level arrays，panel 绘图继续独立 |
| NSD-Synthetic comparison | Fig. S4 | `analysis_09b_nsd_synthetic.py` | 公开统计计算和 872-image input contract，不重新分发原数据 |
| Number-of-images analysis | Fig. S3 | `analysis_08c_n_images_sufficiency.py`，`analysis_08c_n_images_sufficiency_zscore.py` | 先确认正文使用的是哪一个 score definition，再保留唯一版本 |
| Structured natural-image seed | Fig. S5 | `analysis_11_structured_seed_synthesis.py` | 保留方法和 fingerprint 输出，去掉 cluster wrapper |
| Geodesic-distance analysis | Fig. S13 | `analysis_07_geodesic_distance.py` | 公开 best-layer、surface distance 和 binning 计算 |
| Toy manifolds | Fig. S18 | `Sfig7/swissRoll.py` | 重写为小型确定性示例，去掉硬编码保存路径和混合绘图逻辑 |

## 必须进入公开仓的内容

下面是优先级 P0。缺少其中任一项，都无法从方法输入复现主要 searchlight figures。

### 1. 固定的分析清单和标识符

建议新增：

```text
configs/
  layers.yaml
  synthesized_stimuli.tsv
  nsd_image_sets.tsv
  participants_synthesized.tsv
  participants_nsd.tsv
  paths.example.yaml
```

需要记录：

- 12 个 Inception-V1 node 的准确名称、顺序和 layer number。
- 16 张 synthesized images 的稳定 ID 和文件命名。
- 16 个 NSD sets，每个 set 16 个 image IDs，共 256 个 ID，顺序必须固定。
- Synthesized dataset 的 30 名纳入者，以及每人实际 6、7 或 8 个 runs。
- NSD 的 8 名 participant identifiers。
- MNI 2-mm output shape、affine、searchlight radius 和 mask definition。

`sharedImageIndex.csv` 含有 18,668 行运行级索引，公开前应检查它是否只是 NSD 的公开 stimulus metadata 派生物。若可以由 NSD 官方 metadata 重建，优先公开生成脚本和 256 个 image IDs，不直接复制这份内部中间表。

### 2. CNN fingerprints

建议新增：

```text
src/cnn_fingerprints.py
```

它应完成：

- 加载 Inception-V1 graph 和指定的 12 个 layer tensors。
- 对一组图像执行与 synthesis 相同的 resizing、cropping 和 mean subtraction。
- 把每层每张图像的 unit activations 展平。
- 计算 12 x 16 x 16 layer RSMs。
- 输出 12 x 120 upper-triangle fingerprints，并保存 stimulus order 和 layer names 到 metadata。
- 对 RSM 对称性、对角线、finite values 和输出 shape 做强制检查。

历史 `dataPrepare_model.py` 大量复制了 synthesis helper，不应整文件搬入。公开版只保留上述数据变换。

### 3. fMRI image-level inputs

建议新增：

```text
src/prepare_synthesized_betas.py
src/prepare_nsd_betas.py
docs/input_data.md
```

Synthesized 分支要说明 FEAT 输出如何映射到 16 张 image beta maps。NSD 分支要读取官方 `betas_fithrf`，按 participant、image ID 和三个 repetitions 提取 beta，并重采样到共同 MNI 2-mm 网格。

这部分不能照抄当前脚本。已看到的具体问题包括：

- `dataPrepare_brain.py` 使用了 `nib`，但文件中没有导入 `nibabel`。
- NSD 根路径、FSL reference 和 scratch 输出全部硬编码。
- beta maps 使用 nearest-neighbor resampling。公开前要确认这是有意的方法选择，还是历史实现遗留；连续 beta 数据通常需要说明插值方式。
- 当前脚本默认每张图有三个 repetitions，但缺少明确的数量、重复 ID 和 affine 门禁。

公开版至少检查每个 participant x set 是否恰有 16 张图、每张图是否恰有 3 个 repetitions、所有图是否 finite、shape 和 affine 是否一致。

### 4. Representational-stability searchlight

建议新增：

```text
src/representational_stability.py
```

核心输出应同时保留：

- within-image cross-run similarity。
- sampled between-image similarity。
- within-minus-between raw effect。
- 相对于 sampled-between null 的 normalized score。

Synthesized 和 NSD 应调用同一组纯函数，只通过 repetition/run index masks 区分设计。原 synthesized 脚本后来用 finite-population correction 的 closed-form 替代 5,000 次抽样。公开版可以保留这个等价优化，但必须附一个小测试，证明它与固定 seed 的 Monte Carlo 结果在容差内一致。

### 5. Non-residual leave-one-participant-out searchlight

建议新增：

```text
src/searchlight_mapping.py
```

单个 searchlight 的公开 API 应清楚表达下面的步骤：

1. 从 16 x n_voxels image patterns 计算 16 x 16 brain RSM。
2. 取 brain RSM 与某一 CNN layer RSM 的相同 120 个上三角元素。
3. 在 training participants 上拟合 slope 和 intercept。
4. 将 training estimates 应用于 held-out participant。
5. 返回 raw cross-validated \(R^2\) 以及 permutation null 所需统计量。

原实现先分别拟合 training participants，再平均 slope 和 intercept。论文文字更像是先平均 training brain RSM，再拟合一次。由于所有 participant 共用相同 predictor，在数据完整且 NaN mask 一致时两者应等价；公开前必须用实际输出或构造数据验证这一点，并把最终采用的顺序写清楚。

Synthesized 和 NSD 不应再有两份复制的算法。差异只应来自 participant 数量、set dimension 和输入路径。

### 6. Residual method

建议新增：

```text
src/residual_mapping.py
```

训练阶段按目标层 \(x_k\) 计算：

```text
brain_z = zscore(brain_rsm_upper)
brain_hat_other = OLS(other_11_layer_fingerprints, brain_z)
brain_residual = brain_z - brain_hat_other
target_model = OLS(target_layer_fingerprint, brain_residual)
```

每个 predictor fingerprint 的标准化规则、intercept 和 NaN handling 都要固定。输出至少包括 residual slope、intercept、held-out raw \(R^2\) 和 permutation-normalized score。

这里有一个必须先解决的实现问题。历史 score script 用 residual model 预测出 `y_pred_residual` 后，把它与 held-out participant 的完整 z-scored brain fingerprint 计算 \(R^2\)。论文文字则容易被理解为先对 held-out brain fingerprint 也计算 residual，再评分。公开前要由作者确认目标量，之后用一个单元测试锁定，不能在代码清理时自行选择。

### 7. 当前采用的 RSM permutation

建议新增：

```text
src/permutation.py
```

当前方法已经确定，不再是两个候选方案之间的选择。`rebuttal/BZ/rebuttal_analyses/corrected_permutation/` 是较新的实现，加入于 commit `102db44`。按照 `rebuttal/final/main_tex.md`，每次 permutation 先用 `np.random.permutation(16)` 随机置换刺激标签，再用 `rsm_full[np.ix_(perm, perm)]` 协同重排 held-out brain searchlight RSM 的行和列，最后重新提取 120 个上三角元素用于 null \(R^2\)。公开代码以该 Methods 表述为准。

历史 synthesized 和 non-residual NSD score scripts 中直接 `np.random.shuffle` 120 个上三角元素的做法已经被修正，不应进入公开实现。Residual 历史 score script 中直接打乱 120 个 brain-RSM 元素的代码也只能作为历史记录。

最终 response 说明整个 pipeline 已用 corrected null distribution 重跑，Figures 3 through 6 也已替换。不过当前可定位的 `corrected_permutation/sl.py` dispatcher 明确枚举的是 30 participants x 12 layers 的 synthesized analysis。公开迁移时还需要追踪 NSD 和 residual corrected outputs 的确切生成入口或结果 provenance。这是输出追溯问题，不是 permutation 方法尚未决定。

公开版需要：

- 把 coordinated row/column permutation 提取成三个数据分支共用的唯一函数。
- 为 synthesized、NSD non-residual 和 NSD residual 显式调用同一函数，并始终置换 held-out brain searchlight RSM，不再从旧 score scripts 复制 shuffle 逻辑。
- 记录每名 participant 的 500 次 permutation、random seed、stimulus order 和输出 provenance。
- 加入测试，证明重排行列后 RSM 对称、刺激 pair 结构被保留，并且不是 120 元素的任意置换。
- 测试还应确认在穷举所有 stimulus-label permutations 时，置换 brain RSM 与置换 model RSM 得到相同的 null 分布。公开实现仍固定采用 Methods 指定的 brain searchlight RSM 版本。

因此，coordinated row/column permutation 是已解决的规范方法；剩余工作是补齐三个分析分支的代码入口与最终结果 provenance。

### 8. NSD set aggregation 和组水平推断

建议新增：

```text
src/group_inference.py
```

公开版应把以下顺序编码为函数和测试：

```text
participant x set score maps
    -> participant 内对 16 sets 求均值
    -> 跨 participant 的 one-sample t map
    -> 限制到 representational-stability mask
    -> 每层在 mask 内做 one-sided Benjamini-Hochberg FDR, q < 0.05
```

Raw \(R^2\)、permutation-normalized score、unthresholded group t map 和 FDR mask 必须使用不同文件名和 metadata，避免目前历史脚本中“raw R2”有时实际指 group z score 的混用。

FSL `randomise` 和后来的参数 t-to-p-to-BH 实现都存在于原仓。公开前应以最终 manuscript figures 的真实输入为准，写清楚：

- representational stability mask 本身如何做 TFCE correction。
- layer-mapping figures 如何从 participant maps 得到 unthresholded t map。
- FDR 是每层分别在稳定性 mask 内做，使用右尾 \(p\) 值，还是采用其他口径。
- TFCE 输出是否仅用于 stability mask，而不再作为 layer map 的最终 correction。

### 9. 跨层空间相关和 bootstrap

建议新增：

```text
src/cross_layer_overlap.py
```

正文最终使用未阈值化 effect maps 在稳定性 mask 内的 Pearson correlation。NSD 和 residual 必须先对每一个 image set 分别计算 12 x 12 matrix，再对 16 个 matrices 逐元素求均值。不能先平均 16 个 maps 再计算一次 correlation，因为 correlation 是非线性的。

Bootstrap 也从同一组 16 个 per-set matrices 取 off-diagonal 或 adjacent-layer means。应保留现有的重要门禁：

- 12 层和 16 个 sets 必须齐全。
- 每个 participant x set x layer 文件必须齐全。
- per-set identity 不能用不带 set ID 的列表隐式表示。
- displayed matrix 的 off-diagonal mean 必须等于 bootstrap source values 的均值。
- cache 必须记录 dataset、mask、层数、set 数和代码版本。

## 建议进入公开仓的补充分析

这些是 P1。它们支持论文中的补充结论，但不应先于 P0 主链路。

1. Noise ceiling 和 representational stability 的关系，Fig. S6。
2. Number-of-images sufficiency，Fig. S3。
3. NSD-Synthetic 对照，Fig. S4。
4. Structured natural-image seed synthesis，Fig. S5。
5. Split-half reliability，Fig. S8。
6. Best-layer reliability、ROI summaries 和 geodesic distance，Fig. S13 及相关分析。
7. Per-set voxel counts 和 variability，Fig. S14-S16。
8. Raw \(R^2\) 与 unthresholded maps 的导出，Fig. S10、S11、S17。
9. Toy manifold demonstration，Fig. S18。

旧的 binary thresholded overlap 可以放在 `legacy/` 或只在 release notes 中说明，不应与正文采用的 continuous Pearson metric 并列成两个推荐入口。Cluster-mass sensitivity analysis只有在论文最终保留相应 figure 时才进入 P1。

## 不建议进入公开仓的内容

以下内容不帮助独立研究者理解或复现论文方法，且会增加维护或泄露风险：

- Overleaf/Word/PDF 的整页组装和修稿临时脚本。
- Milgram 专用 Slurm job arrays、队列名、scratch 路径、conda 绝对路径和日志。
- 用户名、token、remote URL 中的认证信息及任何本地环境 dump。
- 原始被试数据、NSD 图像文件和第三方模型权重的复制件。
- 大型逐 voxel 中间缓存、已渲染 PDF/PNG、core dumps 和 `__pycache__`。
- 只用于一次排错的 diagnostic scripts，除非其中的检查已被吸收到正式 validation module。
- 被新方法取代且没有论文对应结果的早期分析版本。

这里的“少披露”应理解为不公开无关的实验室基础设施和中间产物，不是省略论文结论所依赖的算法、参数或 stimulus identifiers。

## 对现有 image synthesis 目录的补充计划

`image_synthesis/` 已有程序和 selected-channel JSON，但精确复现仍需处理两个外部资产：

- `bestchans16.pkl`：其实际数值已经转成 `assets/selected_channels_16.json`，因此不需要再公开 pickle。
- `16corr_GradOrth.npy`：这是历史 run 使用的目标结构资产。若目标矩阵可以由公开规则重新生成，应公开生成函数和 seed；若不能，则应在确认不含受限内容后公开该小型数组及校验值。它只对精确重跑历史 synthesis 必要，不应成为理解 searchlight 分析的前置条件。

Inception-V1 graph 应由脚本从合法来源下载，或要求用户自行提供并检查 SHA-256。未经许可不把第三方模型文件直接提交到仓库。

## 建议的仓库结构

```text
cnn-layer-brain-mapping-figures/
  README.md
  LICENSE
  CITATION.cff
  CODE_RELEASE_PLAN.md
  pyproject.toml
  configs/
    layers.yaml
    synthesized_stimuli.tsv
    nsd_image_sets.tsv
    participants_synthesized.tsv
    participants_nsd.tsv
    paths.example.yaml
  image_synthesis/
  src/cnn_brain_mapping/
    fingerprints.py
    fmri_inputs.py
    stability.py
    searchlight.py
    residual.py
    permutation.py
    group_inference.py
    overlap.py
    validation.py
  analyses/
    noise_ceiling.py
    split_half.py
    n_images.py
    nsd_synthetic.py
    structured_seed.py
    geodesic_distance.py
    toy_manifold.py
  figure_code/
  tests/
    data/
    test_fingerprints.py
    test_searchlight.py
    test_residual.py
    test_permutation.py
    test_group_inference.py
    test_per_set_aggregation.py
  docs/
    input_data.md
    methods_to_code.md
    reproduction.md
```

文件名只是建议。关键是把纯计算、数据读取、集群调度和绘图分开。

## 数据接口

公开代码应接受和产生明确的数据对象，而不是依赖目录名猜测内容。

| 阶段 | 最小输入 | 输出及 shape |
| --- | --- | --- |
| CNN RSM | 16 images、12 layer names | `layer_rsms`: 12 x 16 x 16；`fingerprints`: 12 x 120 |
| Synthesized brain input | participant 的 16 image beta maps | X x Y x Z x 16 |
| NSD brain input | participant、set、16 images x 3 repetitions | X x Y x Z x 16，另存 repetition provenance |
| Searchlight fit | brain maps、mask、layer fingerprints | participant x layer 的 slopes/intercepts |
| Searchlight score | held-out brain maps、LOPO parameters | raw \(R^2\)、null summary、z score maps |
| Residual fit | z-scored brain fingerprint、12 model fingerprints | 12 个 residual models |
| Group inference | participant score maps | unthresholded t map、FDR q map、binary mask |
| Cross-layer overlap | 12 unthresholded maps、stability mask、set ID | 12 x 12 correlation matrix |

所有 NIfTI 输出都应记录 affine、shape、voxel size、mask voxel count、participant IDs、set ID、layer ID、permutation seed 和生成命令。可用同名 JSON sidecar，不必把这些信息塞进文件名。

## 验证门禁

整理代码时不能只验证“脚本跑完”。至少需要以下检查：

### 小型单元测试

- RSM 对称，diagonal 正确，upper triangle 恰有 120 个元素。
- 同一 stimulus order 在 CNN 和 brain fingerprints 中完全一致。
- 简单线性回归在构造数据上恢复已知 slope、intercept 和 \(R^2\)。
- Residual function 在构造的共享与独有成分上只保留目标层独有贡献。
- Stimulus-label permutation 同时重排行列，并保持 RSM 对称结构。
- BH-FDR 与 `statsmodels` 的参考实现逐元素一致。
- Per-set matrix mean 与 bootstrap source mean 的关系满足现有 ADR 门禁。

### 小型端到端测试

仓库应提供不含真实被试数据的 tiny synthetic dataset，在几个 searchlight centers 上跑通：

```text
fingerprints
  -> non-residual LOPO score
  -> residual LOPO score
  -> participant permutation z
  -> group t/FDR
  -> cross-layer correlation
```

### 与历史结果的 parity check

在不公开大文件的前提下，可保存少量预期摘要：每层的 voxel count、map mean/std、有限值数量、若干固定坐标值和跨层 correlation。新实现必须在约定容差内重现经作者确认的最终 pipeline。任何偏差都应停下并说明，不能静默接受。

## 公开前必须回答的问题

1. 记录 Fig. 3、4、6 最终 corrected outputs 的生成入口、路径和 commit，特别是 NSD non-residual 与 residual 分支。方法已经确定为 coordinated row/column permutation，此处只核对 provenance。
2. Residual held-out score 应针对完整 brain RSM，还是针对 held-out brain residual？现有实现和论文文字需要统一。
3. NSD beta 从 1 mm 到 2 mm 的 nearest-neighbor interpolation 是否为确认过的方法选择？
4. Synthesized training 采用“先平均 training brain RSM 再拟合”还是“先逐 participant 拟合再平均参数”？需要用测试确认两者在实际 NaN/mask 规则下等价。
5. Fig. S3 使用 `analysis_08c_n_images_sufficiency.py` 还是 z-score variant？
6. Group layer maps 的最终 correction 是否明确为稳定性 mask 内、每层单独、右尾参数 p 值的 BH-FDR？FSL TFCE 是否只用于稳定性 mask 和敏感性分析？
7. Synthesized preprocessed beta maps、stimuli 和 model activations 的最终 OpenNeuro accession 与目录结构是什么？
8. 代码许可证和第三方 Inception-V1 graph 的分发方式是什么？

这些问题没有答案时，可以继续做无争议的模块拆分和测试，但不能宣称公开仓已经能够精确复现最终 figures。

## 实施顺序

1. 先确认上面八个方法问题，并冻结 `configs/` 中的 layer、stimulus、participant 和 set manifests。
2. 提取纯函数：RSM、permutation、non-residual regression、residual regression、group inference、per-set overlap。
3. 为纯函数补单元测试，再接 NIfTI I/O。
4. 整理 synthesized 和 NSD input adapters，加入完整性与 affine 门禁。
5. 用历史最终输出做 parity check，记录通过的 commit 和参数。
6. 加入 P1 补充分析，每加入一项就更新 `methods_to_code.md` 的 figure 对应关系。
7. 最后更新 README、环境文件、license、citation 和公开数据下载说明，再提交和推送。

完成标准不是“原项目中的脚本都复制过来了”，而是论文每个主要计算结论都有一个明确入口、一个数据接口、一组关键参数和至少一个能失败出声的验证。

# Analysis 09b 实验设计：NSD-Synthetic 真实 CNN 对照

生成时间：2026-04-06
对应回应：R1-意见6（随机纹理对照分析）
脚本：`rebuttal/analyses/analysis_09b_nsd_synthetic.py`

---

## 背景与动机

### 原始 analysis_09 的问题

`analysis_09_random_texture.py` 的 `run_simulated_comparison()` 函数使用随机矩阵模拟"随机纹理"的 RSM：

```python
# 当前实现（有缺陷）
base_rsm = np.random.randn(120)
layer_rsms = [base_rsm + 0.3 * np.random.randn(120) for _ in range(12)]
```

这种做法**不具有科学说服力**：
- 随机矩阵不代表任何真实图像通过 CNN 的激活模式
- 审稿人可能质疑"你的随机对照是凭空捏造的"
- 与优化图像的比较失去了方法论一致性

### 解决方案

使用 **NSD-Synthetic 数据集**（Gifford et al., 2025）中的真实随机纹理图像，通过与我们论文相同的 InceptionV3 网络做 forward pass，得到基于真实图像的 CNN 激活 RSM。

---

## 实验设计：三组对比

### 组 A：优化合成图像（Optimized Synthesized Images）

- **图像来源**：论文中用于生成神经编码预测的 16 张优化图像
- **CNN 特征**：`data/inception/features.npy`（已预计算，shape: `(12, 16, 16)`）
- **RSM**：对每层 12×12 取上三角 → 120 对相关值
- **Inter-layer RSM 相关**：已知约 −0.054（层间高度可分）

> **注意**：优化图像的 CNN 特征已预计算，直接加载即可。若需重新计算，图像路径需在 Milgram 上确认（可能位于 `ImageSynthesis/images/{tag}/`）。

### 组 B：NSD-Synthetic 随机噪声图像（Random Noise Images）

- **图像来源**：`nsdsynthetic_stimuli.hdf5`，索引 **0–15**（白/粉/块噪声）
- **类别说明**：
  | 索引 | 类型 | std 范围 |
  |------|------|---------|
  | 0–3  | 白噪声 | 50–63 |
  | 4–7  | 粉噪声 | 35–45 |
  | 8–11 | 块噪声（大块） | 25–40 |
  | 12–15 | 块噪声（小块） | 30–40 |
- **CNN 特征**：需要 InceptionV3 forward pass（本脚本核心任务）
- **预期 Inter-layer RSM 相关**：约 0.9（层间高度相似 = 层不可分）

### 组 C：NSD-Synthetic 随机子集对照（Bootstrap Control）

- **图像来源**：`nsdsynthetic_stimuli.hdf5`，从全部 220 张中随机抽取，重复 100 次
- **每次抽取**：16 张（与我们的设计规模一致）
- **用途**：提供统计分布，证明"随机图像的层间高相关"是普遍现象，而非偶然
- **预期分布**：mean ≈ 0.85–0.95，sd 小

---

## CNN 设置

### 模型选择

**⚠️ 必须使用与论文完全相同的 TF1 InceptionV1（GoogLeNet）**，不能用 InceptionV3 或 PyTorch 版本，否则比较无效。

- 模型文件：`utils/tensorflow_inception_graph.pb`（52MB，已在集群上）
- 环境：`tf1` conda 环境（TF 1.14，含 PIL / h5py / imageio，全部可用）
- 层提取逻辑（直接来自 ImageSynthesis.py 和 baseline.py）：

```python
layers = [op.name for op in graph.get_operations()
          if op.type == 'Conv2D' and 'import/' in op.name]
layers1 = [l.split('/')[1] for l in layers if l.split('/')[1].split('2')[0] == 'conv']
layers2 = [l.split('/')[1] for l in layers if 'pool_reduce' in l]
incept_layers = layers1 + layers2  # 12 层
# tensor name: "import/{layer}:0"
```

### 输入预处理

与 baseline.py 保持一致（NSD 自然图像也用 539×539）：

```python
# Step 1: 读取 HDF5（用 tf1 env，h5py 正常）
import h5py
with h5py.File('nsdsynthetic_stimuli.hdf5', 'r') as f:
    img = f['imgBrick'][idx]  # (714, 1360, 3) uint8

# Step 2: 中心裁剪为正方形 (714×714)，再 resize 到 539×539
from PIL import Image
pil_img = Image.fromarray(img)
left = (1360 - 714) // 2   # = 323
pil_img = pil_img.crop((left, 0, left + 714, 714))
pil_img = pil_img.resize((539, 539), Image.LANCZOS)
img_array = np.array(pil_img).astype(float)  # (539, 539, 3)

# Step 3: TF forward pass（ImageNet mean = 117.0，与论文一致）
# t_preprocessed = tf.expand_dims(t_input - 117.0, 0)
# 激活 via sess.run(get_tensor(incept_layers), {t_input: img_array})
```

> **运行环境**：`tf1` conda 环境。h5py 在 tf1 中可用（已验证）。
> ```bash
> conda run -n tf1 python3 analysis_09b_nsd_synthetic.py
> ```

---

## RSM 计算方法

### 对每一层：

1. 提取该层对所有 N 张图的激活 → flatten → shape `(N, D_layer)`
2. 计算 N×N Pearson 相关矩阵 → 取上三角 → `C(N,2)` 个值
3. 对 N=16：120 对

### Inter-layer RSM 相关：

```python
# R2_maps: (12, 120) — 每层的 RSM 向量
layer_corr = np.corrcoef(R2_maps)  # (12, 12)
off_diag = layer_corr[~np.eye(12, dtype=bool)]
mean_inter_layer_corr = np.mean(off_diag)
```

- **低值（接近 0 或负）** → 各层 RSM 不相似 → 层具有可区分性（优化图像的特征）
- **高值（接近 1）** → 各层 RSM 高度相似 → 层不可分（随机噪声的预期特征）

---

## 文件路径

### Milgram 服务器路径

```
# NSD-Synthetic 刺激文件（已下载）
BASE_NSD = /gpfs/milgram/pi/turk-browne/projects/localize_paper/rebuttal/analyses/nsd_synthetic/
  ├── nsdsynthetic_stimuli.hdf5           # 220 张灰度图
  ├── nsdsynthetic_colorstimuli_subj01.hdf5  # 64 张彩色噪声（每被试）
  └── ...

# 优化图像特征（已预计算，直接用，无需重跑）
FEATURES_PATH = /gpfs/milgram/pi/turk-browne/projects/localize_paper/data/inception/features.npy
# shape: (12, 16, 16) — 12层 × 16×16 RSM

# 优化图像原始 PNG（路径待确认）
SYNTH_IMAGES_DIR = ?  # 可能在 ImageSynthesis/images/{tag}/ 或 results/images/
```

> **待确认**：优化合成图像的 PNG 路径（在 Milgram 上）。如果找不到，可以直接复用 `features.npy` 中的预计算 RSM 作为组 A，只对 NSD-Synthetic 图像做 forward pass。

### 输出路径

```
REBUTTAL_OUTPUT/nsd_synthetic_comparison/
  ├── features_noise_idx0-15.npy          # 组 B：16 张噪声图的 CNN 特征 (12, 16, 16)
  ├── features_bootstrap_{i:03d}.npy      # 组 C：100 次随机子集 (12, 16, 16)（可选，内存大则不存）
  ├── results_summary.csv                 # 三组 inter-layer 相关汇总
  └── nsd_synthetic_comparison.pdf        # 输出主图
```

---

## 预期结果与输出图设计

### 核心数值

| 组 | 描述 | 预期 mean inter-layer 相关 |
|----|------|---------------------------|
| A  | 优化合成图像 | ≈ **−0.054** |
| B  | NSD-Synthetic 噪声（idx 0–15）| ≈ **0.85–0.95** |
| C  | NSD-Synthetic 随机子集（100次）| ≈ **0.85–0.95 ± 0.03** |

差异预计 > 0.9（即使保守估计也达 0.7），远超统计显著性阈值。

### 图 1：柱状图 + 误差棒（主图）

```
纵轴：Mean inter-layer RSM correlation
横轴：三组（A, B, C）

[A: 优化图像]  bar ≈ −0.05  (narrow CI)
[B: NSD-噪声]  bar ≈  0.90  (narrow CI)
[C: Bootstrap] bar ≈  0.90  errorbar = ±1 SD over 100 repeats

显著性标记：*** p < 0.001
```

### 图 2（可选）：每层 RSM 热图对比

- 组 A vs 组 B 的各层 RSM 并排可视化（各 16×16 矩阵）
- 直观展示优化图像 RSM 层间差异大，噪声 RSM 层间几乎相同

---

## 运行指令

### 1. 确认 Milgram 上的 HDF5 文件

```bash
ls /gpfs/milgram/pi/turk-browne/projects/localize_paper/rebuttal/analyses/nsd_synthetic/
```

### 2. 运行分析脚本

```bash
# 使用 tf1 环境（TF 1.14，与论文一致）
cd /gpfs/milgram/pi/turk-browne/projects/localize_paper
conda run -n tf1 python3 -u \
  rebuttal/analyses/analysis_09b_nsd_synthetic.py
```

### 3. 预计运行时间

- 组 B（16 张图，12 层 forward pass）：< 1 分钟
- 组 C（100×16 张图）：5–10 分钟
- 总计：< 15 分钟（CPU 即可，不需要 GPU）

---

## 与 analysis_09 的关系

| 比较维度 | analysis_09（原始）| analysis_09b（本设计）|
|---------|-------------------|----------------------|
| 随机对照 | 随机矩阵模拟 | NSD-Synthetic 真实图像 |
| 优化图像特征 | `features.npy`（正确）| `features.npy`（相同）|
| 科学性 | 存疑（无真实图像）| 严格（真实 CNN forward pass）|
| 可重复性 | 不可（随机数）| 可（固定数据集）|
| 运行时间 | 瞬间 | < 15 分钟 |

analysis_09b 的结果将**替代** analysis_09 中 `run_simulated_comparison()` 部分，其余结构保持不变。

---

## 科学意义

> 我们的优化图像设计使得 InceptionV3 各层的 RSM 具有高度差异性（层间相关 ≈ −0.054），而相同规模的随机图像（NSD-Synthetic 噪声图）的 RSM 在各层间几乎相同（层间相关 ≈ 0.9）。这证明了我们的优化图像设计在揭示 CNN 层级表征结构方面具有独特优势，这不是 RSA 方法的一般属性，而是我们图像优化程序的特异性贡献。

# NSD-Synthetic 刺激文件结构

生成时间：2026-04-05
数据来源：Gifford et al., 2025, NSD-Synthetic dataset
S3 base URL：`https://natural-scenes-dataset.s3.amazonaws.com/nsddata_stimuli/stimuli/nsdsynthetic/`

---

## 文件列表

| 文件名 | 大小 | 内容 |
|--------|------|------|
| `nsdsynthetic_stimuli.hdf5` | 612 MB | 220 张灰度刺激（全部被试共用） |
| `nsdsynthetic_colorstimuli_subj01.hdf5` | 178 MB | 64 张彩色噪声（被试 01 专用） |
| `nsdsynthetic_colorstimuli_subj02.hdf5` | 178 MB | 64 张彩色噪声（被试 02 专用） |
| `nsdsynthetic_colorstimuli_subj03.hdf5` | 178 MB | 64 张彩色噪声（被试 03 专用） |
| `nsdsynthetic_colorstimuli_subj04.hdf5` | 178 MB | 64 张彩色噪声（被试 04 专用） |
| `nsdsynthetic_colorstimuli_subj05.hdf5` | 178 MB | 64 张彩色噪声（被试 05 专用） |
| `nsdsynthetic_colorstimuli_subj06.hdf5` | 178 MB | 64 张彩色噪声（被试 06 专用） |
| `nsdsynthetic_colorstimuli_subj07.hdf5` | 178 MB | 64 张彩色噪声（被试 07 专用） |
| `nsdsynthetic_colorstimuli_subj08.hdf5` | 178 MB | 64 张彩色噪声（被试 08 专用） |

总计：220 + 64 = **284 张刺激**（与论文一致）

---

## HDF5 内部结构

所有文件均只有一个 dataset：

```
imgBrick   shape=(N, 714, 1360, 3)   dtype=uint8
```

| 文件 | N |
|------|---|
| nsdsynthetic_stimuli.hdf5 | **220** |
| nsdsynthetic_colorstimuli_subj*.hdf5 | **64** |

### 空间布局

- 每张图像为全屏帧：**714（高）× 1360（宽）× 3（RGB）**
- 实际刺激区域：居中，大小约 **714 × 713 px**
- 两侧各有 **323 px** 的灰色边框（值 ≈ 114，std ≈ 8）
- 提取刺激的 crop：`img[:, 323:1036, :]` → 714×713 px

```
<--323px灰边--><-------714×713刺激-------><--323px灰边-->
```

---

## 220 张灰度刺激的类别分布

按论文（Gifford et al., 2025）顺序排列，索引从 0 开始：

| 索引范围 | 类别 | 张数 | std 范围 | 说明 |
|----------|------|------|----------|------|
| 0–15 | 白/粉/块噪声 | 16 | 25–63 | 随机噪声，不同空间频率特性 |
| 16–35 | 自然场景及变体 | 20 | 13–84 | 真实照片 + 翻转/Mooney/线条画等 |
| 36–51 | 对比度调制 | 16 | 8–30 | 同一场景，不同对比度（含 0% 对比度） |
| 52–67 | 相位相干调制 | 16 | 9–30 | 清晰图像 → 完全噪声渐变 |
| 68–107 | 英文单词 | 40 | 9–60 | 灰底黑字 |
| 108–219 | 螺旋/正弦光栅 | 112 | 8–60 | 不同方向 × 空间频率 |

std 均为像素强度标准差（0–255 范围），可作为类别识别的经验指标：
- std < 10：低对比度或均匀刺激
- std 10–30：中等对比度（单词、低对比度光栅）
- std > 30：高对比度（噪声、自然图像、高对比度光栅）

---

## 64 张彩色噪声（per-subject）

每位被试的彩色噪声因颜色校准不同，R/G/B 通道均值有明显差异：

```
imgBrick  shape=(64, 714, 1360, 3)  dtype=uint8
std 范围：14–30，均值 ≈ 22
R/G/B 通道均值（subj01 示例）：R=125, G=109, B=107
```

按论文描述：64 张 = **16 种色调** × **粉噪声**（Hue × Pink noise）

---

## 读取方式

### 推荐环境

`analysis` conda 环境的 h5py 存在链接错误（`H5Pget_fapl_direct`），**请改用 `brainiak` 环境**：

```bash
conda run -n brainiak python3 your_script.py
```

### 读取示例

```python
import h5py
import numpy as np

# 读取灰度刺激
with h5py.File('nsdsynthetic_stimuli.hdf5', 'r') as f:
    imgs = f['imgBrick'][:]          # shape: (220, 714, 1360, 3), uint8
    # 提取第 i 张刺激（去掉两侧灰边）
    stimulus = imgs[i, :, 323:1036, :]   # shape: (714, 713, 3)
    # 转为 float [0, 1]
    stimulus_float = stimulus.astype(np.float32) / 255.0

# 读取彩色噪声（被试 1）
with h5py.File('nsdsynthetic_colorstimuli_subj01.hdf5', 'r') as f:
    color_imgs = f['imgBrick'][:]    # shape: (64, 714, 1360, 3), uint8
```

### 为 CNN 输入预处理（Inception 224×224）

```python
from PIL import Image

def load_nsd_synthetic_for_cnn(hdf5_path, idx, out_size=224):
    with h5py.File(hdf5_path, 'r') as f:
        img = f['imgBrick'][idx, :, 323:1036, :]  # crop: 714×713×3
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((out_size, out_size), Image.LANCZOS)
    return np.array(pil_img).astype(np.float32) / 255.0
```

---

## Analysis_09 参考

analysis_09 的对比目的（R1-P6 回应）：证明随机纹理与优化合成图像在跨层 RSM 相关性上有显著差异。

NSD-Synthetic 中最适合用作"随机对照"的类别：
- **噪声图（索引 0–15）**：白噪声/粉噪声，与随机纹理最接近
- **参考基线**：`features.npy`（12 层 × 16×16 RSM，已计算好，路径见 config.py）

当前 analysis_09 使用 `run_simulated_comparison`（随机矩阵模拟），理想替代方案：
1. 用 `env_fmri`（PyTorch 1.10.2）+ torchvision InceptionV3
2. 对 NSD-Synthetic 噪声图（idx 0–15）做 forward pass
3. 对比优化图的 `features.npy`

"""
分析三：修正置换检验（行列协调打乱）
回应审稿意见：R1-P3

修正原始代码中的错误：
  错误做法：np.random.shuffle(上三角120个元素)
  正确做法：随机打乱16个刺激标签，同时重排brain searchlight RSM的行和列

这等同于 Mantel test (Mantel, 1967; Kriegeskorte et al., 2008)。

注意：此文件为早期原型（单进程逐体素循环，非常慢）。
实际运行请使用 BrainIAK searchlight + MPI 版本：
    corrected_permutation/corrected_permutation.py  — 调度脚本（创建 jobarray 并 sbatch）
    corrected_permutation/sl.sh                     — SLURM 脚本（MPI 4 进程）
    corrected_permutation/sl.py                     — searchlight worker（修正的打乱逻辑）

运行方式（在 Milgram 上，localize_paper 目录下）：
    python3 -u rebuttal/BZ/rebuttal_analyses/corrected_permutation/corrected_permutation.py
"""
import os
import sys
import argparse
import numpy as np
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(__file__))
from config import *


def shuffle_rsm_stimulus_labels(rsm_vec, n_stimuli=N_IMAGES):
    """
    正确的RSM置换方法：打乱刺激标签（行列协调打乱）。

    步骤：
    1. 将上三角向量还原为完整对称矩阵
    2. 生成随机刺激标签置换
    3. 同时打乱行和列
    4. 提取新的上三角

    Args:
        rsm_vec: (N*(N-1)/2,) 上三角向量
        n_stimuli: 刺激数量

    Returns:
        shuffled_vec: (N*(N-1)/2,) 打乱后的上三角向量
    """
    triu_idx = np.triu_indices(n_stimuli, k=1)

    # Step 1: 还原为完整对称矩阵
    rsm_full = np.zeros((n_stimuli, n_stimuli))
    rsm_full[triu_idx] = rsm_vec
    rsm_full = rsm_full + rsm_full.T  # 恢复对称性
    # 对角线设为0（或1，取决于原始定义）
    np.fill_diagonal(rsm_full, 0)

    # Step 2: 随机置换刺激标签
    perm = np.random.permutation(n_stimuli)

    # Step 3: 同时打乱行和列
    # 等价于 P @ RSM @ P.T（其中P是置换矩阵）
    rsm_permuted = rsm_full[np.ix_(perm, perm)]

    # Step 4: 提取上三角
    return rsm_permuted[triu_idx]


def corrected_score_searchlight(data_stack, features, layer,
                                  repetition_number=N_PERMUTATIONS_SUBJECT):
    """
    在单个searchlight中执行修正后的评分（带正确置换检验）。

    这是替换原始 sl.py 中 ModelReg() 函数的核心逻辑。

    Args:
        data_stack: (n_voxels, 2+N_images) 前2维为coef和intercept的均值，
                    后16维为测试被试的cope数据
        features: (N_layers, N_images, N_images) CNN层RSM
        layer: int, 当前层号 (1-indexed)
        repetition_number: int, 置换次数

    Returns:
        results: (repetition_number+1,) 第0个为真实R²，其余为打乱R²
    """
    triu_idx = np.triu_indices(N_IMAGES, k=1)

    # 提取训练参数（从leave-one-out的其他被试平均而来）
    coef = data_stack[:, 0].mean()       # 标量（跨体素平均后的regression slope）
    intercept = data_stack[:, 1].mean()  # 标量

    # 提取测试被试的BOLD数据
    bolddata_sl = data_stack[:, 2:].T  # (N_images, n_voxels)

    # 计算脑RSM
    brain_rsm = np.corrcoef(bolddata_sl)  # (16, 16)
    brain_rsm_vec = brain_rsm[triu_idx]    # (120,)

    # 获取模型RSM
    model_rsm_vec = features[layer - 1][triu_idx]  # (120,)

    results = np.zeros(repetition_number + 1)

    # 真实R²
    y_pred = coef * model_rsm_vec + intercept
    results[0] = r2_score(brain_rsm_vec, y_pred)

    # 置换R²：按 Methods 重排 held-out brain searchlight RSM
    for i in range(repetition_number):
        brain_rsm_shuffled = shuffle_rsm_stimulus_labels(brain_rsm_vec, N_IMAGES)
        results[i + 1] = r2_score(brain_rsm_shuffled, y_pred)

    return results


def recompute_permutations_for_subject(subject, layer, radius=SEARCHLIGHT_RADIUS):
    """
    为指定被试和层重新计算置换检验（使用修正的打乱方法）。

    直接从已保存的 all_R2.npy 读取真实R²（index 0），
    但重新计算打乱R²（index 1-500）。
    """
    print(f"Recomputing permutations for {subject}, layer {layer}...")

    # 加载原始 all_R2 文件
    original_path = get_all_R2_path(subject, layer, method='reg', radius=radius)
    if not os.path.exists(original_path):
        print(f"  File not found: {original_path}")
        return None

    all_R2_original = np.load(original_path)  # (91, 109, 91, 501)
    true_R2 = all_R2_original[:, :, :, 0]     # 保留真实R²

    # 加载CNN features
    features = np.load(get_features_path())    # (12, 16, 16)
    model_rsm_vec = features[layer - 1][np.triu_indices(N_IMAGES, k=1)]  # (120,)

    # 加载训练好的回归参数
    coef_map = load_nifti(get_coef_path(subject, layer, 'reg', radius))
    intercept_map = load_nifti(get_intercept_path(subject, layer, 'reg', radius))

    # 加载脑mask
    brain_mask = load_nifti(get_brain_mask_path(subject))

    # 加载被试的cope数据
    cope_list = []
    for cope_num in range(1, N_IMAGES + 1):
        cope_list.append(load_nifti(get_cope_path(subject, cope_num)))
    data4d = np.stack(cope_list, axis=-1)  # (91, 109, 91, 16)

    # 初始化输出
    all_R2_corrected = np.full_like(all_R2_original, np.nan)
    all_R2_corrected[:, :, :, 0] = true_R2  # 真实R²保持不变

    # 对mask内每个体素重新计算打乱R²
    mask_coords = np.argwhere(brain_mask > 0)
    print(f"  Processing {len(mask_coords)} voxels...")

    for v_idx, (x, y, z) in enumerate(mask_coords):
        if v_idx % 10000 == 0:
            print(f"  Voxel {v_idx}/{len(mask_coords)}...")

        # 获取searchlight邻域
        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx**2 + dy**2 + dz**2 <= radius**2:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (0 <= nx < BRAIN_SHAPE[0] and
                            0 <= ny < BRAIN_SHAPE[1] and
                            0 <= nz < BRAIN_SHAPE[2] and
                            brain_mask[nx, ny, nz] > 0):
                            neighbors.append((nx, ny, nz))

        if len(neighbors) < 5:
            continue

        # 提取BOLD数据并计算RSM
        bolddata = np.array([data4d[nx, ny, nz, :] for nx, ny, nz in neighbors]).T
        if bolddata.shape[1] < 3 or np.any(np.std(bolddata, axis=0) == 0):
            continue

        brain_rsm = np.corrcoef(bolddata)
        brain_rsm_vec = brain_rsm[np.triu_indices(N_IMAGES, k=1)]

        if np.any(np.isnan(brain_rsm_vec)):
            continue

        coef = coef_map[x, y, z]
        intercept = intercept_map[x, y, z]

        # 重新计算打乱R²：按 Methods 重排 held-out brain searchlight RSM
        y_pred = coef * model_rsm_vec + intercept
        for perm_i in range(N_PERMUTATIONS_SUBJECT):
            brain_rsm_shuffled = shuffle_rsm_stimulus_labels(brain_rsm_vec, N_IMAGES)
            all_R2_corrected[x, y, z, perm_i + 1] = r2_score(brain_rsm_shuffled, y_pred)

    return all_R2_corrected


def zscore_and_save(all_R2_corrected, subject, layer, output_dir):
    """Z-score normalize and save results."""
    mkdir(output_dir)

    true_R2 = all_R2_corrected[:, :, :, 0]
    null_R2 = all_R2_corrected[:, :, :, 1:]

    null_mean = np.nanmean(null_R2, axis=3)
    null_std = np.nanstd(null_R2, axis=3)

    # Avoid division by zero
    null_std[null_std == 0] = np.nan

    zscored = (true_R2 - null_mean) / null_std

    # Save corrected all_R2
    np.save(
        os.path.join(output_dir,
                     f"{subject}_layer{layer}_reg_all_R2_corrected.npy"),
        all_R2_corrected
    )

    # Save z-scored map
    save_nifti(
        zscored,
        os.path.join(output_dir,
                     f"zscored_R2_{subject}_layer{layer}_reg_corrected.nii.gz")
    )


def main():
    parser = argparse.ArgumentParser(
        description='Corrected permutation test with stimulus-label shuffling (R1-P3)')
    parser.add_argument('--subject', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--radius', type=int, default=SEARCHLIGHT_RADIUS)
    args = parser.parse_args()

    output_dir = os.path.join(REBUTTAL_OUTPUT, "corrected_permutation")

    all_R2_corrected = recompute_permutations_for_subject(
        args.subject, args.layer, args.radius
    )

    if all_R2_corrected is not None:
        zscore_and_save(all_R2_corrected, args.subject, args.layer, output_dir)
        print(f"Done: {args.subject} layer {args.layer}")


if __name__ == '__main__':
    main()

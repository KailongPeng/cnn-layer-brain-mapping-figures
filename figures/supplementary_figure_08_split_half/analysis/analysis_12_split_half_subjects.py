"""
分析十二：按被试分半的层选择性可靠性分析（Jeff 建议）
回应审稿意见：R1-P5（overlap 是信号还是噪声？）

逻辑：
  - 把 30 个被试随机分成两组各 15 人
  - 每组独立算每层的 group Stouffer z-score map（仍用全部 16 张图）
  - 算 within-layer 相关：同层两半的 group_z 空间模式 Pearson r
  - 算 between-layer 相关：不同层两半的 group_z Pearson r
  - 如果 within-layer r ≫ between-layer r → 层选择性是真实信号
  - 如果 within-layer r ≈ between-layer r → 审稿人"全是噪声"有道理

  为减少单次随机分组的偶然性，重复 100 次随机分组，报告分布。

数据来源：
  corrected_permutation 的 all_R2.npy（每被试每层 501 个 R² 值）
  路径由 config.get_corrected_perm_all_R2_path() 给出

用法：
    python3 analysis_12_split_half_subjects.py
    python3 analysis_12_split_half_subjects.py --n_splits 50
    python3 analysis_12_split_half_subjects.py --viz_only   # 只画图，跳过计算

输出目录：rebuttal_analyses/split_half_subjects/
"""
import os
import sys
import argparse
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import *


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_FIG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "figures", "components"))
REPO_RESULTS_PATH = os.path.join(
    SCRIPT_DIR, "split_half_data", "split_half_subjects_results.npz")
REPO_PDF_PATH = os.path.join(PAPER_FIG_DIR, "split_half_subjects.pdf")
REPO_PNG_PATH = os.path.join(PAPER_FIG_DIR, "..", "previews", "split_half_subjects.png")
STAGED_PDF_PATH = os.path.join(
    PAPER_FIG_DIR, "..", "final",
    "Sfig8_Split_half_reliability_across_participants.pdf")
WITHIN_LAYER_COLOR = '#2A9D8F'
BETWEEN_LAYER_COLOR = '#8E5A9B'


def load_and_validate_results(path):
    """加载 split-half 缓存，并在作图前严格核对字段和形状。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[数据门禁] split-half 缓存不存在：{path}")
    with np.load(path) as data:
        required = {'within_layer', 'between_layer', 'n_splits'}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"[数据门禁] {path} 缺少字段：{sorted(missing)}")
        within = np.asarray(data['within_layer'], dtype=float)
        between = np.asarray(data['between_layer'], dtype=float)
        n_splits = int(data['n_splits'])

    expected_within = (n_splits, N_LAYERS)
    expected_between = (n_splits, N_LAYERS, N_LAYERS)
    if n_splits < 1 or within.shape != expected_within or between.shape != expected_between:
        raise ValueError(
            f"[数据门禁] 缓存形状不正确：n_splits={n_splits}, "
            f"within={within.shape}（应为 {expected_within}）, "
            f"between={between.shape}（应为 {expected_between}）")
    if np.any(np.sum(np.isfinite(within), axis=0) == 0):
        raise ValueError("[数据门禁] 至少一个 CNN layer 没有有限的 within-layer 值")
    for split_i, matrix in enumerate(between):
        offdiag = matrix[~np.eye(N_LAYERS, dtype=bool)]
        if not np.any(np.isfinite(offdiag)):
            raise ValueError(f"[数据门禁] split {split_i} 没有有限的 between-layer 非对角值")
    print(f"[数据门禁] split-half 缓存 OK：{path} ({os.path.getsize(path):,} bytes)")
    return within, between, n_splits


def sync_repo_results(source_path):
    """把已验证的轻量 .npz 缓存同步进 Git 跟踪目录。"""
    results = load_and_validate_results(source_path)
    os.makedirs(os.path.dirname(REPO_RESULTS_PATH), exist_ok=True)
    if os.path.abspath(source_path) != os.path.abspath(REPO_RESULTS_PATH):
        shutil.copyfile(source_path, REPO_RESULTS_PATH)
        with open(source_path, 'rb') as src, open(REPO_RESULTS_PATH, 'rb') as dst:
            if src.read() != dst.read():
                raise RuntimeError("[数据门禁] 仓库 .npz 与 scratch 源文件字节不一致")
        print(f"[缓存同步] 已复制到 Git 跟踪路径：{REPO_RESULTS_PATH}")
    return results


def resolve_plot_results(output_dir):
    """仓库缓存优先；若尚未入库，显式读取 scratch、验证并同步。"""
    if os.path.isfile(REPO_RESULTS_PATH):
        print(f"[数据源] 使用仓库缓存：{REPO_RESULTS_PATH}")
        return load_and_validate_results(REPO_RESULTS_PATH)
    scratch_path = os.path.join(output_dir, "split_half_subjects_results.npz")
    if not os.path.isfile(scratch_path):
        raise FileNotFoundError(
            "[数据源门禁] 仓库与 scratch 均无 split-half 缓存：\n"
            f"  repo:    {REPO_RESULTS_PATH}\n"
            f"  scratch: {scratch_path}\n"
            "请先运行完整 analysis_12_split_half_subjects.py。")
    print(f"[数据源] 仓库缓存尚未建立；显式使用 scratch：{scratch_path}")
    return sync_repo_results(scratch_path)


def validate_pdf(path):
    if not os.path.isfile(path) or os.path.getsize(path) < 5_000:
        raise RuntimeError(f"[输出门禁] PDF 缺失或异常小：{path}")
    size = os.path.getsize(path)
    with open(path, 'rb') as handle:
        head = handle.read(8)
        handle.seek(max(0, size - 1024))
        tail = handle.read()
    if not head.startswith(b'%PDF-') or b'%%EOF' not in tail:
        raise RuntimeError(f"[输出门禁] PDF 文件头或 EOF 不完整：{path}")
    print(f"[输出门禁] PDF OK：{path} ({size:,} bytes)")


def load_subject_z_scores(subjects, layer, radius=SEARCHLIGHT_RADIUS):
    """
    加载所有被试在某层的 per-subject z_i。

    z_i = (true_R2 - null_mean) / null_std

    Returns:
        z_maps: (N_subs, 91, 109, 91) array, NaN 表示无效体素
        valid_subs: 成功加载的被试列表
    """
    z_list = []
    valid_subs = []

    for sub in subjects:
        perm_path = get_corrected_perm_all_R2_path(sub, layer, radius)
        if not os.path.exists(perm_path):
            print(f"  [{sub}] 文件缺失，跳过：{perm_path}")
            continue

        all_R2 = np.load(perm_path)           # (91,109,91,501)
        true_R2 = all_R2[:, :, :, 0]
        null_R2 = all_R2[:, :, :, 1:]         # (91,109,91,500)
        null_mean = null_R2.mean(axis=-1)
        null_std = null_R2.std(axis=-1)
        del all_R2, null_R2

        with np.errstate(invalid='ignore', divide='ignore'):
            z_i = np.where(null_std > 0, (true_R2 - null_mean) / null_std, np.nan)

        z_list.append(z_i)
        valid_subs.append(sub)

    if len(z_list) == 0:
        return None, []

    z_maps = np.stack(z_list, axis=0)  # (N_valid, 91, 109, 91)
    return z_maps, valid_subs


def compute_group_z(z_maps, sub_indices):
    """
    从 z_maps 中选取指定被试，计算 Stouffer group z。

    group_z = sum(z_i) / sqrt(N)

    Args:
        z_maps: (N_all, 91, 109, 91)
        sub_indices: list/array of indices into z_maps

    Returns:
        group_z: (91, 109, 91)
    """
    selected = z_maps[sub_indices]  # (N_half, 91, 109, 91)

    with np.errstate(invalid='ignore'):
        z_sum = np.nansum(selected, axis=0)
        n_valid = np.sum(np.isfinite(selected), axis=0)
        group_z = np.where(n_valid > 0, z_sum / np.sqrt(n_valid.astype(float)), np.nan)

    return group_z


def run_split_half_analysis(n_splits=100, radius=SEARCHLIGHT_RADIUS, output_dir=None):
    """
    主计算函数。

    Phase 1: 逐层加载所有被试的 z_i，对每次随机分组计算 group_z，
             只保留 brain_mask 内的扁平向量
    Phase 2: 计算 within-layer 和 between-layer 空间相关

    Returns:
        results dict 存入 split_half_subjects_results.npz
    """
    # 完整重算才需要 SciPy；--viz_only 只依赖已跟踪的轻量 .npz。
    from scipy.stats import pearsonr

    subjects = SYNTHESIZED_SUBS
    rng = np.random.default_rng(42)
    n_subs = len(subjects)
    n_half = n_subs // 2  # 15

    # 预生成所有分组方案（固定种子，可复现）
    splits = []
    for _ in range(n_splits):
        perm = rng.permutation(n_subs)
        half_a = np.sort(perm[:n_half])
        half_b = np.sort(perm[n_half:])
        splits.append((half_a, half_b))

    # 确定 brain mask
    print("确定 brain mask ...", flush=True)
    test_path = get_corrected_perm_all_R2_path(subjects[0], 1, radius)
    if not os.path.exists(test_path):
        print(f"错误：{test_path} 不存在，请先运行 analysis_03")
        return None
    test_data = np.load(test_path)
    brain_mask = np.isfinite(test_data[:, :, :, 0]) & (test_data[:, :, :, 0] != 0)
    n_brain_voxels = int(brain_mask.sum())
    del test_data
    print(f"Brain mask: {n_brain_voxels:,} voxels", flush=True)

    # Phase 1: 逐层加载，生成扁平 group_z 向量
    # 存储 group_z_flat[layer_idx, split_idx, half_idx, voxel_idx]
    # 内存: 12 × 100 × 2 × ~200K × 4 bytes ≈ 1.9 GB
    print(f"\n分配内存：{N_LAYERS} × {n_splits} splits × 2 halves × "
          f"{n_brain_voxels:,} voxels ...", flush=True)
    group_z_flat = np.zeros((N_LAYERS, n_splits, 2, n_brain_voxels), dtype=np.float32)

    for layer in range(1, N_LAYERS + 1):
        print(f"\n=== Layer {layer}/{N_LAYERS} ===", flush=True)

        z_maps, valid_subs = load_subject_z_scores(subjects, layer, radius)
        if z_maps is None:
            print(f"  Layer {layer}: 无数据，跳过")
            continue

        n_valid = len(valid_subs)
        print(f"  成功加载 {n_valid}/{n_subs} 被试", flush=True)

        # 如果有缺失被试，重新映射索引
        if n_valid != n_subs:
            print(f"  警告：{n_valid} 个被试有数据（共 {n_subs}），重建索引映射")
            sub_to_idx = {subjects.index(sub): z_idx
                          for z_idx, sub in enumerate(valid_subs)}
        else:
            sub_to_idx = None

        for s_idx, (half_a, half_b) in enumerate(splits):
            if sub_to_idx is not None:
                mapped_a = [sub_to_idx[i] for i in half_a if i in sub_to_idx]
                mapped_b = [sub_to_idx[i] for i in half_b if i in sub_to_idx]
            else:
                mapped_a = half_a
                mapped_b = half_b

            gz_a = compute_group_z(z_maps, mapped_a)
            gz_b = compute_group_z(z_maps, mapped_b)

            group_z_flat[layer - 1, s_idx, 0, :] = gz_a[brain_mask].astype(np.float32)
            group_z_flat[layer - 1, s_idx, 1, :] = gz_b[brain_mask].astype(np.float32)

        del z_maps
        print(f"  Layer {layer} 完成", flush=True)

    # Phase 2: 计算 within-layer 和 between-layer 相关
    print(f"\n=== Phase 2: 计算空间相关 ===", flush=True)

    within_layer = np.full((n_splits, N_LAYERS), np.nan)
    between_layer = np.full((n_splits, N_LAYERS, N_LAYERS), np.nan)

    for s_idx in range(n_splits):
        if (s_idx + 1) % 10 == 0:
            print(f"  Split {s_idx + 1}/{n_splits} ...", flush=True)

        for li in range(N_LAYERS):
            vec_a_li = group_z_flat[li, s_idx, 0, :]
            vec_b_li = group_z_flat[li, s_idx, 1, :]

            # Within-layer：同层 half_A vs half_B
            valid = np.isfinite(vec_a_li) & np.isfinite(vec_b_li)
            if valid.sum() > 100:
                within_layer[s_idx, li] = pearsonr(
                    vec_a_li[valid], vec_b_li[valid]
                )[0]

            # Between-layer：half_A(layer_i) vs half_B(layer_j)
            for lj in range(N_LAYERS):
                vec_b_lj = group_z_flat[lj, s_idx, 1, :]
                valid_bw = np.isfinite(vec_a_li) & np.isfinite(vec_b_lj)
                if valid_bw.sum() > 100:
                    between_layer[s_idx, li, lj] = pearsonr(
                        vec_a_li[valid_bw], vec_b_lj[valid_bw]
                    )[0]

    # 保存结果（npz，无需 pickle）
    mkdir(output_dir)
    results_path = os.path.join(output_dir, "split_half_subjects_results.npz")
    np.savez(
        results_path,
        within_layer=within_layer,
        between_layer=between_layer,
        n_splits=np.array(n_splits),
        n_brain_voxels=np.array(n_brain_voxels),
    )
    sync_repo_results(results_path)
    save_nifti(brain_mask.astype(float),
               os.path.join(output_dir, "brain_mask.nii.gz"))

    print_summary(within_layer, between_layer, n_splits)
    return within_layer, between_layer


def print_summary(within_layer, between_layer, n_splits):
    print(f"\n{'='*60}")
    print(f"Split-Half by Subjects: Results Summary")
    print(f"{'='*60}")
    print(f"N splits: {n_splits}")

    print(f"\n--- Within-layer reliability (same layer, half_A vs half_B) ---")
    print(f"{'Layer':>6}  {'Mean r':>8}  {'SD':>8}  {'Min':>8}  {'Max':>8}")
    print(f"{'-'*42}")
    for l in range(N_LAYERS):
        vals = within_layer[:, l]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            print(f"  {l+1:2d}    {np.mean(vals):+.4f}  {np.std(vals):.4f}  "
                  f"{np.min(vals):+.4f}  {np.max(vals):+.4f}")

    within_mean_all = np.nanmean(within_layer)

    between_offdiag = []
    for s in range(between_layer.shape[0]):
        mat = between_layer[s]
        offdiag = mat[~np.eye(N_LAYERS, dtype=bool)]
        offdiag = offdiag[np.isfinite(offdiag)]
        if len(offdiag) > 0:
            between_offdiag.append(np.mean(offdiag))
    between_mean = np.mean(between_offdiag)
    between_sd = np.std(between_offdiag)

    print(f"\n--- Between-layer correlation ---")
    print(f"  Off-diagonal mean r: {between_mean:+.4f} ± {between_sd:.4f}")
    ratio = within_mean_all / between_mean if between_mean != 0 else float('inf')
    print(f"\n  Within-layer mean r: {within_mean_all:+.4f}")
    print(f"  Within / Between ratio: {ratio:.1f}×")
    print(f"{'='*60}")


def plot_results(output_dir):
    """从已验证的 .npz 生成只含柱状图的正式补充图。"""
    within, between, n_splits = resolve_plot_results(output_dir)

    between_offdiag_per_split = []
    for s in range(n_splits):
        mat = between[s]
        offdiag = mat[~np.eye(N_LAYERS, dtype=bool)]
        offdiag = offdiag[np.isfinite(offdiag)]
        if len(offdiag) > 0:
            between_offdiag_per_split.append(np.mean(offdiag))
    between_mean = np.mean(between_offdiag_per_split)
    between_sd = np.std(between_offdiag_per_split)

    if len(between_offdiag_per_split) != n_splits:
        raise RuntimeError(
            f"[数据门禁] 仅得到 {len(between_offdiag_per_split)}/{n_splits} 个 "
            "between-layer split 均值")

    # Nick/Jeff：正式补充图只保留左侧柱状图，删除统计单位不一致的右侧直方图。
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    layer_means = np.nanmean(within, axis=0)
    layer_sds = np.nanstd(within, axis=0)
    layers = np.arange(1, N_LAYERS + 1)

    within_bars = ax.bar(
        layers, layer_means, yerr=layer_sds, capsize=3,
        color=WITHIN_LAYER_COLOR, edgecolor='white', linewidth=0.5, alpha=0.9,
        label='Within-layer')

    between_line = ax.axhline(
        between_mean, color=BETWEEN_LAYER_COLOR, linewidth=2, linestyle='--',
        label='Between-layer')
    ax.axhspan(between_mean - between_sd, between_mean + between_sd,
               color=BETWEEN_LAYER_COLOR, alpha=0.15)
    # Jeff: “Will need an x axis line.” 由于 y 轴下探到负值，底部 spine 不在 r=0；
    # 单独画清楚的零基线，并与 r=0.019 的梅紫色虚线区分。
    ax.axhline(0.0, color='#333333', linewidth=1.0, zorder=3)

    ax.set_xlabel("CNN Layer", fontsize=12)
    ax.set_ylabel("Pearson r (split-half reliability)", fontsize=12)
    ax.set_xticks(layers)
    # Headroom above the tallest error bar so the legend has room and doesn't
    # overlap the bars (title and split/subject counts moved to the caption).
    top = np.nanmax(layer_means + layer_sds)
    bottom = 0.0
    ax.set_ylim(bottom, top + 0.35 * (top - bottom))
    ax.legend(
        handles=[between_line, within_bars],
        labels=['Between-layer', 'Within-layer'],
        loc='upper right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['bottom'].set_linewidth(1.0)

    plt.tight_layout()
    if os.path.isdir(output_dir) and os.access(output_dir, os.W_OK):
        primary_pdf = os.path.join(output_dir, "split_half_subjects.pdf")
        print(f"[输出路径] 使用可写的分析目录：{primary_pdf}")
    else:
        primary_pdf = REPO_PDF_PATH
        print(f"[输出路径] 分析目录在本机不可写：{output_dir}")
        print(f"[输出路径] 显式直接写入仓库：{primary_pdf}")
    os.makedirs(os.path.dirname(primary_pdf), exist_ok=True)
    os.makedirs(os.path.dirname(REPO_PNG_PATH), exist_ok=True)
    fig.savefig(primary_pdf, bbox_inches='tight')
    fig.savefig(REPO_PNG_PATH, dpi=200, bbox_inches='tight')

    plt.close()
    validate_pdf(primary_pdf)
    for destination in (REPO_PDF_PATH, STAGED_PDF_PATH):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if os.path.abspath(primary_pdf) != os.path.abspath(destination):
            shutil.copyfile(primary_pdf, destination)
        validate_pdf(destination)
        with open(primary_pdf, 'rb') as src, open(destination, 'rb') as dst:
            if src.read() != dst.read():
                raise RuntimeError(
                    f"[输出门禁] {destination} 与主 PDF 字节不一致")
    print(f"Saved preview: {REPO_PNG_PATH}")

    print(f"\n=== Key Numbers for Rebuttal ===")
    print(f"Within-layer mean r:  {np.nanmean(within):+.4f}")
    print(f"Between-layer mean r: {between_mean:+.4f} ± {between_sd:.4f}")
    ratio = np.nanmean(within) / between_mean if between_mean != 0 else float('inf')
    print(f"Ratio (within/between): {ratio:.1f}×")


def main():
    parser = argparse.ArgumentParser(
        description='Split-half by subjects: layer selectivity reliability (Jeff suggestion)')
    parser.add_argument('--n_splits', type=int, default=100,
                        help='Number of random split-half iterations (default: 100)')
    parser.add_argument('--radius', type=int, default=SEARCHLIGHT_RADIUS)
    parser.add_argument('--viz_only', action='store_true',
                        help='Skip computation, only generate plots from saved results')
    args = parser.parse_args()

    output_dir = os.path.join(REBUTTAL_OUTPUT, "split_half_subjects")

    if args.viz_only:
        plot_results(output_dir)
        return

    results = run_split_half_analysis(
        n_splits=args.n_splits,
        radius=args.radius,
        output_dir=output_dir
    )

    if results is not None:
        plot_results(output_dir)


if __name__ == '__main__':
    main()

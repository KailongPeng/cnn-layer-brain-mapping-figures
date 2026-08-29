"""
分析十：噪声天花板归一化 + 修正置换检验显著性图
回应审稿意见：R1-P1（噪声天花板）+ R1-P3（修正置换）

运行模式：
  单层（SLURM array）：
    python3 analysis_10_noise_ceiling_viz.py --dataset synthesized --layer 6
  聚合（所有层完成后，登录节点）：
    python3 analysis_10_noise_ceiling_viz.py --dataset synthesized --aggregate
  调度（自动提交 array + 等待 + 聚合）：
    python3 analysis_10_dispatcher.py

输出目录：rebuttal_analyses/noise_ceiling_viz/
  单层输出（--layer N）：
    normalized_R2_layer{N}_synthesized.nii.gz
    pvalue_layer{N}_synthesized.nii.gz
    pvalue_fdr_layer{N}_synthesized.nii.gz
    masked_R2_layer{N}_synthesized.nii.gz        ← FSLeyes 主要看这个
  聚合输出（--aggregate）：
    best_layer_masked_synthesized.nii.gz
    max_normalized_R2_masked_synthesized.nii.gz
"""
import os
import sys
import argparse
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from config import *

N_PERM   = 500
FDR_ALPHA = 0.05


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR 校正，输入输出均为 1D array（脑内掩膜体素）。"""
    n = len(pvals)
    order   = np.argsort(pvals)
    adj     = np.minimum(1.0, pvals[order] * n / (np.arange(1, n + 1)))
    adj     = np.minimum.accumulate(adj[::-1])[::-1]
    result  = np.empty(n)
    result[order] = adj
    return result


def compute_layer(layer, subjects, nc_lower, suffix, radius, fdr_alpha, output_dir):
    """
    处理单层，直接写出 4 个 NIfTI 文件。
    输出：True（成功）/ False（无数据）
    """
    tag = f"layer{layer}_{suffix}"

    # ── 逐被试 Stouffer z-score + 累积 mean true_R² ───────────────────────────
    z_sum    = np.zeros(BRAIN_SHAPE, dtype=np.float64)
    r2_sum   = np.zeros(BRAIN_SHAPE, dtype=np.float64)
    n_subs   = np.zeros(BRAIN_SHAPE, dtype=np.int32)
    n_r2     = np.zeros(BRAIN_SHAPE, dtype=np.int32)

    for idx, sub in enumerate(subjects):
        perm_path = get_corrected_perm_all_R2_path(sub, layer, radius)
        if not os.path.exists(perm_path):
            print(f"  [{sub} {idx+1}/{len(subjects)}] 文件缺失，跳过：{perm_path}", flush=True)
            continue

        file_mb = os.path.getsize(perm_path) / 1e6
        print(f"  [{sub} {idx+1}/{len(subjects)}] 读取 {file_mb:.0f} MB ...", flush=True)

        all_R2    = np.load(perm_path)          # (91,109,91,501)
        true_R2   = all_R2[:, :, :, 0]
        null_R2   = all_R2[:, :, :, 1:]         # (91,109,91,500)
        null_mean = null_R2.mean(axis=-1)
        null_std  = null_R2.std(axis=-1)
        del all_R2, null_R2

        with np.errstate(invalid='ignore', divide='ignore'):
            z_i = np.where(null_std > 0, (true_R2 - null_mean) / null_std, np.nan)

        valid_z = np.isfinite(z_i)
        z_sum[valid_z]  += z_i[valid_z]
        n_subs[valid_z] += 1

        # 累积 true_R² 用于计算 mean_R²（只取有限值）
        valid_r2 = np.isfinite(true_R2)
        r2_sum[valid_r2]  += true_R2[valid_r2]
        n_r2[valid_r2]    += 1
        print(f"  [{sub}] 有效体素 {valid_z.sum():,}", flush=True)

    # ── 计算 mean R² 并归一化 ─────────────────────────────────────────────────
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_R2      = np.where(n_r2 > 0, r2_sum / n_r2, np.nan)
        normalized_R2 = np.where(nc_lower > 0, mean_R2 / nc_lower, np.nan)

    print(f"[Layer {layer}] mean_R2 max={np.nanmax(mean_R2):.4f}  "
          f"normalized_R2 max={np.nanmax(normalized_R2):.4f}", flush=True)

    # ── 组级 z → p ────────────────────────────────────────────────────────────
    with np.errstate(invalid='ignore', divide='ignore'):
        group_z = np.where(n_subs > 0, z_sum / np.sqrt(n_subs.astype(float)), np.nan)

    brain_mask = np.isfinite(nc_lower) & (nc_lower != 0)
    p_raw  = np.full(BRAIN_SHAPE, np.nan)
    valid  = np.isfinite(group_z)
    p_raw[valid] = 1.0 - norm.cdf(group_z[valid])

    # ── FDR 校正（只在脑内掩膜内）─────────────────────────────────────────────
    p_fdr = np.full(BRAIN_SHAPE, np.nan)
    mask_valid = brain_mask & valid
    if mask_valid.sum() > 0:
        p_fdr[mask_valid] = fdr_bh(p_raw[mask_valid], alpha=fdr_alpha)

    sig_mask = p_fdr < fdr_alpha
    n_sig    = int(np.nansum(sig_mask))
    n_brain  = int(brain_mask.sum())
    print(f"\n[Layer {layer}] 组级 z mean={np.nanmean(group_z):.3f}  "
          f"FDR<{fdr_alpha} 显著: {n_sig:,} / {n_brain:,} ({100*n_sig/max(n_brain,1):.1f}%)",
          flush=True)

    # ── masked normalized_R² ──────────────────────────────────────────────────
    masked_R2 = np.where(sig_mask, normalized_R2, np.nan)

    # ── 保存 ─────────────────────────────────────────────────────────────────
    save_nifti(normalized_R2, os.path.join(output_dir, f"normalized_R2_{tag}.nii.gz"))
    save_nifti(group_z,       os.path.join(output_dir, f"group_z_{tag}.nii.gz"))
    save_nifti(p_raw,         os.path.join(output_dir, f"pvalue_{tag}.nii.gz"))
    save_nifti(p_fdr,         os.path.join(output_dir, f"pvalue_fdr_{tag}.nii.gz"))
    save_nifti(masked_R2,     os.path.join(output_dir, f"masked_R2_{tag}.nii.gz"))
    print(f"[Layer {layer}] 完成，5 个 NIfTI 已保存", flush=True)
    return True


def aggregate(suffix, output_dir):
    """
    读取所有层的 masked_R2 NIfTI，生成 best_layer 和 max_R2 图。
    在登录节点快速运行（NIfTI 文件小）。
    """
    print("聚合：生成 best_layer 和 max_R2 图 ...", flush=True)
    R2_stack = []
    available_layers = []

    for layer in range(1, N_LAYERS + 1):
        path = os.path.join(output_dir, f"masked_R2_layer{layer}_{suffix}.nii.gz")
        if os.path.exists(path):
            R2_stack.append(load_nifti(path))
            available_layers.append(layer)
            print(f"  加载 Layer {layer}", flush=True)
        else:
            print(f"  Layer {layer} 缺失：{path}", flush=True)
            R2_stack.append(np.full(BRAIN_SHAPE, np.nan))

    if not available_layers:
        print("没有找到任何层的结果，退出", flush=True)
        return

    stack = np.stack(R2_stack, axis=-1)          # (91,109,91,12)
    any_sig = np.any(np.isfinite(stack), axis=-1)

    best_layer_map = np.full(BRAIN_SHAPE, np.nan)
    max_R2_map     = np.full(BRAIN_SHAPE, np.nan)

    if any_sig.sum() > 0:
        R2_for_argmax = np.where(np.isfinite(stack), stack, -np.inf)
        best_idx = np.argmax(R2_for_argmax, axis=-1)

        xi, yi, zi = np.where(any_sig)
        best_layer_map[xi, yi, zi] = (best_idx[xi, yi, zi] + 1).astype(float)
        max_R2_map[xi, yi, zi]     = stack[xi, yi, zi, best_idx[xi, yi, zi]]

    save_nifti(best_layer_map,
               os.path.join(output_dir, f"best_layer_masked_{suffix}.nii.gz"))
    save_nifti(max_R2_map,
               os.path.join(output_dir, f"max_normalized_R2_masked_{suffix}.nii.gz"))
    print(f"聚合完成。有效被试层数：{len(available_layers)}/{N_LAYERS}", flush=True)
    print(f"在 {any_sig.sum():,} 个体素上有显著激活", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   type=str, default='synthesized',
                        choices=['synthesized'])
    parser.add_argument('--layer',     type=int, default=None,
                        help='仅处理该层（SLURM array 用），不做聚合')
    parser.add_argument('--aggregate', action='store_true',
                        help='跳过计算，只做跨层聚合（所有层完成后在登录节点运行）')
    parser.add_argument('--fdr_alpha', type=float, default=FDR_ALPHA)
    parser.add_argument('--radius',    type=int,   default=SEARCHLIGHT_RADIUS)
    args = parser.parse_args()

    output_dir = os.path.join(REBUTTAL_OUTPUT, "noise_ceiling_viz")
    mkdir(output_dir)

    suffix   = args.dataset
    subjects = SYNTHESIZED_SUBS

    # ── 聚合模式 ──────────────────────────────────────────────────────────────
    if args.aggregate:
        aggregate(suffix, output_dir)
        return

    # ── 加载 noise ceiling ────────────────────────────────────────────────────
    nc_lower_path = get_noise_ceiling_path(suffix, bound='lower')
    if not os.path.exists(nc_lower_path):
        print(f"错误：{nc_lower_path} 不存在，请先运行 noise_ceiling/noise_ceiling.py")
        return
    nc_lower = load_nifti(nc_lower_path)
    print(f"Noise ceiling lower: mean={np.nanmean(nc_lower):.4f}, "
          f"max={np.nanmax(nc_lower):.4f}", flush=True)

    # ── 单层模式 ──────────────────────────────────────────────────────────────
    if args.layer is not None:
        print(f"=== 单层模式：Layer {args.layer} ===", flush=True)
        compute_layer(args.layer, subjects, nc_lower, suffix,
                      args.radius, args.fdr_alpha, output_dir)
        return

    # ── 顺序全跑（登录节点 fallback，不推荐）────────────────────────────────
    print("顺序处理全部 12 层（建议改用 analysis_10_dispatcher.py）", flush=True)
    for layer in range(1, N_LAYERS + 1):
        compute_layer(layer, subjects, nc_lower, suffix,
                      args.radius, args.fdr_alpha, output_dir)
    aggregate(suffix, output_dir)


if __name__ == '__main__':
    main()

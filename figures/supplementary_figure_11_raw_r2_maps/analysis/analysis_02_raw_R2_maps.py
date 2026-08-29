"""
分析二：效应量图（Stouffer group z-score）
回应审稿意见：R1-P2, R1-P4

从 analysis_10 已生成的 group_z_layer{N}_synthesized.nii.gz 文件读取
Stouffer z-score（跨30被试的聚合z，是有意义的连续效应大小指标），
直接展示为连续色标脑图。不做 TFCE，不做阈值化。

注意：cross-validated R² 绝对值在 searchlight 中因边界效应和零体素稀释
通常为大量负数（如 -27），不适合直接展示。Per-subject z-score 再 Stouffer
聚合后的 group_z 才是有意义的效应大小。

用法：
    python analysis_02_raw_R2_maps.py --dataset synthesized
    python analysis_02_raw_R2_maps.py --dataset nsd
    python analysis_02_raw_R2_maps.py --dataset nsd_residual
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import *

# analysis_10 输出目录（group_z NIfTI 文件在此）
NC_VIZ_DIR = os.path.join(REBUTTAL_OUTPUT, "noise_ceiling_viz")


def extract_raw_R2_synthesized(subjects, radius=SEARCHLIGHT_RADIUS):
    """
    Load Stouffer group z-score maps from analysis_10 output.
    These are computed as: group_z = sum(z_i) / sqrt(N), where
    z_i = (true_R2_i - null_mean_i) / null_std_i per subject.

    Returns:
        R2_maps: dict {layer: (91,109,91) array of group z-score}
    """
    R2_maps = {}

    for layer in range(1, N_LAYERS + 1):
        gz_path = os.path.join(NC_VIZ_DIR, f"group_z_layer{layer}_synthesized.nii.gz")
        if not os.path.exists(gz_path):
            print(f"  Layer {layer}: group_z file missing: {gz_path}")
            continue

        gz = load_nifti(gz_path)  # (91, 109, 91)
        R2_maps[layer] = {
            'mean': gz,
            'n_subjects': len(subjects)
        }
        print(f"  Layer {layer}: group_z mean={np.nanmean(gz):.4f}, "
              f"max={np.nanmax(gz):.4f}, "
              f"n_nonzero={np.sum(gz != 0)}")

    return R2_maps


def extract_raw_R2_nsd(subjects, n_sets=N_NSD_SETS, radius=SEARCHLIGHT_RADIUS,
                        residual=False):
    """
    Extract cross-validated R² for NSD images.
    Average across subjects and across NSD sets for each layer.

    For non-residual (residual=False): loads all_R2.npy (index 0 = true R²)
    For residual (residual=True): loads zscored_R2.nii.gz (already z-scored,
        no index 0 extraction needed — the residual pipeline does not save all_R2.npy)

    Returns:
        R2_maps: dict {layer: {'mean': array, 'per_set': list of arrays}}
    """
    R2_maps = {}

    for layer in range(1, N_LAYERS + 1):
        print(f"Processing Layer {layer} ({'residual' if residual else 'non-residual'})...")
        R2_per_set = []

        for which_set in range(1, n_sets + 1):
            R2_per_subject = []
            for sub in subjects:
                if residual:
                    # Residual pipeline only saves zscored_R2, not all_R2.npy
                    zscored_path = get_nsd_residual_zscored_path(
                        sub, layer, which_set, method='reg', radius=radius
                    )
                    if not os.path.exists(zscored_path):
                        continue
                    true_R2 = load_nifti(zscored_path)  # Already z-scored R²
                else:
                    zscored_path = get_nsd_zscored_path(
                        sub, layer, which_set, method='reg', radius=radius
                    )
                    if not os.path.exists(zscored_path):
                        continue
                    true_R2 = load_nifti(zscored_path)
                R2_per_subject.append(true_R2)

            if len(R2_per_subject) > 0:
                R2_set_mean = np.nanmean(np.stack(R2_per_subject, axis=-1), axis=-1)
                R2_per_set.append(R2_set_mean)

        if len(R2_per_set) == 0:
            print(f"  No data found for Layer {layer}")
            continue

        # Average across NSD sets
        R2_all_sets = np.stack(R2_per_set, axis=-1)
        R2_grand_mean = np.nanmean(R2_all_sets, axis=-1)

        R2_maps[layer] = {
            'mean': R2_grand_mean,
            'per_set': R2_per_set,
            'n_sets': len(R2_per_set)
        }

        print(f"  Layer {layer}: mean R²={np.nanmean(R2_grand_mean):.6f}, "
              f"max R²={np.nanmax(R2_grand_mean):.6f}, "
              f"N_sets={len(R2_per_set)}")

    return R2_maps


def save_R2_maps(R2_maps, output_dir, suffix):
    """Save all layer R² maps as NIfTI files."""
    mkdir(output_dir)

    for layer, data in R2_maps.items():
        R2_mean = data['mean']

        # Save raw (unthresholded) R² map
        save_nifti(
            R2_mean,
            os.path.join(output_dir, f"R2_raw_layer{layer}_{suffix}.nii.gz")
        )

        # Also save SE if available
        if 'se' in data:
            save_nifti(
                data['se'],
                os.path.join(output_dir, f"R2_se_layer{layer}_{suffix}.nii.gz")
            )

    # Save the best-layer map (argmax R² across layers, for later analyses)
    all_layers = sorted(R2_maps.keys())
    R2_stack = np.stack([R2_maps[l]['mean'] for l in all_layers], axis=-1)
    best_layer_map = np.argmax(R2_stack, axis=-1) + 1  # 1-indexed layer number
    max_R2_map = np.max(R2_stack, axis=-1)

    save_nifti(
        best_layer_map.astype(float),
        os.path.join(output_dir, f"best_layer_{suffix}.nii.gz")
    )
    save_nifti(
        max_R2_map,
        os.path.join(output_dir, f"max_R2_{suffix}.nii.gz")
    )

    print(f"\nSaved {len(R2_maps)} layer maps + best_layer + max_R2 to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Raw R² effect size maps (R1-P2)')
    parser.add_argument('--dataset', type=str, default='synthesized',
                        choices=['synthesized', 'nsd', 'nsd_residual'])
    parser.add_argument('--radius', type=int, default=SEARCHLIGHT_RADIUS)
    args = parser.parse_args()

    output_dir = os.path.join(REBUTTAL_OUTPUT, "raw_R2_maps")

    if args.dataset == 'synthesized':
        R2_maps = extract_raw_R2_synthesized(SYNTHESIZED_SUBS, args.radius)
        save_R2_maps(R2_maps, output_dir, "synthesized")
    elif args.dataset == 'nsd':
        R2_maps = extract_raw_R2_nsd(NSD_SUBS, residual=False, radius=args.radius)
        save_R2_maps(R2_maps, output_dir, "nsd_mean")
    elif args.dataset == 'nsd_residual':
        R2_maps = extract_raw_R2_nsd(NSD_SUBS, residual=True, radius=args.radius)
        save_R2_maps(R2_maps, output_dir, "nsd_residual_mean")


if __name__ == '__main__':
    main()

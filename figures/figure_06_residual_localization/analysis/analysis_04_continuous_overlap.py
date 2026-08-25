"""
分析四：基于连续效应量图的层间重叠分析
回应审稿意见：R1-P5

不在阈值化后的二值图上计算重叠，而在未阈值化的连续效应量图上衡量层间相似性。
其中 synthesized 数据使用 Stouffer group z-score 图；NSD / NSD residual
沿用 analysis_02 导出的连续图。

提供三种互补的连续重叠指标：
  A. 空间模式的 Pearson 相关
  B. 余弦相似度
  C. 加权 Jaccard

用法：
    python analysis_04_continuous_overlap.py --dataset synthesized
    python analysis_04_continuous_overlap.py --dataset nsd
"""
import os
import sys
import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import *


def load_effect_maps(dataset, output_base):
    """
    Load precomputed continuous effect-size maps from analysis_02.

    For synthesized, these are Stouffer group z-score maps exported by analysis_02.
    For nsd / nsd_residual, these are the continuous maps saved by analysis_02.

    Returns:
        effect_maps: dict {layer: (91,109,91) array}
        brain_mask: (91,109,91) boolean array
    """
    R2_dir = os.path.join(output_base, "raw_R2_maps")
    effect_maps = {}

    suffix_map = {
        'synthesized': 'synthesized',
        'nsd': 'nsd_mean',
        'nsd_residual': 'nsd_residual_mean'
    }
    suffix = suffix_map[dataset]

    for layer in range(1, N_LAYERS + 1):
        fpath = os.path.join(R2_dir, f"R2_raw_layer{layer}_{suffix}.nii.gz")
        if os.path.exists(fpath):
            effect_maps[layer] = load_nifti(fpath)
        else:
            print(f"WARNING: {fpath} not found")

    # Create brain mask from non-NaN voxels
    if effect_maps:
        first_layer = list(effect_maps.values())[0]
        brain_mask = ~np.isnan(first_layer) & (first_layer != 0)
    else:
        brain_mask = np.zeros(BRAIN_SHAPE, dtype=bool)

    return effect_maps, brain_mask


def compute_continuous_overlap(effect_maps, brain_mask):
    """
    Compute three continuous overlap metrics between all pairs of layers.

    Returns:
        overlap_pearson: (N_LAYERS, N_LAYERS) Pearson correlation of spatial patterns
        overlap_cosine: (N_LAYERS, N_LAYERS) cosine similarity
        overlap_weighted_jaccard: (N_LAYERS, N_LAYERS) weighted Jaccard
    """
    layers = sorted(effect_maps.keys())
    n = len(layers)

    overlap_pearson = np.full((n, n), np.nan)
    overlap_cosine = np.full((n, n), np.nan)
    overlap_wjaccard = np.full((n, n), np.nan)

    # Extract vectorized effect-size values for each layer (only within brain mask)
    effect_vecs = {}
    for l in layers:
        vec = effect_maps[l][brain_mask].flatten()
        # Pearson uses raw continuous values; cosine/Jaccard use non-negative versions
        vec_clipped = np.clip(vec, 0, None)
        effect_vecs[l] = {'raw': vec, 'clipped': vec_clipped}

    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            vi_raw = effect_vecs[li]['raw']
            vj_raw = effect_vecs[lj]['raw']
            vi_clip = effect_vecs[li]['clipped']
            vj_clip = effect_vecs[lj]['clipped']

            # A. Pearson correlation of spatial patterns
            if np.std(vi_raw) > 0 and np.std(vj_raw) > 0:
                overlap_pearson[i, j], _ = pearsonr(vi_raw, vj_raw)

            # B. Cosine similarity
            norm_i = np.linalg.norm(vi_clip)
            norm_j = np.linalg.norm(vj_clip)
            if norm_i > 0 and norm_j > 0:
                overlap_cosine[i, j] = np.dot(vi_clip, vj_clip) / (norm_i * norm_j)

            # C. Weighted Jaccard (generalized Jaccard for non-negative vectors)
            min_sum = np.sum(np.minimum(vi_clip, vj_clip))
            max_sum = np.sum(np.maximum(vi_clip, vj_clip))
            if max_sum > 0:
                overlap_wjaccard[i, j] = min_sum / max_sum

    return overlap_pearson, overlap_cosine, overlap_wjaccard


def compute_off_diagonal_mean(matrix):
    """Compute mean of off-diagonal elements."""
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return np.nanmean(matrix[mask])


def compute_adjacent_mean(matrix):
    """Compute mean of elements one off the diagonal (adjacent layers)."""
    n = matrix.shape[0]
    vals = []
    for i in range(n - 1):
        vals.append(matrix[i, i + 1])
        vals.append(matrix[i + 1, i])
    return np.nanmean(vals)


def plot_overlap_matrix(matrix, title, output_path, layer_labels=None):
    """Plot a heatmap of the overlap matrix."""
    if layer_labels is None:
        layer_labels = [f"L{i}" for i in range(1, matrix.shape[0] + 1)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-0.2, vmax=1.0, aspect='equal')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(layer_labels)))
    ax.set_yticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, rotation=45, ha='right')
    ax.set_yticklabels(layer_labels)
    ax.set_xlabel('CNN Layer')
    ax.set_ylabel('CNN Layer')
    ax.set_title(title)

    # Add text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.2f}',
                        ha='center', va='center', fontsize=6,
                        color='white' if abs(matrix[i, j]) > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Continuous overlap analysis based on continuous effect-size maps (R1-P5)')
    parser.add_argument('--dataset', type=str, default='synthesized',
                        choices=['synthesized', 'nsd', 'nsd_residual'])
    args = parser.parse_args()

    output_dir = os.path.join(REBUTTAL_OUTPUT, "continuous_overlap")
    mkdir(output_dir)

    # Load continuous effect-size maps
    effect_maps, brain_mask = load_effect_maps(args.dataset, REBUTTAL_OUTPUT)
    if not effect_maps:
        print("No effect-size maps found. Run analysis_02_raw_R2_maps.py first.")
        return

    # Compute continuous overlap
    ov_pearson, ov_cosine, ov_wjaccard = compute_continuous_overlap(effect_maps, brain_mask)

    # Save matrices
    suffix = args.dataset
    np.save(os.path.join(output_dir, f"overlap_pearson_{suffix}.npy"), ov_pearson)
    np.save(os.path.join(output_dir, f"overlap_cosine_{suffix}.npy"), ov_cosine)
    np.save(os.path.join(output_dir, f"overlap_wjaccard_{suffix}.npy"), ov_wjaccard)

    # Plot heatmaps
    plot_overlap_matrix(ov_pearson, f"Spatial Pattern Pearson Correlation ({suffix})",
                        os.path.join(output_dir, f"overlap_pearson_{suffix}.pdf"))
    plot_overlap_matrix(ov_cosine, f"Spatial Pattern Cosine Similarity ({suffix})",
                        os.path.join(output_dir, f"overlap_cosine_{suffix}.pdf"))
    plot_overlap_matrix(ov_wjaccard, f"Spatial Pattern Weighted Jaccard ({suffix})",
                        os.path.join(output_dir, f"overlap_wjaccard_{suffix}.pdf"))

    # Print summary
    print(f"\n=== Continuous Overlap Summary ({suffix}) ===")
    print(f"Pearson: off-diag mean={compute_off_diagonal_mean(ov_pearson):.4f}, "
          f"adjacent mean={compute_adjacent_mean(ov_pearson):.4f}")
    print(f"Cosine:  off-diag mean={compute_off_diagonal_mean(ov_cosine):.4f}, "
          f"adjacent mean={compute_adjacent_mean(ov_cosine):.4f}")
    print(f"Jaccard: off-diag mean={compute_off_diagonal_mean(ov_wjaccard):.4f}, "
          f"adjacent mean={compute_adjacent_mean(ov_wjaccard):.4f}")


if __name__ == '__main__':
    main()

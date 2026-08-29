# Usage:
#   python3 -u rebuttal/analyses/analysis_07_geodesic_distance.py
#   sbatch rebuttal/analyses/job_07.sh
# Run from localize_paper root.
# Only requires: Wang atlas (wang_atlas_mni2mm.nii.gz) + main pipeline outputs.
# Does NOT require analysis_02 or analysis_05 to have been run first.

"""
分析七：最佳层编号与离V1距离的空间梯度
回应审稿意见：R1-P5（Jeff's comment on geodesic distance from V1）

检验假设：沿视觉层级，随体素与V1距离增大，CNN最佳层编号是否系统性增大？

本脚本自给自足——直接从主流程逐被试 z-scored R² NIfTI 文件计算
Stouffer 组z-score 和最佳层图，不依赖 analysis_02/05 的缓存输出。

两个互补的"最佳层"度量：
  1. 离散最佳层 (discrete):  argmax across layers of Stouffer group z-score
  2. 连续加权层 (continuous): Σ(z⁺ × layer_i) / Σ(z⁺)
     z⁺ = max(group_z, 0)，减少单层噪声的影响

距离度量：
  体积欧氏距离（mm），从 Wang atlas 双侧V1（V1v + V1d）最近体素算起

Output: figures/components/geodesic_distance_analysis.pdf
"""

import sys
print("[PY] script entered", flush=True)
import os
print("[PY] os imported", flush=True)
import time
T0 = time.time()
def _ts(msg):
    print(f"[PY +{time.time()-T0:6.1f}s] {msg}", flush=True)

_ts("about to import numpy")
import numpy as np
_ts(f"numpy {np.__version__}")
import nibabel as nib
_ts(f"nibabel {nib.__version__}")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
_ts("matplotlib imported")
from scipy.stats import spearmanr
from scipy.ndimage import distance_transform_edt
_ts("scipy imported")

sys.path.insert(0, os.path.dirname(__file__))
from config import *
_ts("config imported")

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ATLAS_PATH    = os.path.join(SCRIPT_DIR, "wang_atlas_mni2mm.nii.gz")
PAPER_FIG_DIR = os.path.join(SCRIPT_DIR, "..", "..", "figures", "components")
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# Wang atlas bilateral V1: V1v-L(1), V1d-L(2), V1v-R(26), V1d-R(27)
V1_LABELS = [1, 2, 26, 27]

# ── helpers ────────────────────────────────────────────────────────────────────

def load_wang_v1_mask(atlas_path):
    atlas = nib.load(atlas_path).get_fdata()
    v1 = np.zeros(atlas.shape, dtype=bool)
    for lab in V1_LABELS:
        v1 |= (atlas == lab)
    print(f"V1 mask: {v1.sum()} voxels (bilateral V1v+V1d, labels {V1_LABELS})")
    return v1


def compute_dist_from_v1(v1_mask, voxel_size_mm=2.0):
    """Euclidean distance (mm) from each voxel to the nearest V1 voxel."""
    dist_vox = distance_transform_edt(~v1_mask)
    return dist_vox * voxel_size_mm


def compute_group_z_maps(subjects, radius=SEARCHLIGHT_RADIUS):
    """
    Load per-subject z-scored R² NIfTI files (from main pipeline) and compute
    Stouffer group z-score per layer: group_z = Σzᵢ / √N_valid.

    Returns:
        group_z: dict {layer: (91,109,91) Stouffer z-score map}
        brain_mask: (91,109,91) bool — voxels with data in ALL layers for >50% subjects
    """
    _ts(f"Loading per-subject z-scored R² maps: {len(subjects)} subs × {N_LAYERS} layers")
    # Probe first path to validate
    probe = get_zscored_path(subjects[0], 1, method='reg', radius=radius)
    _ts(f"probe path: {probe}")
    _ts(f"probe exists: {os.path.exists(probe)}")

    # shape: (N_layers, N_subjects, 91, 109, 91)
    z_stack = np.full((N_LAYERS, len(subjects)) + BRAIN_SHAPE, np.nan,
                       dtype=np.float32)
    _ts(f"z_stack allocated: shape={z_stack.shape} ({z_stack.nbytes/1e9:.2f} GB)")

    total = N_LAYERS * len(subjects)
    done = 0
    for l_idx, layer in enumerate(range(1, N_LAYERS + 1)):
        n_loaded = 0
        for s_idx, sub in enumerate(subjects):
            path = get_zscored_path(sub, layer, method='reg', radius=radius)
            if os.path.exists(path):
                z_stack[l_idx, s_idx] = nib.load(path).get_fdata().astype(np.float32)
                n_loaded += 1
            done += 1
            if done % 30 == 0:
                _ts(f"  load progress: {done}/{total}  (layer {layer}, sub {sub})")
        _ts(f"  Layer {layer}: {n_loaded}/{len(subjects)} subjects loaded")

    # Stouffer group z per layer
    group_z = {}
    for l_idx, layer in enumerate(range(1, N_LAYERS + 1)):
        layer_data = z_stack[l_idx]               # (N_sub, 91, 109, 91)
        valid      = np.isfinite(layer_data)       # (N_sub, 91, 109, 91)
        n_valid    = valid.sum(axis=0)             # (91, 109, 91)
        sum_z      = np.where(valid, layer_data, 0).sum(axis=0)
        gz         = np.where(n_valid > 0, sum_z / np.sqrt(n_valid), np.nan)
        group_z[layer] = gz

    # Brain mask: at least one layer has data
    has_data = np.zeros(BRAIN_SHAPE, dtype=bool)
    for gz in group_z.values():
        has_data |= np.isfinite(gz)

    print(f"Brain voxels with any data: {has_data.sum()}")
    return group_z, has_data


def compute_best_layer_maps(group_z):
    """
    From group z-score dict, compute:
      1. discrete: argmax across layers (1-indexed)
      2. continuous: z⁺-weighted mean layer index
    """
    layers = sorted(group_z.keys())
    gz_stack = np.stack([group_z[l] for l in layers], axis=-1)  # (91,109,91,12)

    # Discrete: argmax (1-indexed)
    all_nan = np.all(~np.isfinite(gz_stack), axis=-1)
    gz_filled = np.where(np.isfinite(gz_stack), gz_stack, -np.inf)
    disc_bl   = (np.argmax(gz_filled, axis=-1) + 1).astype(float)
    disc_bl[all_nan] = np.nan

    # Continuous: z⁺-weighted mean
    z_pos      = np.maximum(gz_stack, 0)        # (91,109,91,12)
    weight_sum = z_pos.sum(axis=-1)
    layer_idx  = np.arange(1, N_LAYERS + 1, dtype=float)
    num        = np.sum(z_pos * layer_idx, axis=-1)
    cont_bl    = np.where(weight_sum > 0, num / weight_sum, np.nan)

    return disc_bl, cont_bl


def make_z_threshold_mask(group_z, z_threshold=1.65):
    """
    Significance proxy: voxels where max(group_z across layers) > z_threshold.
    z=1.65 ≈ one-sided p<0.05 uncorrected; z=2.0 is more conservative.
    Used when TFCE masks are unavailable. Described in paper as:
      "voxels where at least one layer reached group z > {z_threshold}"
    """
    gz_stack = np.stack([group_z[l] for l in sorted(group_z.keys())], axis=-1)
    max_z    = np.nanmax(gz_stack, axis=-1)
    mask     = max_z > z_threshold
    print(f"  z-threshold mask (max group_z > {z_threshold}): {mask.sum()} voxels")
    return mask


def load_tfce_any_layer_mask(group_z=None, radius=SEARCHLIGHT_RADIUS,
                              z_threshold=1.65, strict=True):
    """Union of TFCE corrp > 0.95 voxels across all 12 layers.

    strict=True (default): require all 12 TFCE masks. If any is missing, raise
    instead of silently switching the significance regime. A silent fall back to
    the group-z threshold changes the voxel set (n went 77,647 -> 191,114) and
    the reported Spearman r (-0.22 -> -0.09); that swap previously went unnoticed.
    strict=False: knowingly opt into the group-z threshold fallback
    (max z > z_threshold for any layer) when masks are missing.
    """
    any_sig = np.zeros(BRAIN_SHAPE, dtype=bool)
    n_found = 0
    missing = []
    for layer in range(1, N_LAYERS + 1):
        p = get_tfce_mask_path(layer, radius=radius)
        if os.path.exists(p):
            tfce = nib.load(p).get_fdata()
            any_sig |= (tfce > 0.95)
            n_found += 1
        else:
            missing.append(p)

    if n_found < N_LAYERS:
        msg = (f"TFCE masks incomplete: found {n_found}/{N_LAYERS}. "
               f"First missing: {missing[0] if missing else '(none)'}")
        if strict:
            raise FileNotFoundError(
                msg + "\nRefusing to silently fall back to a group-z threshold "
                "mask: that changes the voxel set (n) and the reported r. Restore "
                "or repoint the TFCE masks (config.get_tfce_mask_path already "
                "tries both localize_bkp and localize), or pass strict=False to "
                f"knowingly opt into the max-z>{z_threshold} fallback."
            )
        print(f"  WARNING (strict=False): {msg}\n"
              f"  falling back to group-z threshold (max z > {z_threshold}).")
        return make_z_threshold_mask(group_z, z_threshold)

    print(f"TFCE masks loaded for {n_found}/{N_LAYERS} layers. "
          f"Any-layer significant: {any_sig.sum()} voxels")
    return any_sig


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Analysis 07: Best-layer gradient vs. distance from V1")
    print("=" * 60)

    assert os.path.exists(ATLAS_PATH), (
        f"Wang atlas not found: {ATLAS_PATH}\n"
        f"Run first:  python3 rebuttal/analyses/download_wang_atlas.py"
    )

    # Step 1: V1 mask + distance map
    print("\n[1] Computing distance from V1...")
    v1_mask = load_wang_v1_mask(ATLAS_PATH)
    dist_mm = compute_dist_from_v1(v1_mask)

    out_dir = os.path.join(REBUTTAL_OUTPUT, "best_layer_reliability")
    mkdir(out_dir)
    save_nifti(dist_mm, os.path.join(out_dir, "dist_from_v1_mm.nii.gz"))

    # Step 2: group z-score maps (self-contained, from main pipeline)
    print("\n[2] Computing Stouffer group z-score maps (30 subjects × 12 layers)...")
    group_z, brain_mask = compute_group_z_maps(SYNTHESIZED_SUBS)

    # Step 3: best-layer maps
    print("\n[3] Computing best-layer maps...")
    disc_bl, cont_bl = compute_best_layer_maps(group_z)

    # Step 4: significance mask — TFCE required (strict; no silent fallback)
    print("\n[4] Loading TFCE significance mask (strict: all 12 required)...")
    sig_mask = load_tfce_any_layer_mask(group_z=group_z, strict=True)

    # Step 5: combined mask
    valid_mask = sig_mask & np.isfinite(disc_bl) & np.isfinite(cont_bl)
    n_valid = valid_mask.sum()
    print(f"\n[5] Valid voxels (TFCE sig + finite best layer): {n_valid:,}")

    if n_valid < 100:
        print("ERROR: Too few valid voxels. Check TFCE mask paths and pipeline outputs.")
        sys.exit(1)

    dist_vals = dist_mm[valid_mask]
    disc_vals = disc_bl[valid_mask]
    cont_vals = cont_bl[valid_mask]

    # Step 6: Spearman correlations
    rho_disc, p_disc = spearmanr(dist_vals, disc_vals)
    rho_cont, p_cont = spearmanr(dist_vals, cont_vals)
    print(f"\n  Discrete best layer: Spearman r={rho_disc:.3f}, p={p_disc:.2e}")
    print(f"  Continuous best layer: Spearman r={rho_cont:.3f}, p={p_cont:.2e}")

    # Step 7: 10 mm distance bins
    bin_edges  = np.arange(0, dist_vals.max() + 10, 10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    disc_m, disc_e, cont_m, cont_e, ns = [], [], [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        idx = (dist_vals >= lo) & (dist_vals < hi)
        n = idx.sum()
        ns.append(n)
        for vals, m_list, e_list in [(disc_vals, disc_m, disc_e),
                                      (cont_vals, cont_m, cont_e)]:
            if n >= 5:
                m_list.append(np.mean(vals[idx]))
                e_list.append(np.std(vals[idx], ddof=1) / np.sqrt(n))
            else:
                m_list.append(np.nan)
                e_list.append(np.nan)

    disc_m = np.array(disc_m);  disc_e = np.array(disc_e)
    cont_m = np.array(cont_m);  cont_e = np.array(cont_e)
    ns     = np.array(ns)

    # Step 8: single clean panel — discrete binned mean best-layer index vs distance from V1.
    # （原为 4-panel A/B/C/D + 散点云，读者看不懂；这里只保留最清楚的 Panel B，
    #   并修正原标题写反的方向 bug: index 是 DECREASES 而非 increases。）
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ok = np.isfinite(disc_m) & (ns >= 5)
    ax.fill_between(bin_centers[ok], disc_m[ok] - disc_e[ok], disc_m[ok] + disc_e[ok],
                    alpha=0.25, color='steelblue', linewidth=0)
    ax.plot(bin_centers[ok], disc_m[ok], 'o-', color='steelblue', lw=2.2, ms=6)
    ax.set_xlabel('Distance from V1 (mm)', fontsize=12)
    ax.set_ylabel('Mean best-layer index (argmax) ± SEM', fontsize=12)
    # Title (discrete argmax, 10 mm bins, Spearman r, N) and gridlines removed
    # per reviewer feedback; the stats now live in the caption instead.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # NOTE: no in-figure "how to read" box — the interpretation lives in the
    # markdown caption (Fig R1-5) instead, per reviewer/author preference.
    fig.tight_layout()

    out_pdf = os.path.join(PAPER_FIG_DIR, "geodesic_distance_analysis.pdf")
    fig.savefig(out_pdf, bbox_inches='tight', dpi=150)

    # 同时导出 PNG 到仓库 git 追踪路径 md_png/（用 __file__ 定位，与 cwd 无关）
    _png = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "figures", "previews", "geodesic_distance_analysis.png"))
    try:
        fig.savefig(_png, bbox_inches='tight', dpi=150)
        print(f"Saved PNG (git-tracked): {_png}")
    except Exception as e:
        print(f"[warn] could not write {_png}: {e}")
    plt.close()

    print(f"\n=== Summary ===")
    print(f"Valid voxels: {n_valid:,}")
    print(f"Distance range: {dist_vals.min():.0f}–{dist_vals.max():.0f} mm")
    print(f"Discrete  best layer: Spearman r={rho_disc:.3f}, p={p_disc:.2e}")
    print(f"Continuous best layer: Spearman r={rho_cont:.3f}, p={p_cont:.2e}")
    print(f"Saved: {out_pdf}")
    print("done")


if __name__ == '__main__':
    main()

"""
分析 09b：NSD-Synthetic 真实 CNN 对照
回应审稿意见：R1-意见6（随机纹理对照分析）

使用 NSD-Synthetic 数据集的真实随机纹理图像，通过 InceptionV1 前向传播计算
CNN 特征 RSM，与优化合成图像的 RSM 比较，证明优化图像的科学意义。

三组比较：
  A: 优化合成图像     — 直接加载 features.npy（已预计算 RSM）
  B: NSD-Synthetic 噪声图（idx 0–15）— 真实 InceptionV1 前向传播
  C: NSD-Synthetic 随机子集 × 100   — Bootstrap 控制

运行环境：tf1 conda 环境（TF 1.14，与论文一致）

并行模式（推荐）：
  --mode groupAB          # 计算 Group A + B（快速，< 1 min）
  --mode bootstrap        # Bootstrap 单 chunk（配合 SLURM array 并行）
      --chunk_id INT      # 当前 chunk 编号（0-indexed）
      --n_chunks INT      # 总 chunk 数（默认 10）
  --mode merge            # 合并所有 bootstrap chunk，生成最终图

单机顺序模式：
  --mode all              # 顺序运行全部（默认）

SLURM 提交见 job_09b_submit.sh
"""
import os
import sys
import argparse
import shutil
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from config import *

# ============================================================
# Paths
# ============================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))

HDF5_PATH = os.path.join(_THIS_DIR, 'nsd_synthetic', 'nsdsynthetic_stimuli.hdf5')
MODEL_PATH = os.path.join(_PROJECT_ROOT, 'utils', 'tensorflow_inception_graph.pb')
FEATURES_PATH = get_features_path()
OUTPUT_DIR = os.path.join(REBUTTAL_OUTPUT, 'nsd_synthetic_comparison')
TRACKED_PLOT_CACHE = os.path.join(
    _THIS_DIR, 'nsd_synthetic_data', 'nsd_synthetic_plot_cache.npz'
)

# ============================================================
# Constants
# ============================================================
N_IMAGES_GROUP_AB = 16    # match our synthesized image count
N_LAYERS = 12
N_BOOTSTRAP = 100
IMAGENET_MEAN = 117.0
IMG_SIZE = 539            # InceptionV1 input size used in the paper


# ============================================================
# InceptionV1 model loading
# ============================================================
def load_inception_model():
    """
    Load InceptionV1 (.pb) and return (graph, sess, t_input, incept_layers).
    Replicates the exact setup in ImageSynthesis.py / baseline.py.
    """
    import tensorflow as tf

    graph = tf.Graph()
    with graph.as_default(), graph.device('/cpu:0'):
        with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        t_input = tf.placeholder(np.float32, name='input')
        t_preprocessed = tf.expand_dims(t_input - IMAGENET_MEAN, 0)
        tf.import_graph_def(graph_def, {'input': t_preprocessed})

    # Layer names — identical logic to ImageSynthesis.py
    all_ops = [op.name for op in graph.get_operations()
               if op.type == 'Conv2D' and 'import/' in op.name]
    layers1 = [l.split('/')[1] for l in all_ops
               if l.split('/')[1].split('2')[0] == 'conv']
    layers2 = [l.split('/')[1] for l in all_ops
               if 'pool_reduce' in l]
    incept_layers = layers1 + layers2  # 12 layers, same order as ImageSynthesis.py

    assert len(incept_layers) == N_LAYERS, (
        f"Expected {N_LAYERS} InceptionV1 layers, got {len(incept_layers)}: {incept_layers}"
    )

    sess = tf.Session(graph=graph)
    return graph, sess, t_input, incept_layers


# ============================================================
# Image preprocessing
# ============================================================
def preprocess_image(img_arr):
    """
    HDF5 image → (539, 539, 3) float, matching baseline.py's pre-processing.

    Input:  uint8 array of arbitrary H×W (and 1 or 3 channels)
    Output: (539, 539, 3) float64
    """
    # Ensure 3-channel
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    elif img_arr.shape[2] == 1:
        img_arr = np.repeat(img_arr, 3, axis=2)

    pil_img = Image.fromarray(img_arr.astype(np.uint8))
    w, h = pil_img.size  # PIL uses (width, height)

    # Center-crop to square
    if w > h:
        left = (w - h) // 2
        pil_img = pil_img.crop((left, 0, left + h, h))
    elif h > w:
        top = (h - w) // 2
        pil_img = pil_img.crop((0, top, w, top + w))

    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(pil_img).astype(float)


# ============================================================
# Forward pass + RSM
# ============================================================
def run_forward_pass(graph, sess, t_input, incept_layers, images):
    """
    Run a list of images through InceptionV1.

    Args:
        images: list of (539, 539, 3) float arrays

    Returns:
        rsms: (12, N, N) — Pearson RSM per layer
    """
    n = len(images)
    tensors = [graph.get_tensor_by_name(f'import/{l}:0') for l in incept_layers]

    # Collect flattened activations: acts[layer_idx] = list of flat vectors
    acts = [[] for _ in range(N_LAYERS)]

    for img in images:
        layer_acts = sess.run(tensors, {t_input: img})  # list of (H, W, C) arrays
        for li, a in enumerate(layer_acts):
            acts[li].append(a.flatten())

    # Compute RSM per layer
    rsms = np.zeros((N_LAYERS, n, n))
    for li in range(N_LAYERS):
        mat = np.array(acts[li])  # (n, D)
        rsms[li] = np.corrcoef(mat)

    return rsms


# ============================================================
# Inter-layer RSM correlation
# ============================================================
def inter_layer_corr(rsms):
    """
    Mean off-diagonal inter-layer RSM correlation.

    Args:
        rsms: (12, N, N)

    Returns:
        corr_matrix: (12, 12)
        mean_off_diag: scalar
    """
    n_layers, n_img, _ = rsms.shape
    triu = np.triu_indices(n_img, k=1)
    vecs = np.array([rsms[l][triu] for l in range(n_layers)])  # (12, K)
    corr_matrix = np.corrcoef(vecs)                             # (12, 12)
    mask = ~np.eye(n_layers, dtype=bool)
    return corr_matrix, float(np.mean(corr_matrix[mask]))


# ============================================================
# Group A: optimized synthesized images (pre-computed)
# ============================================================
def group_A():
    print("=== Group A: Optimized Synthesized Images (features.npy) ===")
    rsms = np.load(FEATURES_PATH)  # (12, 16, 16)
    assert rsms.shape == (N_LAYERS, N_IMAGES_GROUP_AB, N_IMAGES_GROUP_AB), rsms.shape
    corr_mat, mean_corr = inter_layer_corr(rsms)
    print(f"  Mean inter-layer RSM correlation: {mean_corr:.4f}")
    return rsms, corr_mat, mean_corr


# ============================================================
# Group B: NSD-Synthetic noise images (idx 0–15)
# ============================================================
def group_B(graph, sess, t_input, incept_layers):
    print("=== Group B: NSD-Synthetic Noise Images (idx 0–15) ===")
    import h5py
    with h5py.File(HDF5_PATH, 'r') as f:
        raw = f['imgBrick'][:N_IMAGES_GROUP_AB]  # (16, H, W) or (16, H, W, 3)

    images = [preprocess_image(raw[i]) for i in range(N_IMAGES_GROUP_AB)]
    print(f"  Loaded {len(images)} images, running forward pass...")

    rsms = run_forward_pass(graph, sess, t_input, incept_layers, images)
    corr_mat, mean_corr = inter_layer_corr(rsms)
    print(f"  Mean inter-layer RSM correlation: {mean_corr:.4f}")
    return rsms, corr_mat, mean_corr


# ============================================================
# Group C: Bootstrap control (100 random subsets of 16)
# ============================================================
def group_C(graph, sess, t_input, incept_layers,
            n_bootstrap=N_BOOTSTRAP, chunk_id=0, n_chunks=1):
    """
    Bootstrap control. When n_chunks > 1, only processes the slice
    [chunk_id*chunk_size, (chunk_id+1)*chunk_size) of the 100 iterations.
    Random seeds are fixed globally so results are reproducible regardless
    of chunking scheme.
    """
    # Determine which iterations this chunk owns
    chunk_size = (n_bootstrap + n_chunks - 1) // n_chunks  # ceiling division
    iter_start = chunk_id * chunk_size
    iter_end   = min(iter_start + chunk_size, n_bootstrap)
    print(f"=== Group C Bootstrap: chunk {chunk_id}/{n_chunks}, "
          f"iterations {iter_start}–{iter_end-1} ===")

    print("  Pre-loading all NSD-Synthetic images from HDF5...")
    import h5py
    with h5py.File(HDF5_PATH, 'r') as f:
        raw_all = f['imgBrick'][:]
    n_total = raw_all.shape[0]
    print(f"  Total images: {n_total}")
    all_images = [preprocess_image(raw_all[i]) for i in range(n_total)]

    # Use a single global RNG seeded at 42; advance it to the right position
    # by generating all indices up to iter_start first (ensures reproducibility)
    rng = np.random.RandomState(42)
    for _ in range(iter_start):
        rng.choice(n_total, N_IMAGES_GROUP_AB, replace=False)  # consume RNG state

    mean_corrs = []
    for b in range(iter_start, iter_end):
        if (b - iter_start) % 5 == 0:
            print(f"  Iteration {b}/{n_bootstrap}...")
        indices = rng.choice(n_total, N_IMAGES_GROUP_AB, replace=False)
        images  = [all_images[i] for i in indices]
        rsms    = run_forward_pass(graph, sess, t_input, incept_layers, images)
        _, mean_corr = inter_layer_corr(rsms)
        mean_corrs.append(mean_corr)

    bootstrap_corrs = np.array(mean_corrs)
    print(f"  Chunk done. mean r = {np.mean(bootstrap_corrs):.4f}")
    return bootstrap_corrs


# ============================================================
# Plotting
# ============================================================
def validate_plot_data(corr_mat_A, mean_A, corr_mat_B, mean_B, bootstrap_corrs):
    matrices = (("optimized", corr_mat_A, mean_A),
                ("NSD-Synthetic", corr_mat_B, mean_B))
    for name, matrix, stored_mean in matrices:
        if matrix.shape != (N_LAYERS, N_LAYERS) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} correlation matrix must be finite 12x12")
        if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0):
            raise ValueError(f"{name} correlation matrix is not symmetric")
        if not np.allclose(np.diag(matrix), 1.0, atol=1e-10, rtol=0):
            raise ValueError(f"{name} correlation matrix diagonal is not one")
        if np.min(matrix) < -1.000001 or np.max(matrix) > 1.000001:
            raise ValueError(
                f"{name} correlation values fall outside [-1, 1]: "
                f"[{np.min(matrix)}, {np.max(matrix)}]"
            )
        calculated_mean = float(np.mean(matrix[~np.eye(N_LAYERS, dtype=bool)]))
        if not np.isclose(calculated_mean, stored_mean, atol=1e-10, rtol=0):
            raise ValueError(
                f"{name} stored mean {stored_mean} does not equal the matrix "
                f"off-diagonal mean {calculated_mean}"
            )

    bootstrap_corrs = np.asarray(bootstrap_corrs)
    if bootstrap_corrs.shape != (N_BOOTSTRAP,):
        raise ValueError(
            f"bootstrap correlations must have shape ({N_BOOTSTRAP},), "
            f"got {bootstrap_corrs.shape}"
        )
    if not np.all(np.isfinite(bootstrap_corrs)):
        raise ValueError("bootstrap correlations contain non-finite values")
    if np.min(bootstrap_corrs) < -1.000001 or np.max(bootstrap_corrs) > 1.000001:
        raise ValueError("bootstrap correlations fall outside [-1, 1]")


def plot_results(corr_mat_A, mean_A, corr_mat_B, mean_B,
                 bootstrap_corrs, primary_output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 5))
    panel_letter_size = 22
    colorbar_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    colorbar_ticklabels = ['−1.0', '−0.5', '0.0', '0.5', '1.0']

    # --- Panel 1: Bar chart ---
    ax1 = fig.add_subplot(1, 3, 1)

    labels = ['Optimized\nSynthesized', 'NSD Noise\n(idx 0–15)', 'Bootstrap\n(100 × random 16)']
    means  = [mean_A, mean_B, float(np.mean(bootstrap_corrs))]
    colors = ['steelblue', 'darkorange', 'dimgray']

    bars = ax1.bar(labels, means, color=colors, alpha=0.85, width=0.5, edgecolor='black', linewidth=0.7)
    ax1.errorbar(
        x=2, y=means[2], yerr=float(np.std(bootstrap_corrs)),
        fmt='none', color='black', capsize=6, linewidth=1.5
    )

    # Scatter bootstrap points
    rng_plot = np.random.RandomState(0)
    jitter = rng_plot.uniform(-0.18, 0.18, len(bootstrap_corrs))
    ax1.scatter(
        np.full(len(bootstrap_corrs), 2) + jitter, bootstrap_corrs,
        color='dimgray', alpha=0.25, s=12, zorder=3
    )

    # Significance bracket A vs B
    y_br = max(means) + 0.13
    x0, x1 = 0, 1
    ax1.plot([x0, x0, x1, x1], [y_br - 0.02, y_br, y_br, y_br - 0.02],
             color='black', linewidth=1.2)
    ax1.text((x0 + x1) / 2, y_br + 0.01, '***', ha='center', fontsize=14)

    ax1.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
    ax1.set_ylabel('Mean Inter-Layer RSM Correlation', fontsize=11)
    # Title ("InceptionV1 Layer Distinguishability, 16 images x 12 layers")
    # removed per reviewer feedback; moved to the caption.
    ax1.set_ylim(-0.25, 1.1)
    ax1.tick_params(axis='x', labelsize=9)

    validate_plot_data(corr_mat_A, mean_A, corr_mat_B, mean_B, bootstrap_corrs)

    # Pearson correlations use their complete, symmetric domain. This matches
    # the correlation matrices in Figure 1 and avoids visually exaggerating the
    # negative values by stretching only [-0.3, 0] across half the colorbar.
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    # --- Panel 2: Heatmap — Optimized ---
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(corr_mat_A, cmap='RdBu_r', norm=norm)
    ax2.set_xlabel('InceptionV1 Layer', fontsize=10)
    ax2.set_ylabel('InceptionV1 Layer', fontsize=10)
    cbar2 = plt.colorbar(
        im2, ax=ax2, fraction=0.046, pad=0.04, label='RSM correlation'
    )
    cbar2.set_ticks(colorbar_ticks)
    cbar2.set_ticklabels(colorbar_ticklabels)

    # --- Panel 3: Heatmap — NSD Noise ---
    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(corr_mat_B, cmap='RdBu_r', norm=norm)
    ax3.set_xlabel('InceptionV1 Layer', fontsize=10)
    ax3.set_ylabel('InceptionV1 Layer', fontsize=10)
    cbar3 = plt.colorbar(
        im3, ax=ax3, fraction=0.046, pad=0.04, label='RSM correlation'
    )
    cbar3.set_ticks(colorbar_ticks)
    cbar3.set_ticklabels(colorbar_ticklabels)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    for label, x in zip('ABC', (0.005, 0.338, 0.671)):
        fig.text(x, 0.955, label, fontsize=panel_letter_size,
                 fontweight='normal', va='top', ha='left')

    repo_pdf = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "figures", "components",
        "nsd_synthetic_comparison_labeled.pdf"))
    staged_pdf = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "figures", "final",
        "Sfig4_NSD_synthetic_layer_distinguishability.pdf"))
    out_path = primary_output or os.path.join(
        OUTPUT_DIR, 'nsd_synthetic_comparison.pdf'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {out_path}")

    os.makedirs(os.path.dirname(repo_pdf), exist_ok=True)
    os.makedirs(os.path.dirname(staged_pdf), exist_ok=True)
    if os.path.abspath(out_path) != os.path.abspath(repo_pdf):
        shutil.copyfile(out_path, repo_pdf)
    if os.path.abspath(out_path) != os.path.abspath(staged_pdf):
        shutil.copyfile(out_path, staged_pdf)
    print(f"Saved (repo, git-tracked): {repo_pdf}")
    print(f"Saved (Overleaf upload): {staged_pdf}")

    # 整张图（不用裁）直接写进仓库 git 跟踪路径，不用再手动转/拷。
    repo_png = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "figures", "previews", "nsd_synthetic_comparison.png"))
    os.makedirs(os.path.dirname(repo_png), exist_ok=True)
    plt.savefig(repo_png, dpi=150, bbox_inches='tight')
    print(f"Saved (repo, git-tracked): {repo_png}")

    plt.close()


def run_tracked_plot_cache():
    """Render S11 locally from the compact, git-tracked plotting cache."""
    if not os.path.isfile(TRACKED_PLOT_CACHE):
        raise FileNotFoundError(
            "Tracked S11 plotting cache is missing. Run "
            "track_nsd_synthetic_plot_cache.sh on Milgram, commit/push the NPZ, "
            f"then pull it locally: {TRACKED_PLOT_CACHE}"
        )
    with np.load(TRACKED_PLOT_CACHE, allow_pickle=False) as cache:
        expected_keys = {
            'corr_mat_A', 'corr_mat_B', 'mean_A', 'mean_B', 'bootstrap_corrs'
        }
        if set(cache.files) != expected_keys:
            raise RuntimeError(
                f"Tracked S11 cache keys are {sorted(cache.files)}; "
                f"expected {sorted(expected_keys)}"
            )
        corr_mat_A = cache['corr_mat_A']
        corr_mat_B = cache['corr_mat_B']
        mean_A = float(cache['mean_A'])
        mean_B = float(cache['mean_B'])
        bootstrap_corrs = cache['bootstrap_corrs']

    repo_pdf = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "figures", "components",
        "nsd_synthetic_comparison_labeled.pdf"))
    plot_results(
        corr_mat_A, mean_A, corr_mat_B, mean_B, bootstrap_corrs,
        primary_output=repo_pdf,
    )


# ============================================================
# Merge helper
# ============================================================
def run_merge():
    """Merge all bootstrap chunks and generate final figure."""
    print("=== Merge: collecting bootstrap chunks ===")

    # Load only the compact arrays used by the plot. The full RSM arrays are not
    # needed for rendering and therefore are not required here.
    corr_mat_A = np.load(os.path.join(OUTPUT_DIR, 'corr_mat_A.npy'))
    mean_A     = float(np.load(os.path.join(OUTPUT_DIR, 'mean_A.npy')))
    corr_mat_B = np.load(os.path.join(OUTPUT_DIR, 'corr_mat_B.npy'))
    mean_B     = float(np.load(os.path.join(OUTPUT_DIR, 'mean_B.npy')))

    # Collect chunk files
    chunk_files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith('bootstrap_chunk_') and f.endswith('.npy')
    ])
    assert len(chunk_files) > 0, f"No bootstrap chunk files found in {OUTPUT_DIR}"
    print(f"  Found {len(chunk_files)} chunk file(s): {chunk_files}")

    all_corrs = np.concatenate([
        np.load(os.path.join(OUTPUT_DIR, f)) for f in chunk_files
    ])
    print(f"  Total bootstrap iterations merged: {len(all_corrs)}")
    np.save(os.path.join(OUTPUT_DIR, 'bootstrap_corrs.npy'), all_corrs)

    # CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, 'results_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'description', 'mean_inter_layer_corr', 'std'])
        w.writerow(['A', 'Optimized synthesized images', f'{mean_A:.6f}', '0.000000'])
        w.writerow(['B', 'NSD-Synthetic noise idx 0-15', f'{mean_B:.6f}', '0.000000'])
        w.writerow(['C_mean', 'Bootstrap mean (100 x random 16)',
                    f'{np.mean(all_corrs):.6f}', f'{np.std(all_corrs):.6f}'])
    print(f"  Saved CSV: {csv_path}")

    plot_results(corr_mat_A, mean_A, corr_mat_B, mean_B, all_corrs)

    p_val = float(np.mean(all_corrs <= mean_A))
    print("\n" + "=" * 55)
    print("RESULTS SUMMARY")
    print("=" * 55)
    print(f"  Group A (Optimized):  mean r = {mean_A:.4f}")
    print(f"  Group B (NSD Noise):  mean r = {mean_B:.4f}")
    print(f"  Group C (Bootstrap):  mean r = {np.mean(all_corrs):.4f} "
          f"± {np.std(all_corrs):.4f}  (n={len(all_corrs)})")
    print(f"  Difference B − A:  {mean_B - mean_A:.4f}")
    print(f"  p (bootstrap ≤ optimized): {p_val:.4f}")
    print("=" * 55)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        choices=['all', 'groupAB', 'bootstrap', 'merge',
                                 'plot-cache'],
                        default='all')
    parser.add_argument('--chunk_id',  type=int, default=0,
                        help='0-indexed chunk (bootstrap mode)')
    parser.add_argument('--n_chunks',  type=int, default=10,
                        help='Total number of chunks (bootstrap mode)')
    parser.add_argument('--n_bootstrap', type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    if args.mode == 'plot-cache':
        run_tracked_plot_cache()
        return

    mkdir(OUTPUT_DIR)

    # ── merge: no model needed ────────────────────────────────
    if args.mode == 'merge':
        run_merge()
        return

    # ── all other modes need file checks ─────────────────────
    for path, name in [(HDF5_PATH, 'HDF5'),
                       (MODEL_PATH, 'InceptionV1 .pb'),
                       (FEATURES_PATH, 'features.npy')]:
        assert os.path.exists(path), f"Required file not found: {path} ({name})"

    # ── groupAB ───────────────────────────────────────────────
    if args.mode in ('all', 'groupAB'):
        rsms_A, corr_mat_A, mean_A = group_A()
        np.save(os.path.join(OUTPUT_DIR, 'rsms_A.npy'),     rsms_A)
        np.save(os.path.join(OUTPUT_DIR, 'corr_mat_A.npy'), corr_mat_A)
        np.save(os.path.join(OUTPUT_DIR, 'mean_A.npy'),     np.array(mean_A))

        print("\nLoading InceptionV1 model...")
        graph, sess, t_input, incept_layers = load_inception_model()
        print(f"  Layers ({len(incept_layers)}): {incept_layers}")

        print()
        rsms_B, corr_mat_B, mean_B = group_B(graph, sess, t_input, incept_layers)
        np.save(os.path.join(OUTPUT_DIR, 'rsms_B.npy'),     rsms_B)
        np.save(os.path.join(OUTPUT_DIR, 'corr_mat_B.npy'), corr_mat_B)
        np.save(os.path.join(OUTPUT_DIR, 'mean_B.npy'),     np.array(mean_B))
        sess.close()

        if args.mode == 'groupAB':
            print("\ngroupAB done.")
            return

    # ── bootstrap chunk ───────────────────────────────────────
    if args.mode in ('all', 'bootstrap'):
        if args.mode == 'all':
            chunk_id, n_chunks = 0, 1          # single chunk = all 100
        else:
            chunk_id, n_chunks = args.chunk_id, args.n_chunks

        print("\nLoading InceptionV1 model for bootstrap...")
        graph, sess, t_input, incept_layers = load_inception_model()

        partial = group_C(graph, sess, t_input, incept_layers,
                          n_bootstrap=args.n_bootstrap,
                          chunk_id=chunk_id, n_chunks=n_chunks)
        sess.close()

        out_file = os.path.join(OUTPUT_DIR,
                                f'bootstrap_chunk_{chunk_id:02d}.npy')
        np.save(out_file, partial)
        print(f"  Saved: {out_file}")

        if args.mode == 'all':
            # sequential: immediately merge the single chunk
            run_merge()


if __name__ == '__main__':
    main()

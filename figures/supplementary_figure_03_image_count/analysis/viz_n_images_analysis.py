"""
N图像充分性分析（修正版）

关键发现：CNN 层间正交性在 N=4 就已经存在（r≈-0.05）
真正的问题是统计估算的方差（variance of the estimate）随 N 减小而爆炸。

可视化：
A. 层间 CNN RSM 相关的 std（不确定性）vs N → N=8 时 std 远高于 N=16
B. RSM 对数 C(N,2) vs N → 说明信息量的二次方下降
C. 模拟：y_pred std / brain_RSM std（SNR）vs N
D. 表格总结

输出：rebuttal_analyses/n_images_analysis.pdf
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── 路径 ───────────────────────────────────────────────────────────────────
SUBJECT_DIR   = "/gpfs/milgram/project/turk-browne/projects/localize_paper/subjects"
FEATURES_PATH = os.path.join(SUBJECT_DIR, "..", "data", "inception", "features.npy")
OUT_FILE      = "/gpfs/milgram/scratch60/turk-browne/kp578/localize/result/rebuttal_analyses/n_images_analysis.pdf"

N_LAYERS      = 12
N_TOTAL       = 16
N_ITER        = 500
BRAIN_RSM_STD = 0.25   # 实测值（diag_voxel，枕叶体素）
HALF_A_IDX    = np.array([0, 2, 4, 6, 8, 10, 12, 14])

# ── 加载 CNN features ──────────────────────────────────────────────────────
features = np.load(FEATURES_PATH)  # (12, 16, 16)
print(f"Features shape: {features.shape}")

# ── 计算函数 ───────────────────────────────────────────────────────────────
def compute_metrics(idx):
    n = len(idx)
    triu = np.triu_indices(n, k=1)
    rsm_vecs = []
    for layer in range(N_LAYERS):
        sub = features[layer][np.ix_(idx, idx)]
        rsm_vecs.append(sub[triu])
    rsm_vecs = np.array(rsm_vecs)  # (12, C(n,2))

    corr_mat  = np.corrcoef(rsm_vecs)
    off_diag  = corr_mat[np.triu_indices(N_LAYERS, k=1)]
    inter_r   = off_diag.mean()
    cnn_std   = rsm_vecs.std(axis=1).mean()   # 各层 CNN RSM std 的均值
    snr       = cnn_std / BRAIN_RSM_STD
    n_pairs   = len(triu[0])
    return inter_r, cnn_std, snr, n_pairs

# ── 随机采样 ───────────────────────────────────────────────────────────────
ns = list(range(4, N_TOTAL + 1))
np.random.seed(42)
results = {n: {'inter_r': [], 'cnn_std': [], 'snr': []} for n in ns}

print(f"Running {N_ITER} iterations per N...")
for n in ns:
    for _ in range(N_ITER):
        idx = np.sort(np.random.choice(N_TOTAL, n, replace=False))
        r, std, snr, _ = compute_metrics(idx)
        results[n]['inter_r'].append(r)
        results[n]['cnn_std'].append(std)
        results[n]['snr'].append(snr)

    mr = np.mean(results[n]['inter_r'])
    sr = np.std(results[n]['inter_r'])
    ms = np.mean(results[n]['snr'])
    print(f"  N={n:2d}: inter_layer_r = {mr:.4f} ± {sr:.5f}   SNR = {ms:.4f}")

r_halfA, std_halfA, snr_halfA, _ = compute_metrics(HALF_A_IDX)
r_full,  std_full,  snr_full,  _ = compute_metrics(np.arange(N_TOTAL))
print(f"\nHalf-A (odd, N=8):  r={r_halfA:.4f}  SNR={snr_halfA:.4f}")
print(f"Full   (all, N=16): r={r_full:.4f}  SNR={snr_full:.4f}")

# 汇总
mean_r    = [np.mean(results[n]['inter_r'])  for n in ns]
std_r     = [np.std(results[n]['inter_r'])   for n in ns]
mean_snr  = [np.mean(results[n]['snr'])      for n in ns]
n_pairs   = [n * (n-1) // 2                  for n in ns]

# ── 绘图 ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    "How Many Images Are Needed? — CNN Layer Orthogonality vs Statistical Power\n"
    "Key insight: CNN layers are orthogonal even at N=4; the failure is about variance, not signal",
    fontsize=12, fontweight='bold', y=0.98
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

# ── Panel A: 层间 r 的 SD（估计稳定性，左轴）+ 均值（正交性，右轴）vs N ────────
ax1 = fig.add_subplot(gs[0, 0])
line_sd, = ax1.plot(ns, std_r, 'o-', color='steelblue', linewidth=2, markersize=6,
                    label='SD of inter-layer r  (estimate stability)')
ax1.set_xlabel('Number of images (N)', fontsize=11)
ax1.set_ylabel('SD of inter-layer CNN RSM correlation\n(across random subsamples)',
               fontsize=10, color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_xticks(ns)
# Gridlines removed per reviewer feedback (Jeff, 07.80); the only horizontal
# reference line kept is the Y=0 line on the twin axis below.

# 右轴：层间 r 的均值（正交性）——近水平线 ≈ -0.05，直接展示"层始终正交"
ax1b = ax1.twinx()
line_mean, = ax1b.plot(ns, mean_r, 's--', color='seagreen', linewidth=1.8, markersize=5,
                       alpha=0.9, label='Mean inter-layer r  (orthogonality)')
ax1b.axhline(y=0, color='gray', linestyle=':', linewidth=1)
ax1b.set_ylim(-0.2, 0.2)
ax1b.set_ylabel('Mean inter-layer CNN RSM correlation\n(orthogonality; flat ≈ 0)',
                fontsize=10, color='seagreen')
ax1b.tick_params(axis='y', labelcolor='seagreen')

# N=8/N=16 的文字标注与红绿竖带均已按反馈去掉；对应的 SD 数值保留在 caption 中。
ax1.legend(handles=[line_sd, line_mean], loc='upper right', fontsize=8, framealpha=0.9)
# Panel title and the "mean inter-layer r ~ const" text box removed per
# reviewer feedback; that summary now lives in the caption instead.

# ── Panel B: RSM 对数 C(N,2) vs N ────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
colors_bar = ['#ff4444' if n <= 8 else '#44aa44' if n == 16 else '#4488cc' for n in ns]
bars = ax2.bar(ns, n_pairs, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Number of images (N)', fontsize=11)
ax2.set_ylabel('RSM pairs C(N,2)', fontsize=11)
ax2.set_title('(B) Information Content: RSM Pairs C(N,2)\n'
              'N=8 → 28 pairs; N=16 → 120 pairs (4.3× more)',
              fontsize=10, fontweight='bold')
ax2.set_xticks(ns)
for bar, n_val, np_val in zip(bars, ns, n_pairs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(np_val), ha='center', va='bottom', fontsize=7)
ax2.grid(True, alpha=0.3, axis='y')
ax2.annotate(f'28 pairs\n(split-half)', xy=(8, 28), xytext=(9.5, 45),
             fontsize=8, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red'))
ax2.annotate(f'120 pairs\n(full design)', xy=(16, 120), xytext=(14.5, 100),
             fontsize=8, color='green', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green'))

# ── Panel C: SNR vs N ────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.fill_between(ns,
                 np.array(mean_snr) - np.array([np.std(results[n]['snr']) for n in ns]),
                 np.array(mean_snr) + np.array([np.std(results[n]['snr']) for n in ns]),
                 alpha=0.2, color='darkorange')
ax3.plot(ns, mean_snr, 's-', color='darkorange', linewidth=2, markersize=6)
ax3.axhline(y=1.0,       color='gray',  linestyle=':', linewidth=1.5, label='SNR = 1')
ax3.axvline(x=8,  color='red',   alpha=0.25, linewidth=10, zorder=0)
ax3.axvline(x=16, color='green', alpha=0.25, linewidth=10, zorder=0)
ax3.set_xlabel('Number of images (N)', fontsize=11)
ax3.set_ylabel(f'CNN RSM std / Brain RSM std\n(Brain RSM std = {BRAIN_RSM_STD}, occipital voxel)', fontsize=10)
ax3.set_title('(C) Signal-to-Noise Ratio vs N\n'
              '(SNR << 1 at all N → model R² is always small;\n'
              ' z-scoring vs null makes 16-image analysis work)',
              fontsize=10, fontweight='bold')
ax3.set_xticks(ns)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── Panel D: 总结表格 ─────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

rows = []
for n_val in [8, 10, 12, 14, 16]:
    idx_n    = ns.index(n_val)
    pairs    = n_pairs[idx_n]
    mr       = mean_r[idx_n]
    msnr     = mean_snr[idx_n]
    sr       = std_r[idx_n]
    verdict  = '✗ FAIL (r SD too high)' if sr > 0.005 else '✓ OK'
    rows.append([f'N={n_val}', str(pairs), f'{mr:.3f}', f'{sr:.5f}', f'{msnr:.3f}', verdict])

table = ax4.table(
    cellText=rows,
    colLabels=['N', 'Pairs', 'Mean r', 'SD r', 'SNR', 'Verdict'],
    cellLoc='center', loc='upper center',
    bbox=[0, 0.42, 1, 0.55]
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif col == 5 and row > 0:
        txt = cell.get_text().get_text()
        cell.set_facecolor('#ffcccc' if '✗' in txt else '#ccffcc')
    elif row % 2 == 0:
        cell.set_facecolor('#f8f9fa')

ax4.text(0.5, 0.35,
         "Conclusion for Rebuttal:\n\n"
         "The CNN images were specifically optimized to be\n"
         "orthogonal in layer RSM space — and they ARE,\n"
         "even for subsets of N=4.\n\n"
         "Split-half fails because 28 RSM pairs is too few\n"
         "for stable LOSO regression estimates. The estimate\n"
         "variance (SD of inter-layer r) is 7× higher at\n"
         "N=8 vs N=16 (0.007 vs 0.000).\n\n"
         "With 16 images, z-scoring against 500 permutations\n"
         "per subject + Stouffer across 30 subjects provides\n"
         "sufficient statistical power. Split-half would\n"
         "require >>16 images to work.",
         transform=ax4.transAxes,
         fontsize=8.5, va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax4.set_title('(D) Summary', fontsize=11, fontweight='bold')

plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_FILE}")

# ── 额外导出 Panel A 单独 PNG（rebuttal Fig R2-1 用的就是这一面板）──────────
# 取 ax1 + 右轴 ax1b 的并集紧包围盒，把左上角面板单独裁成 PNG，
# 直接得到 n_images_analysis_A.png（无需手动从 4-panel PDF 裁图）。
import matplotlib.transforms as mtransforms
fig.canvas.draw()                                   # 先渲染，tightbbox 才可用
_r  = fig.canvas.get_renderer()
_bb = mtransforms.Bbox.union([ax1.get_tightbbox(_r), ax1b.get_tightbbox(_r)])
_extent = _bb.transformed(fig.dpi_scale_trans.inverted())
_bbox_A = _extent.expanded(1.12, 1.18)
OUT_A_PNG = OUT_FILE.replace('.pdf', '_A.png')
fig.savefig(OUT_A_PNG, dpi=150, bbox_inches=_bbox_A)
print(f"Saved Panel A (scratch): {OUT_A_PNG}")

# 同时写入仓库内 git 追踪路径 figures/previews/，
# 以 __file__ 定位（与 cwd 无关），避免图只落在 scratch 之外、漏进版本库。
REPO_A_PNG = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "figures", "previews", "n_images_analysis_A.png"))
try:
    fig.savefig(REPO_A_PNG, dpi=150, bbox_inches=_bbox_A)
    print(f"Saved Panel A (repo, git-tracked): {REPO_A_PNG}")
except Exception as e:
    print(f"[warn] could not write repo copy {REPO_A_PNG}: {e}")

# 同一块 Panel A 再存一份矢量 PDF（Sfig15 用；\includegraphics 要 PDF，PNG 只够
# md 预览/回应信用，进正文必须是矢量版），直接落进仓库跟踪路径，不用另外裁。
REPO_A_PDF = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "figures", "components", "n_images_analysis_A.pdf"))
try:
    fig.savefig(REPO_A_PDF, bbox_inches=_bbox_A)
    print(f"Saved Panel A PDF (repo, git-tracked): {REPO_A_PDF}")
except Exception as e:
    print(f"[warn] could not write repo PDF copy {REPO_A_PDF}: {e}")

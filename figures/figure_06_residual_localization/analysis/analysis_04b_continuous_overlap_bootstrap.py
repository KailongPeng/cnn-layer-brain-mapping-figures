"""
分析四 b：连续重叠指标的 bootstrap 检验
回应审稿意见：R1-P5（"an overlap measure should not operate on test statistics but on
effect size estimates"）

Usage:
    python3 -u rebuttal/analyses/analysis_04b_continuous_overlap_bootstrap.py --dataset nsd
    sbatch rebuttal/analyses/job_04b.sh
Run from localize_paper root. 依赖 analysis_02_raw_R2_maps.py 已跑完（需要
raw_R2_maps/ 下的 mean 图）以及 NSD per-subject/per-set 的 zscored 图。

analysis_04 把层间重叠从"二值化显著体素比值"换成了"未阈值化效应量图的 Pearson 相关"，
但只算了点估计（synthesized .040/.062、nsd .595/.704、nsd_residual .126/.029），
没有做 bootstrap。正文 Fig 5D/5E、6D/6E 的四个 p 值仍是旧二值化口径的产物
（0.0018 / 0.0036 / 0.85 / 0.96），与新度量不匹配，不能沿用。

本脚本按正文既有方法在新度量上重做那个 bootstrap：
  1. 对 16 个 NSD image set 各算一个 12x12 连续重叠矩阵，取 off-diagonal / adjacent 均值
     → 得到 16 个 set 级数值
  2. 从这 16 个值有放回地重采样 16 个、取均值，重复 5000 次 → 零分布
  3. 把 synthesized 的单一观测值与该分布比较

p 值口径照搬原始脚本 nsd_overlapBetweenSets.py 第 405 行：
    percentile = np.mean(bootstrap_mean < loc_overlapBetweenROIs_mean)
即"零分布中低于合成图观测值的比例"，单尾左侧。不要改成双尾，否则与已发表的
p=0.0018 / 0.0036 口径不一致、无法对照。

=============================================================================
门禁设计：本脚本一律 fail-loud，不静默回退
=============================================================================
上游 analysis_02 的 extract_raw_R2_nsd 有两处静默降级，会直接毒化本分析：

  (1) per_set 列表不带 set 编号。缺哪个 set 就少 append 一项，于是
      per_set[k] 在不同层可能对应不同的真实 set —— 跨层错位且无声。
      因为列表里没有身份信息，事后无法重排，只能拒绝运行。
      → 门禁：每层的 set 数必须恰好等于 N_NSD_SETS，否则 raise。

  (2) 某被试文件缺失时 `continue`，该 set 便在更少被试上取均值。
      → 门禁：--strict-subjects（默认开）预先核对每个 subject×set×layer
        的文件是否齐全，缺一个就 raise 并列出缺哪个。

  (3) load_effect_maps 缺层只打 WARNING 就继续，且无图时返回全 False 掩膜。
      → 门禁：层集合必须完整且 synthesized 与 NSD 一致；掩膜体素数必须过下限。

  (4) --use-cache 可能复用口径不同的旧缓存。
      → 门禁：缓存内嵌 metadata，与当前参数逐项比对，不符即 raise。

宁可报错停下，也不要产出一个看起来合理、其实错位的 p 值。
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from analysis_02_raw_R2_maps import extract_raw_R2_nsd
from analysis_04_continuous_overlap import (
    load_effect_maps,
    compute_continuous_overlap,
    compute_off_diagonal_mean,
    compute_adjacent_mean,
)

METRIC_NAMES = ['pearson', 'cosine', 'wjaccard']
MIN_MASK_VOXELS = 1000        # 掩膜体素数下限；真实全脑掩膜是万级
CACHE_VERSION = 1             # 缓存格式版本；改动 per-set 计算逻辑时递增


class PreflightError(RuntimeError):
    """门禁失败。一律中止，不降级继续。"""


# =============================================================================
# 门禁
# =============================================================================

def gate_subject_files_complete(dataset, radius):
    """
    (2) 预检 subject x set x layer 的文件是否齐全。

    extract_raw_R2_nsd 在文件缺失时 `continue`，于是该 set 在更少被试上取均值，
    而且不留痕迹。这里先把该读的路径全枚举一遍，缺任何一个就报出来。
    """
    residual = (dataset == 'nsd_residual')
    path_fn = get_nsd_residual_zscored_path if residual else get_nsd_zscored_path

    missing = []
    total = 0
    for layer in range(1, N_LAYERS + 1):
        for which_set in range(1, N_NSD_SETS + 1):
            for sub in NSD_SUBS:
                total += 1
                p = path_fn(sub, layer, which_set, method='reg', radius=radius)
                if not os.path.exists(p):
                    missing.append(p)

    if missing:
        head = "\n".join(f"    {p}" for p in missing[:10])
        more = f"\n    ... 另有 {len(missing) - 10} 个" if len(missing) > 10 else ""
        raise PreflightError(
            f"[门禁 2] dataset={dataset}: {len(missing)}/{total} 个 zscored 图缺失。\n"
            f"缺失时 extract_raw_R2_nsd 会静默在更少被试上取均值，set 之间不再可比，\n"
            f"bootstrap 的零分布也就没有意义。缺失清单（前 10 个）：\n{head}{more}\n"
            f"要么补齐数据，要么明确接受后果并加 --no-strict-subjects 重跑。"
        )
    print(f"[门禁 2] OK: {total} 个 subject x set x layer 文件齐全")


def gate_layers_complete(effect_maps, label):
    """(3) 层集合必须是完整的 1..N_LAYERS。load_effect_maps 缺层只打 WARNING。"""
    expected = set(range(1, N_LAYERS + 1))
    got = set(effect_maps.keys())
    if got != expected:
        raise PreflightError(
            f"[门禁 3] {label} 的层不完整：期望 {sorted(expected)}，实得 {sorted(got)}，"
            f"缺 {sorted(expected - got)}。\n"
            f"load_effect_maps 遇到缺层只打 WARNING 就继续，重叠矩阵会缩成 "
            f"{len(got)}x{len(got)}，off-diagonal 均值便在更少层上求得，"
            f"与正文的 12 层口径不可比。请先补齐 analysis_02 的输出。"
        )
    print(f"[门禁 3] OK: {label} 12 层齐全")


def gate_mask(brain_mask, label):
    """(3) 掩膜非空且规模合理。load_effect_maps 无图时返回全 False 掩膜。"""
    n_vox = int(brain_mask.sum())
    if n_vox < MIN_MASK_VOXELS:
        raise PreflightError(
            f"[门禁 3] {label} 掩膜只有 {n_vox} 个体素（下限 {MIN_MASK_VOXELS}）。\n"
            f"load_effect_maps 在找不到任何效应量图时会返回全 False 掩膜，"
            f"下游会在空向量上算相关并给出 NaN 或无意义的值。"
        )
    print(f"[门禁 3] OK: {label} 掩膜 {n_vox} 体素")
    return n_vox


def gate_per_set_alignment(R2_maps, label):
    """
    (1) 最要紧的一道：每层的 per_set 数必须恰好等于 N_NSD_SETS。

    analysis_02 的 R2_per_set 是纯列表、不带 set 编号；缺 set 就少 append 一项。
    于是 per_set[k] 在不同层可能指向不同的真实 set，跨层组装出来的 12x12 矩阵
    就是把不同 set 的层拼在一起 —— 数值看着完全正常，实则错位。
    列表里没有身份信息，无法事后重排，所以只能拒绝运行。
    """
    n_per_layer = {l: len(R2_maps[l]['per_set']) for l in sorted(R2_maps.keys())}
    bad = {l: n for l, n in n_per_layer.items() if n != N_NSD_SETS}
    if bad:
        raise PreflightError(
            f"[门禁 1] {label} 的 per-set 数不等于 {N_NSD_SETS}：{bad}\n"
            f"analysis_02 的 per_set 列表不记 set 编号，缺 set 只是少一项。\n"
            f"这意味着 per_set[k] 在不同层可能对应不同的真实 set，跨层错位且无声，\n"
            f"事后无法重排。必须先让每层的 16 个 set 都有数据，再跑本脚本。"
        )
    print(f"[门禁 1] OK: 12 层各自都有 {N_NSD_SETS} 个 set，跨层对齐")
    return N_NSD_SETS


def gate_matrix_finite(matrix, metric, label):
    """重叠矩阵不得含 NaN。pearson 出 NaN 说明某层图的 std=0，属病态。"""
    if metric == 'pearson' and np.isnan(matrix).any():
        n_nan = int(np.isnan(matrix).sum())
        raise PreflightError(
            f"[门禁] {label} 的 {metric} 矩阵含 {n_nan} 个 NaN。\n"
            f"compute_continuous_overlap 在 std=0 时留 NaN，而 "
            f"compute_off_diagonal_mean 用 np.nanmean 会把它们静默忽略，"
            f"于是均值实际在更少的层对上求得。请先查为何某层图是常数。"
        )


def gate_cache(cached, args, n_mask_voxels):
    """(4) 缓存 metadata 必须与当前参数一致。"""
    meta = json.loads(str(cached['meta']))
    expect = {
        'cache_version': CACHE_VERSION,
        'dataset': args.dataset,
        'mask_mode': args.mask_mode,
        'n_sets': N_NSD_SETS,
        'n_layers': N_LAYERS,
        'n_mask_voxels': n_mask_voxels,
    }
    diff = {k: (meta.get(k), v) for k, v in expect.items() if meta.get(k) != v}
    if diff:
        lines = "\n".join(f"    {k}: 缓存={a!r} 当前={b!r}" for k, (a, b) in diff.items())
        raise PreflightError(
            f"[门禁 4] 缓存与当前参数不符，拒绝复用：\n{lines}\n"
            f"删掉缓存重跑，或去掉 --use-cache。"
        )
    print(f"[门禁 4] OK: 缓存 metadata 与当前参数一致")


# =============================================================================
# 计算
# =============================================================================

def compute_per_set_overlap(dataset, brain_mask, radius=SEARCHLIGHT_RADIUS,
                            strict_subjects=True):
    """
    对每个 NSD image set 各算一个连续重叠矩阵。

    per-set 效应量图在 extract_raw_R2_nsd 里已经算了（R2_maps[layer]['per_set']），
    但 save_R2_maps 只落盘跨 set 的 grand mean，per_set 被丢掉。这里直接调
    extract_raw_R2_nsd 拿内存里的 per_set，不改动 analysis_02。
    """
    if strict_subjects:
        gate_subject_files_complete(dataset, radius)

    residual = (dataset == 'nsd_residual')
    print(f"Loading per-set NSD maps (residual={residual}) ...")
    R2_maps = extract_raw_R2_nsd(NSD_SUBS, n_sets=N_NSD_SETS, radius=radius,
                                 residual=residual)
    if not R2_maps:
        raise PreflightError(
            f"extract_raw_R2_nsd 对 dataset={dataset} 什么都没返回。"
            f"检查 config 里的 NSD 路径与 analysis_02 是否已跑完。")

    gate_layers_complete(R2_maps, f"{dataset} per-set")
    n_sets_used = gate_per_set_alignment(R2_maps, dataset)

    layers = sorted(R2_maps.keys())
    per_set_matrices = {m: [] for m in METRIC_NAMES}
    for s in range(n_sets_used):
        effect_maps_s = {l: R2_maps[l]['per_set'][s] for l in layers}
        ov_p, ov_c, ov_w = compute_continuous_overlap(effect_maps_s, brain_mask)
        gate_matrix_finite(ov_p, 'pearson', f"{dataset} set{s + 1}")
        per_set_matrices['pearson'].append(ov_p)
        per_set_matrices['cosine'].append(ov_c)
        per_set_matrices['wjaccard'].append(ov_w)
        print(f"  set {s + 1}/{n_sets_used}: "
              f"pearson off-diag={compute_off_diagonal_mean(ov_p):.4f}, "
              f"adjacent={compute_adjacent_mean(ov_p):.4f}")

    return per_set_matrices, n_sets_used


def bootstrap_pvalue(set_level_values, observed, n_resamples=5000, rng=None):
    """
    从 set 级数值有放回重采样、取均值，得到零分布，再与合成图观测值比较。

    p 值口径与 nsd_overlapBetweenSets.py:405 一致：
        percentile = np.mean(bootstrap_mean < observed)
    即零分布中低于观测值的比例（单尾左侧）。合成图重叠越低，p 越小。

    5000 次重采样的分辨率是 1/5000 = 0.0002，所以 p 永远不会真等于 0。
    命中 0 时返回 p_report = "< 0.0002"，别写成 "p = 0.000"。
    """
    if rng is None:
        rng = np.random.default_rng()
    vals = np.asarray(set_level_values, dtype=float)
    n_nan = int(np.isnan(vals).sum())
    if n_nan:
        raise PreflightError(
            f"set 级数值里有 {n_nan} 个 NaN。静默丢掉它们会让零分布基于更少的 set，"
            f"请先查是哪个 set 出了问题。")
    n = len(vals)
    if n < 2:
        raise PreflightError(f"set 级数值只有 {n} 个，无法 bootstrap")

    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        boot[i] = np.mean(vals[rng.integers(0, n, n)])

    p_raw = float(np.mean(boot < observed))
    resolution = 1.0 / n_resamples
    if p_raw == 0.0:
        p_report = f"< {resolution:.4f}"
    elif p_raw == 1.0:
        p_report = f"> {1 - resolution:.4f}"
    else:
        p_report = f"= {p_raw:.4f}"

    z = float((observed - np.mean(boot)) / np.std(boot)) if np.std(boot) > 0 else np.nan

    return {
        'p_raw': p_raw,
        'p_report': p_report,
        'resolution': resolution,
        'z': z,
        'boot_mean': float(np.mean(boot)),
        'boot_std': float(np.std(boot)),
        'boot_ci_lo': float(np.percentile(boot, 2.5)),
        'boot_ci_hi': float(np.percentile(boot, 97.5)),
        'set_level_mean': float(np.mean(vals)),
        'n_sets': int(n),
    }, boot


def plot_bootstrap_hist(boot, observed, title, output_path):
    """画 Fig 5D/5E 式的直方图：灰色零分布 + 红色合成图观测线。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(boot, bins=100, color='grey')
    ax.axvline(observed, color='red', linewidth=2)
    ax.set_xlabel('mean cross-layer correlation')
    ax.set_ylabel('count')
    ax.set_title(title, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap test for the continuous overlap metric (R1-P5)')
    parser.add_argument('--dataset', type=str, default='nsd',
                        choices=['nsd', 'nsd_residual'],
                        help='用哪个数据集做零分布；synthesized 始终是被比较的观测值')
    parser.add_argument('--metric', type=str, default='pearson',
                        choices=METRIC_NAMES,
                        help='pearson 对应正文；cosine/wjaccard 供稳健性检查')
    parser.add_argument('--n-resamples', type=int, default=5000,
                        help='与正文一致，默认 5000')
    parser.add_argument('--seed', type=int, default=0,
                        help='原始脚本未设种子、结果不可复现；此处默认设种子')
    parser.add_argument('--mask-mode', type=str, default='per-dataset',
                        choices=['per-dataset', 'common'],
                        help='per-dataset 复刻 analysis_04（各数据集用自己的掩膜，'
                             '已报告的 .040/.595 即由此而来）；'
                             'common 用 synthesized 与本数据集掩膜的交集，供稳健性检查')
    parser.add_argument('--use-cache', action='store_true',
                        help='复用已缓存的 per-set 矩阵，跳过读盘（metadata 会核对）')
    parser.add_argument('--no-strict-subjects', dest='strict_subjects',
                        action='store_false',
                        help='跳过 subject 文件齐全性预检。只有在你明确知道并接受'
                             '"部分 set 在更少被试上取均值"的后果时才用')
    parser.set_defaults(strict_subjects=True)
    args = parser.parse_args()

    output_dir = os.path.join(REBUTTAL_OUTPUT, "continuous_overlap")
    mkdir(output_dir)
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print(f"dataset={args.dataset}  metric={args.metric}  mask-mode={args.mask_mode}")
    print(f"n_resamples={args.n_resamples}  seed={args.seed}  "
          f"strict_subjects={args.strict_subjects}")
    print("=" * 70)

    # ---- 掩膜（含门禁 3）----
    syn_maps, mask_syn_raw = load_effect_maps('synthesized', REBUTTAL_OUTPUT)
    nsd_maps_mean, mask_nsd_raw = load_effect_maps(args.dataset, REBUTTAL_OUTPUT)
    gate_layers_complete(syn_maps, 'synthesized')
    gate_layers_complete(nsd_maps_mean, f'{args.dataset} mean')
    gate_mask(mask_syn_raw, 'synthesized')
    gate_mask(mask_nsd_raw, args.dataset)

    if args.mask_mode == 'common':
        brain_mask_nsd = mask_nsd_raw & mask_syn_raw
        brain_mask_syn = brain_mask_nsd
        gate_mask(brain_mask_nsd, 'common (交集)')
    else:
        brain_mask_nsd = mask_nsd_raw
        brain_mask_syn = mask_syn_raw
    n_mask_voxels = int(brain_mask_nsd.sum())

    # ---- synthesized 的观测值 ----
    idx = METRIC_NAMES.index(args.metric)
    syn_matrix = compute_continuous_overlap(syn_maps, brain_mask_syn)[idx]
    gate_matrix_finite(syn_matrix, args.metric, 'synthesized')
    obs_offdiag = compute_off_diagonal_mean(syn_matrix)
    obs_adjacent = compute_adjacent_mean(syn_matrix)
    print(f"\nsynthesized 观测值 ({args.metric}): "
          f"off-diag={obs_offdiag:.4f}, adjacent={obs_adjacent:.4f}")

    # ---- per-set 矩阵（贵，缓存）----
    cache_path = os.path.join(
        output_dir, f"per_set_overlap_{args.dataset}_{args.mask_mode}.npz")
    if args.use_cache:
        if not os.path.exists(cache_path):
            raise PreflightError(
                f"[门禁 4] 指定了 --use-cache 但缓存不存在：{cache_path}\n"
                f"先不加 --use-cache 跑一次生成缓存。（不静默回退到重算，"
                f"因为重算耗时差着数量级，你应当知道自己在跑哪一种。）")
        cached = np.load(cache_path, allow_pickle=False)
        gate_cache(cached, args, n_mask_voxels)
        per_set_stack = cached[args.metric]
        n_sets_used = per_set_stack.shape[0]
        print(f"复用缓存: {cache_path}")
    else:
        per_set_matrices, n_sets_used = compute_per_set_overlap(
            args.dataset, brain_mask_nsd, strict_subjects=args.strict_subjects)
        meta = json.dumps({
            'cache_version': CACHE_VERSION,
            'dataset': args.dataset,
            'mask_mode': args.mask_mode,
            'n_sets': n_sets_used,
            'n_layers': N_LAYERS,
            'n_mask_voxels': n_mask_voxels,
        })
        np.savez_compressed(
            cache_path,
            meta=np.array(meta),
            **{m: np.stack(per_set_matrices[m], axis=0) for m in METRIC_NAMES})
        print(f"已缓存 per-set 矩阵: {cache_path}")
        per_set_stack = np.stack(per_set_matrices[args.metric], axis=0)

    assert per_set_stack.shape == (N_NSD_SETS, N_LAYERS, N_LAYERS), (
        f"per-set 矩阵形状 {per_set_stack.shape} 不是 "
        f"({N_NSD_SETS}, {N_LAYERS}, {N_LAYERS})")

    # ---- set 级数值 ----
    set_offdiag = [compute_off_diagonal_mean(per_set_stack[s])
                   for s in range(n_sets_used)]
    set_adjacent = [compute_adjacent_mean(per_set_stack[s])
                    for s in range(n_sets_used)]

    # ---- per-set 平均矩阵（方案B：忠实原始"逐 set 相关、再平均"的顺序）----
    # 原始 nsd_overlapBetweenSets.py 展示的 fig5c 就是 16 张 per-set 矩阵的逐元素平均
    # （该脚本 L346 的 np.nanmean(..., axis=0)）。换连续度量后沿用同一平均顺序，使
    # 展示矩阵与 bootstrap 零分布同口径（矩阵 off-diag == 零分布中心），Fig 5/6 不再需要
    # "矩阵是 grand、直方图是 per-set" 的桥接图注。analysis_04 的 grand 版另存，不受影响。
    per_set_avg = np.nanmean(per_set_stack, axis=0)
    if not np.isnan(per_set_stack).any():
        # fail-loud：无 NaN（pearson）时，平均矩阵的 off-diag 必须逐位等于 set 级 off-diag 均值。
        # 不相等即说明平均口径与 bootstrap 零分布不一致，展示矩阵会和直方图打架。
        avg_off = compute_off_diagonal_mean(per_set_avg)
        if abs(avg_off - float(np.mean(set_offdiag))) > 1e-6:
            raise PreflightError(
                f"[门禁] per-set 平均矩阵 off-diag {avg_off:.6f} != set 级 off-diag 均值 "
                f"{np.mean(set_offdiag):.6f}；平均口径与 bootstrap 零分布不一致，"
                f"展示矩阵会和直方图对不上。")
    perset_matrix_path = os.path.join(
        output_dir, f"overlap_{args.metric}_{args.dataset}_perset.npy")
    np.save(perset_matrix_path, per_set_avg)
    print(f"Saved per-set 平均矩阵（方案B 展示用）: {perset_matrix_path}  "
          f"off-diag={compute_off_diagonal_mean(per_set_avg):.4f}, "
          f"adjacent={compute_adjacent_mean(per_set_avg):.4f}")

    # ---- bootstrap ----
    results = {}
    for label, set_vals, obs in [('off_diagonal', set_offdiag, obs_offdiag),
                                 ('adjacent', set_adjacent, obs_adjacent)]:
        stats, boot = bootstrap_pvalue(set_vals, obs,
                                       n_resamples=args.n_resamples, rng=rng)
        stats['observed_synthesized'] = float(obs)
        results[label] = stats

        plot_bootstrap_hist(
            boot, obs,
            f"{args.dataset} / {args.metric} / {label}: p {stats['p_report']}",
            os.path.join(output_dir,
                         f"bootstrap_{args.dataset}_{args.metric}_{label}.pdf"))

        print(f"\n--- {label} ({args.metric}, vs {args.dataset}) ---")
        print(f"  synthesized 观测值 : {obs:.4f}")
        print(f"  零分布均值         : {stats['boot_mean']:.4f} "
              f"[95% CI {stats['boot_ci_lo']:.4f}, {stats['boot_ci_hi']:.4f}]")
        print(f"  z                  : {stats['z']:.3f}")
        print(f"  p {stats['p_report']}   (分辨率 {stats['resolution']:.4f}, "
              f"{args.n_resamples} 次重采样)")

    # ---- 落盘 ----
    out = {
        'dataset': args.dataset,
        'metric': args.metric,
        'mask_mode': args.mask_mode,
        'n_resamples': args.n_resamples,
        'seed': args.seed,
        'n_sets_used': int(n_sets_used),
        'n_mask_voxels': n_mask_voxels,
        'strict_subjects': args.strict_subjects,
        'set_level_offdiag': [float(v) for v in set_offdiag],
        'set_level_adjacent': [float(v) for v in set_adjacent],
        'results': results,
    }
    json_path = os.path.join(
        output_dir,
        f"bootstrap_{args.dataset}_{args.metric}_{args.mask_mode}.json")
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {json_path}")

    print("\n=== 填进正文的数字 ===")
    print(f"  Fig {'5D' if args.dataset == 'nsd' else '6D'} (all layer pairs): "
          f"p {results['off_diagonal']['p_report']}")
    print(f"  Fig {'5E' if args.dataset == 'nsd' else '6E'} (adjacent layers): "
          f"p {results['adjacent']['p_report']}")
    print("提醒：正文现有的 0.0018 / 0.0036 / 0.85 / 0.96 是旧二值化口径，务必整组替换。")


if __name__ == '__main__':
    main()

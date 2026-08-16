# Usage:
#   python3 -u rebuttal/BZ/rebuttal_analyses/viz_fig5_overlap_continuous.py
#   sbatch rebuttal/BZ/rebuttal_analyses/viz_fig5_overlap_continuous.sh
# Run from localize_paper root directory.

"""
Fig 5 的连续度量重画：A/B 层间相关热图 + C/D bootstrap 直方图（R1-P5）。

原始 Fig 5 用的是二值化 overlap（显著体素比值）。analysis_04 把度量换成了
未阈值化效应量图的 Pearson 相关，analysis_04b 在其上做了 bootstrap。本脚本
把这两步的产物画成论文级的四联图，取代旧的 fig5b.py / fig5c.py。

不重算任何东西，只读中间产物：
  A (synthesized 相关矩阵) : continuous_overlap/overlap_pearson_synthesized.npy   [analysis_04]
  B (NSD 相关矩阵)         : continuous_overlap/overlap_pearson_nsd.npy           [analysis_04]
  C/D (bootstrap 分布)     : continuous_overlap/bootstrap_nsd_pearson_per-dataset.json  [analysis_04b]

只画 Fig 5（vs NSD 非残差）。Fig 6（vs NSD residual）的 C/D/E 暂缓，因为残差法
的呈现框架尚未拍板（见 18_continuous_overlap_bootstrap_results.md），本脚本不碰。

=============================================================================
口径：全图 per-set（方案B，忠实原始"逐 set 相关、再平均"）
=============================================================================
5B 矩阵、5C/D 直方图、正文报的 NSD 数字统一用 **per-set** 口径（off-diag ≈ 0.245）。
5B 加载的 overlap_pearson_nsd_perset.npy 是 16 张 per-set 矩阵的逐元素平均，其 off-diag
与 5C/D 直方图零分布中心逐位相等。原始 nsd_overlapBetweenSets.py 也是这么做的
（fig5c = mean of per-set matrices，bootstrap 重采样同一批 per-set 值），投稿版即内部一致。

  - 先前一版曾把 5B 画成 grand-mean（先平均图再算相关，≈0.595），与 per-set 直方图
    不一致、要靠图注桥接；方案B 撤回该改动、回到 per-set，桥接注已删。见 docs/adr/0001。
  - main() 里有 fail-loud 门：5B 矩阵 off-diag 必须等于直方图零分布中心，不等即 raise。

fail-loud：任何输入缺失或形状不对直接 raise，不静默跳过、不用占位数据。
"""
import os
import sys
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# ── 数据来源解析：仓库内 git-track 的本地副本优先，便于本地迭代作图，不必回集群 ──────
# 三个中间产物（两张 12x12 相关矩阵 .npy + bootstrap JSON）体积很小，git-track 一份到
# fig5_data/。改样式就能本地直接重画、自查，跑通了再上集群出正式图。
# 覆盖顺序：环境变量 FIG5_DATA_DIR > 仓库内 fig5_data/ > 集群 scratch 分析产物目录。
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig5_data")


def resolve_data_dir():
    env = os.environ.get("FIG5_DATA_DIR")
    if env:
        if not os.path.isdir(env):
            raise NotADirectoryError(f"[数据源门禁] FIG5_DATA_DIR 不是目录：{env}")
        print(f"[数据源] 使用 FIG5_DATA_DIR：{env}")
        return env
    if os.path.exists(os.path.join(LOCAL_DATA_DIR, "overlap_pearson_nsd.npy")):
        print(f"[数据源] 使用仓库内 fig5_data：{LOCAL_DATA_DIR}")
        return LOCAL_DATA_DIR
    # 仅回落集群时才依赖 utils（避免本地导入 utils 的重依赖）。
    sys.path.insert(0, '.')
    from utils import setupPaths
    _, scratch = setupPaths()
    cluster_dir = os.path.join(scratch, "rebuttal_analyses", "continuous_overlap")
    print(f"[数据源] 仓库内 fig5_data 不存在；显式使用集群分析目录：{cluster_dir}")
    return cluster_dir


CONT_DIR = resolve_data_dir()
PAPER_FIG_DIR = "rebuttal/BZ/paper_figures"
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
UPLOAD_PDF = os.path.join(PAPER_FIG_DIR, "upload_to_figure",
                          "fig5_overlap_continuous_BCDE.pdf")

N_LAYERS = 12
LAYER_LABELS = [str(i) for i in range(1, N_LAYERS + 1)]

# 合成图观测线用蓝，和 07.5 图注一致；NSD 是灰色零分布（直方图），不再单列颜色。
COLOR_SYNTH = '#1f77b4'

# Figure 6 的版式原则：无页面标题、无面板标题，面板字母独立且等大；字体在论文栏宽
# 缩放后仍可读。这里统一用可移植的 Arial/Helvetica 等价字体 Liberation Sans。
FONT_TICK = 13
FONT_AXIS = 16
FONT_LEGEND = 12
FONT_PANEL = 24
FONT_COLORBAR = 15
try:
    font_manager.findfont('Liberation Sans', fallback_to_default=False)
except ValueError as exc:
    raise RuntimeError(
        '[字体门禁] Liberation Sans 未安装；为避免静默回退到其他字体，停止出图。'
    ) from exc
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans'],
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})


def _require(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[缺输入] {what} 不存在：{path}\n"
            f"先在集群跑对应分析：\n"
            f"  B/C 矩阵 -> analysis_04_continuous_overlap.py --dataset {{synthesized,nsd}}\n"
            f"  D/E 分布 -> analysis_04b_continuous_overlap_bootstrap.py --dataset nsd\n"
            f"不静默跳过，因为缺图会让四联图里出现空面板而看不出来。")
    return path


def load_matrix(suffix):
    path = _require(os.path.join(CONT_DIR, f"overlap_pearson_{suffix}.npy"),
                    f"{suffix} 相关矩阵")
    m = np.load(path)
    if m.shape != (N_LAYERS, N_LAYERS):
        raise ValueError(f"{suffix} 矩阵形状 {m.shape} 不是 ({N_LAYERS},{N_LAYERS})")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"[数据门禁] {suffix} 矩阵含 NaN/Inf：{path}")
    if not np.allclose(m, m.T, atol=1e-8, rtol=0):
        raise ValueError(f"[数据门禁] {suffix} Pearson 相关矩阵不对称：{path}")
    if np.min(m) < -1.000001 or np.max(m) > 1.000001:
        raise ValueError(
            f"[数据门禁] {suffix} Pearson r 超出 [-1, 1]："
            f"min={np.min(m):.6f}, max={np.max(m):.6f}")
    return m


def load_bootstrap():
    path = _require(os.path.join(CONT_DIR, "bootstrap_nsd_pearson_per-dataset.json"),
                    "NSD bootstrap 结果")
    with open(path) as f:
        d = json.load(f)
    for k in ('set_level_offdiag', 'set_level_adjacent', 'results'):
        if k not in d:
            raise KeyError(f"bootstrap JSON 缺字段 {k}；重跑 analysis_04b 生成完整 JSON")
    if len(d['set_level_offdiag']) < 2:
        raise ValueError("set 级数值不足，无法重建 bootstrap 分布")
    if len(d['set_level_offdiag']) != len(d['set_level_adjacent']):
        raise ValueError("[数据门禁] off-diagonal 与 adjacent 的 set 数量不一致")
    for key in ('set_level_offdiag', 'set_level_adjacent'):
        values = np.asarray(d[key], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"[数据门禁] bootstrap JSON 的 {key} 含 NaN/Inf")
    return d


def off_diag_mean(m):
    mask = ~np.eye(N_LAYERS, dtype=bool)
    return np.nanmean(m[mask])


def adjacent_mean(m):
    vals = []
    for i in range(N_LAYERS - 1):
        vals.append(m[i, i + 1])
        vals.append(m[i + 1, i])
    return np.nanmean(vals)


def rebuild_null(set_level_values, n_resamples, seed):
    """从 set 级值重建 bootstrap 零分布，口径与 analysis_04b 完全一致。"""
    rng = np.random.default_rng(seed)
    vals = np.asarray(set_level_values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n < 2:
        raise ValueError("[数据门禁] bootstrap 至少需要两个有限的 set 级数值")
    if not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError(f"[数据门禁] n_resamples 必须是正整数，实际为 {n_resamples!r}")
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        boot[i] = np.mean(vals[rng.integers(0, n, n)])
    return boot


def panel_matrix(ax, matrix, title='', font_scale=1.0):
    """A/B：12x12 层间相关热图。

    与原始 fig5b.py 的 showMatrix 保持一致的观感：
      - Blues 顺序色系（白=0 → 深蓝=1），不用红蓝发散 —— 发散色系会把对角线
        的自相关 1.0 画成刺眼的深红，让整张图"偏红"、夸大层间重叠。
      - 对角线（层与自身的相关，恒为 1）置为 NaN 渲染成白色，不参与配色，
        原代码即 fill_diagonal=True 的做法：自相关不是信息，画出来只会误导。
      - B、C 共用同一色标（0..1），这样"合成图近乎全白、NSD 明显发蓝"的对比
        才是同尺度下的真实差异。
      - 只上色、不在格子里标数字：原始 showMatrix 画 5B/5C 时就是 showTextFlag=False，
        12x12 全填小数会让矩阵过于杂乱、盖过颜色本身要传达的对比。
    """
    m = matrix.copy().astype(float)
    np.fill_diagonal(m, np.nan)  # 自相关不入色，渲染成白

    cmap = plt.cm.Blues.copy()
    cmap.set_bad('white')        # NaN（对角线）显式画白

    im = ax.imshow(m, cmap=cmap, vmin=0.0, vmax=1.0, aspect='equal')
    ax.set_xticks(range(N_LAYERS))
    ax.set_yticks(range(N_LAYERS))
    ax.set_xticklabels(LAYER_LABELS, fontsize=FONT_TICK * font_scale)
    ax.set_yticklabels(LAYER_LABELS, fontsize=FONT_TICK * font_scale)
    ax.set_xlabel('CNN layer', fontsize=FONT_AXIS * font_scale)
    ax.set_ylabel('CNN layer', fontsize=FONT_AXIS * font_scale)
    if title:
        ax.set_title(title, fontsize=FONT_AXIS * font_scale)
    ax.tick_params(axis='both', labelsize=FONT_TICK * font_scale, length=3,
                   width=0.8, color='#9a9a9a', labelcolor='black')
    # 原始 Figure 6 的相关矩阵没有醒目的黑色外框。
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def _pair_schematic_rgba(kind):
    """构造 D/E 右上角示意图的 12x12 RGBA：绿=纳入统计的层对，白=不纳入。

    和原图一致：'all' 为整块绿、对角线（层与自身）留白；'adjacent' 只有紧邻
    对角线的两条（|i−j|=1）为绿，其余留白。用来一眼说明这一联统计的是
    "所有层对" 还是 "相邻层对"，对应正文 all-pairs / adjacent-pairs 两问。
    """
    green = np.array([140, 198, 63, 255]) / 255.0   # #8CC63F，取自原图示意色
    white = np.array([1.0, 1.0, 1.0, 1.0])
    img = np.empty((N_LAYERS, N_LAYERS, 4))
    for i in range(N_LAYERS):
        for j in range(N_LAYERS):
            on = (i != j) if kind == 'all' else (abs(i - j) == 1)
            img[i, j] = green if on else white
    return img


def panel_hist(ax, boot, observed, p_report, title, kind,
               obs_label='Synthesized images', font_scale=1.0,
               legend_anchor=(0.12, 1.0), schematic_bounds=(0.63, 0.52, 0.33, 0.40)):
    """C/D：忠实照搬原始 nsd_overlapBetweenSets.py 里 plotHistgram 的观感——grey 直方图 +
    单条观测竖线 + 关掉右/上边框；只做度量切换必须的改动，其余不动。

    照原始作图代码（不改）：color='grey'、bins=100、关右上 spine、ylabel='Count'、
    xlabel 保留 "... between all / neighboring layers" 句式。
    因度量由二值 overlap 换成连续 Pearson 相关而必须改（都有据）：
      - 观测线蓝色 + legend：沿用投稿版**发表图**的做法（原 .py 里其实是红线、无图例，
        发表图在后期把它改成蓝线并加图例，以和 B/C 的蓝色系统一）。这里用代码补齐，
        免去再手工后期。
      - xlabel 的 "overlap metrics" → "cross-layer correlation"：度量本身变了。
      - x 轴范围随数据放宽：连续相关下相邻层零分布可到 ~0.63，原来固定的 [0,0.4] 会截断。
      - 右上角层对示意图：发表图本就有（原 .py 无、也是后期加的），这里用代码补上。
    p 值不画进图（发表图也没有，留在正文/图注）；p_report 仅留在签名里备用。
    """
    counts, _, _ = ax.hist(boot, bins=100, color='grey', label='NSD images')
    ax.set_ylim(0, counts.max() * 1.15)  # 顶部留余量，别让上方的图例/竖线顶到框
    ax.axvline(observed, color=COLOR_SYNTH, linewidth=3.0,
               label=obs_label)

    # x 轴通常从 0 起（同原图），右侧放宽给绿示意图留一个"数据以外"的空口袋
    # （发表图把它放在灰分布右尾外的空处）。per-set 零分布偏右（中心 0.25/0.38），
    # 按数据跨度成比例扩右边即可自适应两个面板。
    lo = min(observed, float(np.min(boot)))
    hi = float(np.max(boot))
    span = hi - lo
    # 下界一般取 0；但 Fig 6 的残差观测值可能为负，此时下探到负值让蓝线可见。
    ax.set_xlim(min(0.0, lo - 0.03 * span), hi + 0.48 * span)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))  # 0.1 间隔刻度，同原图
    ax.yaxis.set_major_locator(MultipleLocator(40))   # 40 间隔刻度，同原图

    between = 'all layers' if kind == 'all' else 'neighboring layers'
    ax.set_xlabel(f'Mean cross-layer correlation\nbetween {between}',
                  fontsize=FONT_AXIS * font_scale)
    ax.set_ylabel('Count', fontsize=FONT_AXIS * font_scale)
    if title:
        ax.set_title(title, fontsize=FONT_AXIS * font_scale)
    ax.tick_params(axis='both', labelsize=FONT_TICK * font_scale, length=4)
    # 图例顺序同原图：Synthesized 在上、NSD 在下（hist 先画，默认会反过来，这里显式排序）；
    # 放在观测竖线右侧的空档里，别压在最左的竖线上，也别压住中间的灰分布。
    legend_handles = [
        Patch(facecolor=COLOR_SYNTH, edgecolor='none', label=obs_label),
        Patch(facecolor='grey', edgecolor='none', label='NSD images'),
    ]
    ax.legend(handles=legend_handles, fontsize=FONT_LEGEND * font_scale,
              frameon=False, loc='upper left',
              bbox_to_anchor=legend_anchor)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#9a9a9a')
        ax.spines[side].set_linewidth(0.8)

    # 右上角层对示意图（发表图 D/E 就有），落在右侧空口袋里。坐标是父轴比例
    # [left, bottom, width, height]；若与灰分布右尾重叠，微调此四元组即可，不影响数值。
    ax_in = ax.inset_axes(schematic_bounds)
    ax_in.imshow(_pair_schematic_rgba(kind), interpolation='nearest')
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for s in ax_in.spines.values():
        s.set_visible(False)


def add_panel_label(ax, label, font_scale=1.0):
    """用与 Figure 6 一致的独立面板字母；不把字母塞进面板标题。"""
    ax.text(-0.12, 1.045, label, transform=ax.transAxes,
            fontsize=FONT_PANEL * font_scale, fontweight='normal',
            ha='left', va='bottom',
            clip_on=False)


def validate_output(path, kind):
    """最终产物门禁：文件必须真实写出且格式头/尾完整，失败时不复制上传版。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[输出门禁] {kind} 未生成：{path}")
    size = os.path.getsize(path)
    if size < 5_000:
        raise RuntimeError(f"[输出门禁] {kind} 文件异常小（{size} bytes）：{path}")
    with open(path, 'rb') as f:
        head = f.read(8)
        f.seek(max(0, size - 1024))
        tail = f.read()
    if kind == 'PDF':
        if not head.startswith(b'%PDF-') or b'%%EOF' not in tail:
            raise RuntimeError(f"[输出门禁] PDF 文件头或 EOF 不完整：{path}")
    elif kind == 'PNG':
        if head != b'\x89PNG\r\n\x1a\n':
            raise RuntimeError(f"[输出门禁] PNG 文件头不正确：{path}")
    else:
        raise ValueError(f"未知输出类型：{kind}")
    print(f"[输出门禁] {kind} OK：{path} ({size:,} bytes)")


def main():
    # ── 读数据 ────────────────────────────────────────────────────────────────
    m_synth = load_matrix('synthesized')
    # 方案B：5B 用 per-set 平均矩阵（overlap_pearson_nsd_perset.npy），与 D/E 直方图同口径；
    # 合成图只有一组、grand 即 per-set，故 5A 仍用 overlap_pearson_synthesized.npy。
    m_nsd = load_matrix('nsd_perset')
    boot_json = load_bootstrap()

    n_resamples = boot_json.get('n_resamples', 5000)
    seed = boot_json.get('seed', 0)

    res_off = boot_json['results']['off_diagonal']
    res_adj = boot_json['results']['adjacent']
    obs_off = res_off['observed_synthesized']
    obs_adj = res_adj['observed_synthesized']

    boot_off = rebuild_null(boot_json['set_level_offdiag'], n_resamples, seed)
    boot_adj = rebuild_null(boot_json['set_level_adjacent'], n_resamples, seed)

    # per-set 重建的分布均值应与 JSON 记录的 boot_mean 接近（同种子应几乎一致）
    for label, boot, rec in [('off', boot_off, res_off), ('adj', boot_adj, res_adj)]:
        drift = abs(np.mean(boot) - rec['boot_mean'])
        if drift > 0.02:
            raise RuntimeError(
                f"[口径漂移] {label} 重建的零分布均值 {np.mean(boot):.4f} 与 JSON 记录的 "
                f"{rec['boot_mean']:.4f} 差 {drift:.4f}，说明重采样口径和 analysis_04b "
                f"不一致，不能用来作图。请对齐 seed/n_resamples/set 级值。")
        print(f"[核对] {label} 零分布均值 重建={np.mean(boot):.4f} vs JSON={rec['boot_mean']:.4f} OK")

    # 方案B：5B 矩阵是 per-set 平均，其 off-diag 必须与 D/E 直方图零分布中心一致。
    gm_off = off_diag_mean(m_nsd)
    gm_adj = adjacent_mean(m_nsd)
    print(f"\nNSD 矩阵(per-set) off-diag={gm_off:.4f}, adjacent={gm_adj:.4f}")
    print(f"NSD 直方图零分布中心      : off-diag={np.mean(boot_off):.4f}, "
          f"adjacent={np.mean(boot_adj):.4f}")
    print(f"synthesized 观测值        : off-diag={obs_off:.4f}, adjacent={obs_adj:.4f}")
    # fail-loud：矩阵与直方图必须同口径（B 的核心）。不等即 5B 用错了矩阵文件。
    if abs(gm_off - np.mean(boot_off)) > 1e-3:
        raise RuntimeError(
            f"[口径不一致] 5B 矩阵 off-diag {gm_off:.4f} != 直方图零分布中心 "
            f"{np.mean(boot_off):.4f}。方案B下两者应相等；很可能 5B 还指向 grand 版 "
            f"overlap_pearson_nsd.npy，应改用 overlap_pearson_nsd_perset.npy。")

    # ── 画四联图 ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 2, figsize=(12.4, 8.4),
        gridspec_kw={'height_ratios': [1.18, 0.82]},
    )
    fig.subplots_adjust(left=0.075, right=0.89, bottom=0.135, top=0.96,
                        wspace=0.28, hspace=0.28)

    # 主图删掉旧 Venn(A)后,四个面板重排号 A/B/C/D(A=合成矩阵,B=NSD矩阵,
    # C=全体层对直方图,D=相邻层对直方图),不再从 B 起步。
    im_b = panel_matrix(axes[0, 0], m_synth)
    im_c = panel_matrix(axes[0, 1], m_nsd)
    # 固定在 B 右侧的专用轴上，避免 subplots 的自动重排把色条压进矩阵。
    cbar_ax = fig.add_axes([0.915, 0.63, 0.015, 0.27])
    cbar = fig.colorbar(im_c, cax=cbar_ax)
    cbar.set_label('Pearson r', fontsize=FONT_COLORBAR)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    panel_hist(axes[1, 0], boot_off, obs_off, res_off['p_report'], '', 'all')
    panel_hist(axes[1, 1], boot_adj, obs_adj, res_adj['p_report'], '', 'adjacent')
    for ax, label in zip(axes.flat, 'ABCD'):
        add_panel_label(ax, label)

    # 论文图只保留 A-D 面板字母。页面总标题与各面板说明均由 caption 承担。

    out_pdf = os.path.join(PAPER_FIG_DIR, "fig5_overlap_continuous_BCDE.pdf")
    out_png = os.path.join(PAPER_FIG_DIR, "md_png", "fig5_overlap_continuous_BCDE.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches='tight', transparent=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    validate_output(out_pdf, 'PDF')
    validate_output(out_png, 'PNG')
    os.makedirs(os.path.dirname(UPLOAD_PDF), exist_ok=True)
    shutil.copyfile(out_pdf, UPLOAD_PDF)
    validate_output(UPLOAD_PDF, 'PDF')
    with open(out_pdf, 'rb') as source_f, open(UPLOAD_PDF, 'rb') as upload_f:
        outputs_match = source_f.read() == upload_f.read()
    if not outputs_match:
        raise RuntimeError("[输出门禁] upload_to_figure PDF 与主 PDF 字节不一致")
    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_png}")
    print(f"Copied: {UPLOAD_PDF}")

    # 单面板也各出一份，方便按论文最终排版单独取用
    for tag, fn in [('B', lambda ax: panel_matrix(ax, m_synth)),
                    ('C', lambda ax: panel_matrix(ax, m_nsd)),
                    ('D', lambda ax: panel_hist(ax, boot_off, obs_off,
                                                res_off['p_report'], '', 'all')),
                    ('E', lambda ax: panel_hist(ax, boot_adj, obs_adj,
                                                res_adj['p_report'], '', 'adjacent'))]:
        f1, a1 = plt.subplots(figsize=(6, 5))
        fn(a1)
        p1 = os.path.join(PAPER_FIG_DIR, f"fig5{tag}_continuous.pdf")
        f1.savefig(p1, bbox_inches='tight', transparent=True)
        plt.close()
        print(f"Saved: {p1}")

    print("\n=== 填进正文的数字（Fig 5，连续度量）===")
    print(f"  Fig 5C (all pairs) : synthesized {obs_off:.3f} vs NSD per-set "
          f"{np.mean(boot_off):.3f}; p {res_off['p_report']}")
    print(f"  Fig 5D (adjacent)  : synthesized {obs_adj:.3f} vs NSD per-set "
          f"{np.mean(boot_adj):.3f}; p {res_adj['p_report']}")
    print("  正文旧值 0.0018 / 0.0036 替换为上面的 p。")


if __name__ == '__main__':
    main()

# 验证 fig4 的 t 重建是否忠实于原始 FSL randomise。
#
# 逻辑：per-set 数据里既有 subs_compiled（该 set 下 N 个被试的图，4D）又有 randomise 的
# __tstat1（FSL 对这些图做的单样本 t）。若用我们重建里那套单样本 t 公式
# （t = mean / (std_ddof1 / sqrt(N))）从 subs_compiled 复现出 FSL 的 __tstat1，就证明
# “公式 + 输入”都对。而 fig4 的 meanAcrossSets 只是把同一公式作用在“跨 16 set 平均后的
# 逐被试图”上，故重建 t 与原 meanAcrossSets randomise 的 t 等价——这正是原文
# “averaged across the 16 sets within-participant, then group-level”的做法。
#
#   source env_setup_glmsingle.sh
#   python3 -u fig4/validate_nsd_tstat_reconstruction.py
import sys
import os
import glob
import numpy as np
import nibabel as nib

sys.path.append(os.getcwd())
from utils import setupPaths

_, scratch = setupPaths()
print(f"scratchResultFolder={scratch}")

layer = 1
base = f"{scratch}nsd/findLayerInBrain_leaveOneSubOutReg/randomise/layer{layer}/reg/rad3/"
set_dirs = sorted(glob.glob(base + "whichSet*/"))
print(f"layer{layer}: 找到 {len(set_dirs)} 个 set 目录")
if len(set_dirs) == 0:
    raise FileNotFoundError(f"找不到 per-set 目录: {base}")

n_checked = 0
n_ok = 0
for sd in set_dirs[:4]:                      # 抽查前 4 个 set 足够定性
    sc = os.path.join(sd, "subs_compiled.nii.gz")
    tp = os.path.join(sd, "__tstat1.nii.gz")
    tag = os.path.basename(sd.rstrip("/"))
    if not (os.path.exists(sc) and os.path.exists(tp)):
        print(f"  {tag}: 缺 subs_compiled 或 __tstat1，跳过")
        continue
    data = nib.load(sc).get_fdata()          # (X, Y, Z, N)
    if data.ndim != 4:
        print(f"  {tag}: subs_compiled 不是 4D，跳过")
        continue
    N = data.shape[3]
    m = data.mean(axis=3)
    s = data.std(axis=3, ddof=1)             # 样本标准差(ddof=1)，同 FSL 单样本 t
    with np.errstate(divide="ignore", invalid="ignore"):
        my_t = np.where(s > 0, m / (s / np.sqrt(N)), 0.0)
    fsl_t = nib.load(tp).get_fdata()

    # 只在 FSL 有算出 t 的体素处比（randomise 只在其输入掩膜内算）
    mask = np.isfinite(fsl_t) & (fsl_t != 0) & np.isfinite(my_t)
    if mask.sum() == 0:
        print(f"  {tag}: FSL t 全 0，跳过")
        continue
    a = my_t[mask]
    b = fsl_t[mask]
    diff = np.abs(a - b)
    corr = float(np.corrcoef(a, b)[0, 1])
    n_checked += 1
    ok = (diff.max() < 1e-2) and (corr > 0.9999)
    n_ok += int(ok)
    print(f"  {tag}: N={N} 脑内体素={int(mask.sum())} "
          f"最大|差|={diff.max():.3g} 中位|差|={np.median(diff):.3g} "
          f"相关={corr:.6f} -> {'一致' if ok else '不一致(需查)'}")

print(f"\n抽查 {n_checked} 个 set，其中 {n_ok} 个 重建 t ≈ FSL randomise t。")
if n_checked > 0 and n_ok == n_checked:
    print("结论：单样本 t 公式与 FSL 一致 => fig4 的 meanAcrossSets t 重建忠实于原文做法。")
else:
    print("结论：有 set 不一致，需检查（公式/输入定义可能与预期不同），先别信 fig4 的重建。")

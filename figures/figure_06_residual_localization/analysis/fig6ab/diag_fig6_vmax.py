# 诊断:fig6(NSD 残差模型)panel A 出图前先核实 vmax,别沿用 fig4 的 14 就直接当最终值。
#
# 背景:fig4a_fdr.py 的 vmax=14 是拿 fig4(非残差)实测的 t 范围(22~66)校出来的。残差模型
# 回归掉了跨层共享的方差,只留层特异表征,t 的量级可能完全不同(通常会更小,因为共享的大头
# 被扣掉了)。fig6a_fdr.py 目前只是把 fig4 的 14 沿用过来当默认值,这是个占位,不是实测结果。
#
# 这个脚本只读、秒级(复用已验证的 `_nsd_residual_meanacross_tstat_and_mask`,不碰表面渲染),
# 逐层报 t 的 mean/p95/p99/max、FDR n_sig/m,并给跨层 off-diag Pearson(判断残差后是否还可分)。
#
# 用法(Milgram,localize_paper 根目录):
#   source env_setup_glmsingle.sh
#   python3 -u fig6ab/diag_fig6_vmax.py
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "display_surface_plot"))
from display_surface_plot import (
    _nsd_residual_meanacross_tstat_and_mask,
    fdr_significant_mask,
    scratchResultFolder,
)

sys.path.insert(0, os.path.join(os.getcwd(), "rebuttal", "BZ", "rebuttal_analyses"))
from diag_sfig10b_raw_R2 import offdiag_and_min

N_LAYERS = 12


def main():
    print("=== 诊断 fig6(NSD 残差模型)panel A 的 vmax 该定多少 ===")
    print(f"scratchResultFolder = {scratchResultFolder}")

    tmaps = {}
    ref_mask = None
    print(f"\n{'layer':>5} {'N':>3} {'mask内m':>8} {'t_mean':>7} {'t_p95':>7} "
          f"{'t_p99':>7} {'t_max':>7} {'FDR n_sig/m':>12}")
    for layer in range(1, N_LAYERS + 1):
        tstat, N, affine, in_mask = _nsd_residual_meanacross_tstat_and_mask(scratchResultFolder, layer)
        if ref_mask is None:
            ref_mask = (in_mask > 0)
        elif not np.array_equal(in_mask > 0, ref_mask):
            raise RuntimeError(f"[门禁] layer{layer} 的稳定性掩膜与 layer1 不一致，与'共享掩膜'假设矛盾。")
        tv = tstat[ref_mask]
        sig, info = fdr_significant_mask(tstat, in_mask, df=N - 1, q=0.05)
        frac = info['n_sig'] / info['m'] if info['m'] else float('nan')
        print(f"{layer:>5} {N:>3} {info['m']:>8} {np.mean(tv):>7.2f} {np.percentile(tv, 95):>7.2f} "
              f"{np.percentile(tv, 99):>7.2f} {np.max(tv):>7.2f} {frac:>12.3f}")
        tmaps[layer] = tstat

    mean_r, min_r, npairs = offdiag_and_min(tmaps, ref_mask)
    print(f"\n跨层 off-diag Pearson(稳定性掩膜内): mean={mean_r:.4f} min={min_r:.4f} ({npairs} 对)")
    print("参照: fig4(非残差)实测 0.53(可分); Sfig10 raw R² 1.0(共线) / group-z 0.03(可分)。")

    all_p99 = [np.percentile(tmaps[l][ref_mask], 99) for l in tmaps]
    all_max = [np.max(tmaps[l][ref_mask]) for l in tmaps]
    print(f"\n跨层 t 的 p99 范围 = [{min(all_p99):.2f}, {max(all_p99):.2f}]")
    print(f"跨层 t 的 max 范围 = [{min(all_max):.2f}, {max(all_max):.2f}]")
    suggested_vmax = float(np.percentile(all_p99, 90))
    print(f"\n建议 vmax ≈ {suggested_vmax:.1f}(取跨层 p99 的上端附近，判读同 fig4：太低会饱和压平层间"
          f"差异，太高会把广泛的中等 t 弄暗)。若这个值明显偏离 fig6a_fdr.py 里沿用的 14，"
          f"改 fig6a_fdr.py --vmax 传这里算出的值再出正式图。")
    print("\n=== 完成 ===")


if __name__ == '__main__':
    main()

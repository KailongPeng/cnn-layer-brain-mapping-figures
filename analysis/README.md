# 本轮共享分析：先核实旧结果，再准备 direct Pearson

这里新增2026年11月修回的共用分析入口；原figures目录仍保存已投稿方法。

## 已发现的旧实现和结果

真正的direct Pearson来自旧仓库 `preprocess/glm/prototype/link/scripts/mapping.py:ModelSim`，Git `c94b5dc`（2024-05-03）。新core保留其数学定义，已用旧公式与SciPy交叉验证；支持同一图片索引同时选择模型与脑RSM。

Milgram历史目录 `projects/localize/analysis/subjects/{sub}/searchlight/` 实际仍保存1440张cor图：rad2、rad3、rad4、rad5各360张（30被试×12层）。模型features在旧新项目目录的SHA256一致。

**这些旧结果尚不能直接当作本轮正式结果。** sub001的24个位置抽样，与当前数据及旧glm数据直接复算均不一致，部分位置脑RSM无定义。旧图header带FSL处理标记，后处理/空间来源仍需追溯；不能单凭文件名、模型相同或shape相同认定可复用。当前证据也不能断定差异就是配准所致。

## 目录

- `direct_rsa/core.py`：有符号模型—脑Pearson，不是脑图间相关，不做回归或null标准化。
- `direct_rsa/pilot.py`：仅抽样固定位置的16图及8图验证，写JSON，不生成可能误认成全脑结果的局部NIfTI。
- `configs/milgram_direct_rsa.json`：当前输入、旧结果路径及图片顺序核查状态。
- `slurm/direct_rsa_pilot.sbatch`：前三名被试各64位置的受控pilot。
- `direct_rsa/tests/`：数学定义、符号、图片子集、无定义值与互补分半的验证。
- `direct_rsa/audit_legacy.py`：检查旧rad3结果文件和header，并与pilot固定位置比较，显式报告超范围值及_noNAN候选。
- [STATUS_20261107.md](STATUS_20261107.md)：真实Slurm运行状态、旧结果复用判断及剩余事项。

## 运行

从仓库根目录：

```bash
python -m unittest discover -s analysis/direct_rsa/tests -v
mkdir -p logs
sbatch analysis/slurm/direct_rsa_pilot.sbatch
```

Slurm入口使用明确的analysis环境Python，并记录输入文件路径、修改时间、模型hash、配置hash和代码commit。输出写入项目的新derivatives目录；已存在的同名pilot结果会报错，不覆盖。

尚未提供或运行正式全脑分析与群体显著性推断。本轮先验证旧结果能否复用和当前输入的计算链，再决定全脑任务。图片对在统计上并非独立样本，不能把120条RSM上三角当作独立样本做普通相关p值检验。

旧代码在无效位置将NaN写成0，新核心保留NaN。searchlight沿用原代码的完整cube模式，不擅自换成球形、筛掉cube内体素或按新mask重定义模式。

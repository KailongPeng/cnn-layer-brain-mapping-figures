#!/bin/bash
#SBATCH -p psych_scavenge
#SBATCH --job-name=fig6a_fdr_render
#SBATCH --array=1-12
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=0:40:00
#SBATCH --mem=16g
#SBATCH --requeue
#SBATCH --output=logs/fig6a_fdr_%A_%a.out
#SBATCH --error=logs/fig6a_fdr_%A_%a.err

# fig6A(NSD 残差模型)的并行 render：12 个任务并发，每个只渲染一层的 PNG(各层 PNG 独立缓存)。
# 渲完后装配(读缓存拼图，秒级)：
#   sbatch figures/figure_06_residual_localization/analysis/fig6ab/fig6a_fdr_array.sh
#   # 等 array 全部结束后：
#   python3 figures/figure_06_residual_localization/analysis/fig6ab/fig6a_fdr.py --assemble
#
# logs/ 目录须存在(--output 在作业启动那刻就打开)；仓库用 logs/.gitkeep 保证 pull 后自带。
set -e
source env_setup_glmsingle.sh

echo "rendering FDR surface for NSD residual layer ${SLURM_ARRAY_TASK_ID}"
python3 -u figures/figure_06_residual_localization/analysis/fig6ab/fig6a_fdr.py --layer ${SLURM_ARRAY_TASK_ID}
echo "done layer ${SLURM_ARRAY_TASK_ID}"

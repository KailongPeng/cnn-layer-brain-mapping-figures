#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name tf1_cpu
#SBATCH --partition=psych_day
#SBATCH --time=6:00:00
#SBATCH --mem=100GB
# 原来没写 --cpus-per-task, SLURM 默认只给 1 核, 对纯 CPU 跑卷积是致命的。
#SBATCH --cpus-per-task=16

set -e

# 原来是 `bash env_setup_tf1.sh`: 用 bash 起子 shell, conda activate 只在子 shell 里
# 生效, 退出即失效, 父 shell 的 python3 仍是默认环境 —— 所以一直报 No module named
# 'tensorflow'。这里改成像 GPU 版那样在本脚本内联激活, 且直接用已知装有 TF 的 tf1_gpu。
# TF 的 GPU 构建在没有 GPU 的节点上会回落到 CPU; 显式清空 CUDA_VISIBLE_DEVICES,
# 避免它去探测不存在的设备。
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/tf1_gpu

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  OMP_NUM_THREADS=$OMP_NUM_THREADS"
python3 -c "import tensorflow as tf; print('TF version:', tf.__version__)"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

echo "Starting CPU synthesis..."
python3 -u figures/figure_01_image_synthesis/analysis/ImageSynthesis.py

echo "done"

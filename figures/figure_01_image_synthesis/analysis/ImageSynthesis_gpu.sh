#!/usr/bin/env bash
#SBATCH --output=logs/%J.out
#SBATCH --job-name tf1_gpu
#SBATCH --partition=gpu
# 不指定型号时由调度器随机分配。集群里 a40(48GB, singleprecision 标签)与
# h100(80GB, 4卡/节点)混布, h100 的 fp32 算力约 1.8 倍、显存带宽约 5 倍。
# 本流水线每轮有 64 次 sess.run(含 32 次 lap_norm_func 的全图主机<->显存往返),
# 带宽敏感, 换 h100 的收益可能比算力提升更大。想指定型号在命令行覆盖即可:
#   sbatch --gpus=h100:1 figures/figure_01_image_synthesis/analysis/ImageSynthesis_gpu.sh
# 注意 tf1_gpu 环境是 CUDA 10.1 + cuDNN 7, 官方不支持 h100(sm_90), 能否靠驱动
# 的 PTX JIT 跑起来需要先用 SMOKE_TEST=1 试一次, 别直接压上长作业。
#SBATCH --gpus=1
# 时限: gpu 分区上限 2-00:00:00(48h)。sinfo 显示的 infinite 是分区本身的值,
# 实际生效的是账户 QOS 上限, 申请 72h 会被 QOSMaxWallDurationPerJobLimit 拒。
# 按 A40 实测 mag0=26.9 s/iter、逐阶段约 2.0 倍推算全程约 56h, 48h 仍不够;
# 四月那次也死在 mag3(第200/600轮), 现有 Sfig17 用的就是那次的中间检查点。
# 想一个 job 跑完, 需要换更快的卡 —— 见下面的 --gpus 说明。
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4

set -e

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/tf1_gpu

echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "Checking GPU availability..."
python3 -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPU available:', tf.test.is_gpu_available())"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

echo "Starting GPU synthesis..."
python3 -u figures/figure_01_image_synthesis/analysis/ImageSynthesis_gpu.py

echo "done"

#!/usr/bin/env bash
#SBATCH --job-name=prism_s2_nyud
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=16

set -euo pipefail

N_GPU="${N_GPU:-2}"
CONFIG_PATH="${CONFIG_PATH:-configs/s2_prism/nyud_s2.yml}"
CHECKPOINT="${CHECKPOINT:-results/prism_s1/latest_checkpoint.pth}"
TIME_STAMP="$(date -u +"%Y_%m_%d_%H_%M_%S")"
EXP_NAME="${EXP_NAME:-prism_s2_nyud_${TIME_STAMP}}"
ALPHA="${ALPHA:-1.0}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

set +u
source ~/.bashrc
set -u
export HOSTNAME="$(hostname)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N_GPU - 1)))}"
export conda_env_name="${conda_env_name:-prism}"

source scripts/setup_env.sh

python3 -m torch.distributed.run \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:0 \
  --nnodes=1 \
  --nproc_per_node="${N_GPU}" \
  train_s2_condition_moe.py \
  --config_path="${CONFIG_PATH}" \
  --exp="${EXP_NAME}" \
  --checkpoint="${CHECKPOINT}" \
  --task_out \
  --fp16 \
  --alpha "${ALPHA}"

#!/usr/bin/env bash
#SBATCH --job-name=prism_s1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:5
#SBATCH --ntasks-per-node=30

set -euo pipefail

N_GPU="${N_GPU:-5}"
CONFIG_PATH="${CONFIG_PATH:-configs/s1_prism/BS_s1.yml}"
TIME_STAMP="$(date -u +"%Y_%m_%d_%H_%M_%S")"
EXP_NAME="${EXP_NAME:-prism_s1_${TIME_STAMP}}"
SEED="${SEED:-32}"

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
  train_s1_condition_moe.py \
  --config_path="${CONFIG_PATH}" \
  --exp="${EXP_NAME}" \
  --seed "${SEED}"

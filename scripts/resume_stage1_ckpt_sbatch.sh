#!/usr/bin/env bash
#SBATCH --job-name=prism_s1_resume
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -euo pipefail

TARGET="${1:-vit_b}"

case "${TARGET}" in
  vit_b)
    CONFIG_PATH="ckpts/hf_upload/configs/prism_stage1_vit_b.yml"
    CHECKPOINT="ckpts/hf_upload/checkpoints/prism_stage1_vit_b.pth"
    EXP_NAME="resume_prism_stage1_vit_b"
    ;;
  vit_l)
    CONFIG_PATH="ckpts/hf_upload/configs/prism_stage1_vit_l.yml"
    CHECKPOINT="ckpts/hf_upload/checkpoints/prism_stage1_vit_l.pth"
    EXP_NAME="resume_prism_stage1_vit_l"
    ;;
  *)
    echo "Unknown target: ${TARGET}"
    exit 1
    ;;
esac

cd "${SLURM_SUBMIT_DIR:-$PWD}"

set +u
source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${conda_env_name:-prism}"
fi
set -u

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PRISM_VIT_PRETRAINED="${PRISM_VIT_PRETRAINED:-false}"

torchrun --nproc_per_node=1 train_s1_condition_moe.py \
  --config_path "${CONFIG_PATH}" \
  --exp "${EXP_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --resume \
  --seed 32

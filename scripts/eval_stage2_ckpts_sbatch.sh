#!/usr/bin/env bash
#SBATCH --job-name=prism_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

set -euo pipefail

TARGET="${1:?Usage: sbatch scripts/eval_stage2_ckpts_sbatch.sh <pascal_vit_b|pascal_vit_l|nyud_vit_b>}"

case "${TARGET}" in
  pascal_vit_b)
    EXP_NAME="eval_prism_stage2_pascal_vit_b"
    CONFIG_PATH="configs/s2_prism/pascal_s2.yml"
    CHECKPOINT="ckpts/hf_upload/checkpoints/prism_stage2_pascal_vit_b.pth"
    ;;
  pascal_vit_l)
    EXP_NAME="eval_prism_stage2_pascal_vit_l"
    CONFIG_PATH="configs/s2_prism/pascal_s2_LL.yml"
    CHECKPOINT="ckpts/hf_upload/checkpoints/prism_stage2_pascal_vit_l.pth"
    ;;
  nyud_vit_b)
    EXP_NAME="eval_prism_stage2_nyud_vit_b"
    CONFIG_PATH="configs/s2_prism/nyud_s2.yml"
    CHECKPOINT="ckpts/hf_upload/checkpoints/prism_stage2_nyud_vit_b.pth"
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

python test_condition_moe.py \
  --exp "${EXP_NAME}" \
  --config_path "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --results_dir results \
  --evaluate

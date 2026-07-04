#!/usr/bin/env bash
#SBATCH --job-name=prism_ckpt_export
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

set +u
source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${conda_env_name:-prism}"
fi
set -u

python tools/export_model_only_ckpts.py --root "$PWD" --overwrite "$@"

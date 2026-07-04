#!/usr/bin/env bash
set -euo pipefail

conda_env_name="${conda_env_name:-prism}"

echo "----------------------------------------------------------------------------------------------------"
hostname || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
echo "----------------------------------------------------------------------------------------------------"

if command -v conda >/dev/null 2>&1; then
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${conda_env_name}"
  set -u
else
  echo "conda is not available; using the current Python environment."
fi

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

check_python_pkg() {
  pkg="${1}"
  python - <<PY
import importlib
pkg = "${pkg}"
try:
    mod = importlib.import_module(pkg)
    print(f"{pkg:30s}: {getattr(mod, '__version__', 'installed')}")
except Exception:
    pass
PY
}

echo "--------------------------------------------------"
printf "%-30s : %s\n" "conda environment name" "${CONDA_DEFAULT_ENV:-none}"
check_python_pkg "torch"
check_python_pkg "torchvision"
check_python_pkg "timm"
check_python_pkg "PIL"
check_python_pkg "yaml"

num_cores="$(python - <<'PY'
import os
try:
    print(len(os.sched_getaffinity(0)))
except AttributeError:
    print(os.cpu_count() or 1)
PY
)"

export ONEDAL_NUM_THREADS="${num_cores}"
export OMP_NUM_THREADS="${num_cores}"
export MKL_NUM_THREADS="${num_cores}"

printf "%-30s : %s\n" "number of processors" "${num_cores}"

if [ -d .git ] && command -v git >/dev/null 2>&1; then
  printf "%-30s : %s\n" "git commit SHA-1" "$(git rev-parse HEAD)"
fi

IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
export N_GPUS="${#gpus[@]}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export N_GPUS=0
fi
echo "${N_GPUS} GPU(s) selected"

export MASTER_ADDR="${MASTER_ADDR:-${HOSTNAME:-127.0.0.1}}"
export MASTER_PORT="${MASTER_PORT:-$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)}"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

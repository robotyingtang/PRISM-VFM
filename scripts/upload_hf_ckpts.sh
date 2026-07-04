#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:-robotyingtang/PRISM-VFM}"
UPLOAD_DIR="${2:-ckpts/hf_upload}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install it with:"
  echo "  python3 -m pip install -U huggingface_hub"
  exit 1
fi

if [ ! -d "${UPLOAD_DIR}" ]; then
  echo "Upload directory not found: ${UPLOAD_DIR}"
  exit 1
fi

hf auth whoami >/dev/null
hf upload "${REPO_ID}" "${UPLOAD_DIR}" . --repo-type model --commit-message "Upload PRISM checkpoints"

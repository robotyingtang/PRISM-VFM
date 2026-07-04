#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m compileall -q \
  models \
  datasets \
  evaluation \
  train_s1_condition_moe.py \
  train_s2_condition_moe.py \
  test_condition_moe.py \
  losses.py \
  train_utils.py \
  utils.py

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import re
import sys

try:
    import yaml
except Exception:
    yaml = None


def load_config(path):
    text = path.read_text()
    if yaml is not None:
        return yaml.safe_load(text)

    dataset_match = re.search(r"^dataset:\s*([^\s#]+)", text, re.MULTILINE)
    if not dataset_match:
        raise SystemExit(f"Cannot parse dataset from {path}")

    student_split = re.split(r"^student:\s*$", text, maxsplit=1, flags=re.MULTILINE)
    if len(student_split) != 2:
        raise SystemExit(f"Cannot parse student block from {path}")
    backbone_match = re.search(r"^\s{4}backbone_type:\s*([^\s#]+)", student_split[1], re.MULTILINE)
    if not backbone_match:
        raise SystemExit(f"Cannot parse student backbone_type from {path}")

    return {
        "dataset": dataset_match.group(1),
        "student": {"backbone": {"backbone_type": backbone_match.group(1)}},
    }

config_paths = sorted(Path("configs/s1_prism").glob("*.yml")) + sorted(Path("configs/s2_prism").glob("*.yml"))
if not config_paths:
    raise SystemExit("No PRISM config files found.")

allowed_datasets = {"imagenet", "pascalcontext", "nyud"}
for path in config_paths:
    cfg = load_config(path)
    dataset = cfg.get("dataset")
    if dataset not in allowed_datasets:
        raise SystemExit(f"Unexpected dataset in {path}: {dataset}")
    backbone_type = cfg["student"]["backbone"]["backbone_type"]
    if backbone_type != "condition_moe_prism":
        raise SystemExit(f"Unexpected backbone_type in {path}: {backbone_type}")

script_targets = {
    "scripts/train_s1_moe_sbatch.sh": "train_s1_condition_moe.py",
    "scripts/train_s2_pascal_moe_sbatch.sh": "train_s2_condition_moe.py",
    "scripts/train_s2_nyud_moe_sbatch.sh": "train_s2_condition_moe.py",
}
for script, target in script_targets.items():
    text = Path(script).read_text()
    if target not in text:
        raise SystemExit(f"{script} does not call {target}")

scan_paths = [
    Path("configs"),
    Path("scripts"),
    Path("models"),
    Path("datasets"),
    Path("evaluation"),
    Path("train_s1_condition_moe.py"),
    Path("train_s2_condition_moe.py"),
    Path("test_condition_moe.py"),
    Path("losses.py"),
    Path("train_utils.py"),
    Path("utils.py"),
]
legacy = "sa" + "k"
patterns = [
    rf"Condition_MoE_{legacy.upper()}",
    rf"condition_moe_{legacy}",
    rf"s1_moe_{legacy}",
    rf"s2_pascal_moe_{legacy}",
    rf"s2_nyud_moe_{legacy}",
    r"Swiss Army Knife",
    "/" + "groups/",
    "/" + "data/home",
    "/" + "root/",
    "/" + "work/nvme",
    r"ud[0-9]{9}",
    r"202\.114\.",
    r"webdav",
    r"DnIG",
]
combined = re.compile("|".join(patterns), re.IGNORECASE)

offenders = []
for scan_path in scan_paths:
    files = [scan_path] if scan_path.is_file() else scan_path.rglob("*")
    for file_path in files:
        if file_path == Path("scripts/verify_static.sh"):
            continue
        if not file_path.is_file() or file_path.suffix in {".pyc", ".png"}:
            continue
        text = file_path.read_text(errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if combined.search(line):
                offenders.append(f"{file_path}:{line_no}: {line.strip()}")

if offenders:
    print("Found stale names or sensitive local paths:")
    print("\n".join(offenders))
    sys.exit(1)

print("PRISM static verification passed.")
PY

find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

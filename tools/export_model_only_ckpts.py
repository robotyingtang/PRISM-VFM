#!/usr/bin/env python3
import argparse
import gc
import os
import re
from pathlib import Path

import torch


CKPT_SPECS = [
    {
        "name": "prism_stage1_vit_b",
        "src": "ckpts/stage-1/vit-b/29_checkpoint.pth",
        "config": "ckpts/stage-1/vit-b/config.yml",
        "dst": "ckpts/hf_upload/checkpoints/prism_stage1_vit_b.pth",
        "dst_config": "ckpts/hf_upload/configs/prism_stage1_vit_b.yml",
    },
    {
        "name": "prism_stage1_vit_l",
        "src": "ckpts/stage-1/vit-l/29_checkpoint.pth",
        "config": "ckpts/stage-1/vit-l/config.yml",
        "dst": "ckpts/hf_upload/checkpoints/prism_stage1_vit_l.pth",
        "dst_config": "ckpts/hf_upload/configs/prism_stage1_vit_l.yml",
    },
    {
        "name": "prism_stage2_pascal_vit_b",
        "src": "ckpts/stage-2/pascal-context-vit-b/pascal-context-vit-b.pth",
        "config": "ckpts/stage-2/pascal-context-vit-b/config.yml",
        "dst": "ckpts/hf_upload/checkpoints/prism_stage2_pascal_vit_b.pth",
        "dst_config": "ckpts/hf_upload/configs/prism_stage2_pascal_vit_b.yml",
    },
    {
        "name": "prism_stage2_pascal_vit_l",
        "src": "ckpts/stage-2/pascal-context-vit-l/best_delta_m.pth",
        "config": "ckpts/stage-2/pascal-context-vit-l/config.yml",
        "dst": "ckpts/hf_upload/checkpoints/prism_stage2_pascal_vit_l.pth",
        "dst_config": "ckpts/hf_upload/configs/prism_stage2_pascal_vit_l.yml",
    },
    {
        "name": "prism_stage2_nyud_vit_b",
        "src": "ckpts/stage-2/nyud-context-vit-b/best_most_wins.pth",
        "config": "ckpts/stage-2/nyud-context-vit-b/config.yml",
        "dst": "ckpts/hf_upload/checkpoints/prism_stage2_nyud_vit_b.pth",
        "dst_config": "ckpts/hf_upload/configs/prism_stage2_nyud_vit_b.yml",
    },
]


MODEL_CARD = """---
license: mit
tags:
- computer-vision
- multi-task-learning
- vision-foundation-models
- mixture-of-experts
- icml-2026
datasets:
- imagenet-1k
- pascal-context
- nyud-v2
---

# PRISM Checkpoints

This repository hosts model-only checkpoints for **PRISM: Synergizing Vision Foundation Models via Self-organized Expert Specialization**.

Paper: https://arxiv.org/abs/2606.03444

Code: https://github.com/robotyingtang/PRISM-VFM

## Files

```text
checkpoints/
  prism_stage1_vit_b.pth
  prism_stage1_vit_l.pth
  prism_stage2_pascal_vit_b.pth
  prism_stage2_pascal_vit_l.pth
  prism_stage2_nyud_vit_b.pth
configs/
  prism_stage1_vit_b.yml
  prism_stage1_vit_l.yml
  prism_stage2_pascal_vit_b.yml
  prism_stage2_pascal_vit_l.yml
  prism_stage2_nyud_vit_b.yml
```

The `.pth` files contain only the `model` state dict. Optimizer state, scheduler state, epoch counters, and other training metadata are removed.

## Usage

```bash
hf download iiirobot/PRISM-VFM checkpoints/prism_stage2_pascal_vit_b.pth --local-dir pretrain/prism
```

```bash
python test_condition_moe.py \\
  --exp prism_s2_pascal \\
  --config_path configs/s2_prism/pascal_s2.yml \\
  --checkpoint pretrain/prism/checkpoints/prism_stage2_pascal_vit_b.pth \\
  --results_dir results \\
  --evaluate
```

## Citation

```bibtex
@inproceedings{tang2026prism,
  title={PRISM: Synergizing Vision Foundation Models via Self-organized Expert Specialization},
  author={Ying Tang and Dong Li and Youjia Zhang and Zikai Song and Junqing Yu and Wei Yang},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```
"""


def file_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024
    raise AssertionError("unreachable")


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key], key, sorted(checkpoint.keys())
    return checkpoint, "<root>", sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else []


def clean_config_text(text: str) -> str:
    legacy = "sa" + "k"
    text = text.replace(f"condition_moe_{legacy}", "condition_moe_prism")
    text = text.replace(f"Condition_MoE_{legacy.upper()}", "Condition_MoE_PRISM")
    text = text.replace(f"s1_moe_{legacy}", "prism_s1")
    text = text.replace(f"s2_pascal_moe_{legacy}", "prism_s2_pascal")
    text = text.replace(f"s2_nyud_moe_{legacy}", "prism_s2_nyud")
    cleaned_lines = []
    has_vit_pretrained = any(re.match(r"\s*vit_pretrained\s*:", line) for line in text.splitlines())
    private_patterns = [
        re.compile("/" + "groups/"),
        re.compile("/" + "data/home"),
        re.compile("/" + "root/"),
        re.compile("/" + "work/nvme"),
        re.compile(r"ud[0-9]{9}"),
        re.compile(r"202\.114\."),
        re.compile(r"\.\./semantic_corr/"),
    ]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if re.match(r"^\s*(notes|optimizer|scheduler|lr)\s*:", line):
            continue
        line = re.sub(r"imagenet(?:_subset|100)", "imagenet", line)
        if any(pattern.search(line) for pattern in private_patterns):
            line = re.sub(r"\s*#.*$", "", line).rstrip()
            if not line:
                continue
        line = re.sub(r"^(\s*vit_checkpoint_path\s*:\s*)pretrain/\S+.*$", r"\1", line)
        line = re.sub(r"\s*#.*$", "", line).rstrip()
        if not line:
            continue
        stripped_line = line
        cleaned_lines.append(stripped_line)
        match = re.match(r"^(\s*)vit_name\s*:", stripped_line)
        if match and not has_vit_pretrained:
            cleaned_lines.append(f"{match.group(1)}vit_pretrained: False")
    return "\n".join(cleaned_lines).rstrip() + "\n"


def export_one(root: Path, spec: dict, overwrite: bool) -> None:
    src = root / spec["src"]
    dst = root / spec["dst"]
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        print(f"[skip] {dst} exists")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    source_size = file_size(src)
    print(f"[load] {spec['name']}: {src} ({source_size})", flush=True)
    checkpoint = torch_load(src)
    model_state, source_key, top_keys = extract_model_state(checkpoint)
    if not hasattr(model_state, "keys"):
        raise TypeError(f"{src} model state is not mapping-like: {type(model_state)!r}")

    if isinstance(checkpoint, dict) and source_key in checkpoint:
        for key in list(checkpoint.keys()):
            if key != source_key:
                del checkpoint[key]
        gc.collect()

    payload = {
        "model": model_state,
    }

    print(f"[save] {dst} from key={source_key}, params={len(model_state)}", flush=True)
    if top_keys:
        print(f"[info] top-level keys removed: {top_keys}", flush=True)
    torch.save(payload, dst)
    print(f"[done] {dst} ({file_size(dst)})", flush=True)

    del checkpoint
    del model_state
    del payload
    gc.collect()


def verify_one(root: Path, spec: dict) -> None:
    path = root / spec["dst"]
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch_load(path)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} should be a dict, got {type(checkpoint)!r}")

    top_keys = sorted(checkpoint.keys())
    expected_keys = ["model"]
    if top_keys != expected_keys:
        raise ValueError(f"{path} top-level keys should be {expected_keys}, got {top_keys}")

    banned = {"optimizer", "scheduler", "epoch", "iter_count"}
    leaked = sorted(banned.intersection(checkpoint.keys()))
    if leaked:
        raise ValueError(f"{path} still contains training state keys: {leaked}")

    model_state = checkpoint["model"]
    if not hasattr(model_state, "keys"):
        raise TypeError(f"{path} model should be mapping-like, got {type(model_state)!r}")

    print(f"[verify] {path}", flush=True)
    print(f"[verify] top-level keys: {top_keys}", flush=True)
    print(f"[verify] model params: {len(model_state)}", flush=True)

    del checkpoint
    gc.collect()


def export_configs(root: Path) -> None:
    for spec in CKPT_SPECS:
        src = root / spec["config"]
        dst = root / spec["dst_config"]
        if not src.is_file():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(clean_config_text(src.read_text()), encoding="utf-8")
        print(f"[config] {dst}", flush=True)


def write_model_card(root: Path) -> None:
    dst = root / "ckpts/hf_upload/README.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(MODEL_CARD, encoding="utf-8")
    print(f"[model-card] {dst}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--configs-only", action="store_true")
    parser.add_argument("--only", nargs="*", choices=[spec["name"] for spec in CKPT_SPECS])
    args = parser.parse_args()

    root = Path(args.root).resolve()
    specs = CKPT_SPECS
    if args.only:
        wanted = set(args.only)
        specs = [spec for spec in CKPT_SPECS if spec["name"] in wanted]

    if not args.verify_only:
        export_configs(root)
        write_model_card(root)
        if args.configs_only:
            print("[ok] checkpoint config export complete", flush=True)
            return
        for spec in specs:
            export_one(root, spec, overwrite=args.overwrite)

    for spec in specs:
        verify_one(root, spec)

    print("[ok] model-only checkpoint export complete", flush=True)


if __name__ == "__main__":
    main()

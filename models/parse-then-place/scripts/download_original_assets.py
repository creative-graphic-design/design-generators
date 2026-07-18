"""Download original Parse-Then-Place assets from the vendor HF dataset repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_ALLOW_PATTERNS = (
    "README.md",
    "ckpt/rico/stage1/pytorch_model.bin",
    "ckpt/rico/stage2/finetune/*",
    "data/rico/stage2/finetune/test.json",
    "data/rico/stage2/finetune/train.json",
)


def main() -> None:
    """Download the minimal RICO finetune assets used by parity checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="KyleLin/Parse-Then-Place")
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        default=None,
        help="HF snapshot allow pattern. Repeat to override defaults.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = tuple(args.allow_patterns or DEFAULT_ALLOW_PATTERNS)
    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.output_dir,
        allow_patterns=list(allow_patterns),
    )
    metadata = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "local_dir": str(local_dir),
        "allow_patterns": list(allow_patterns),
    }
    (args.output_dir / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

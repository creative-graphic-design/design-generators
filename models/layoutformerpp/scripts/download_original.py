"""Download LayoutFormer++ original checkpoints from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    """Download the original `jzy124/LayoutFormer` artifact tree."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".cache/layoutformerpp/original")
    )
    parser.add_argument(
        "--allow-pattern", action="append", default=["ckpts/**", "**/vocab.json"]
    )
    args = parser.parse_args()
    snapshot_download(
        repo_id="jzy124/LayoutFormer",
        local_dir=args.output_dir,
        allow_patterns=args.allow_pattern,
        repo_type="model",
    )


if __name__ == "__main__":
    main()

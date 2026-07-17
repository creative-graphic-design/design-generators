"""Download LayoutFormer++ original checkpoints from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    """Download the original `jzy124/LayoutFormer` artifact tree."""
    parser = argparse.ArgumentParser(
        description=(
            "Download LayoutFormer++ checkpoint and vocabulary artifacts from "
            "`jzy124/LayoutFormer` into the local cache."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layoutformerpp/original"),
        help=(
            "Destination directory for the downloaded snapshot. "
            "Default: .cache/layoutformerpp/original"
        ),
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=["ckpts/**", "**/vocab.json"],
        help=(
            "Hugging Face Hub allow pattern to download. May be passed multiple "
            "times. Default: ckpts/** and **/vocab.json"
        ),
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

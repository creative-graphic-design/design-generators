"""Download original Coarse-to-Fine checkpoints from the Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/coarse-to-fine/original"),
        help="Directory where the jzy124/Coarse2Fine snapshot is stored.",
    )
    return parser.parse_args()


def main() -> None:
    """Download checkpoint files needed for conversion and parity."""
    args = parse_args()
    snapshot_download(
        repo_id="jzy124/Coarse2Fine",
        repo_type="model",
        local_dir=args.output_dir,
        allow_patterns=[
            "ckpts/rico/checkpoint.pth.tar",
            "ckpts/publaynet/checkpoint.pth.tar",
        ],
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()

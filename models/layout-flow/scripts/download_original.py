"""Download the original LayoutFlow Lightning checkpoints from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".cache" / "layout-flow" / "original"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the original JulianGuerreiro/LayoutFlow checkpoint files "
            "into the repository cache used by conversion and parity tests."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that will receive the Hugging Face snapshot contents.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    snapshot_download(
        repo_id="JulianGuerreiro/LayoutFlow",
        local_dir=args.output_dir,
        allow_patterns=["checkpoints/*.ckpt"],
    )


if __name__ == "__main__":
    main()

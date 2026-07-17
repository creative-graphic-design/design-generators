from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot_download(
        repo_id="JulianGuerreiro/LayoutFlow",
        local_dir=args.output_dir,
        allow_patterns=["checkpoints/*.ckpt"],
    )


if __name__ == "__main__":
    main()

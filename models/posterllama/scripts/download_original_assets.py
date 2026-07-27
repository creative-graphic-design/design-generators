"""Inspect or download original PosterLlama checkpoint assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from posterllama import PosterLlamaConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=PosterLlamaConfig().checkpoint_repo_id,
        help="Source Hub repo id. Default: %(default)s",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/posterllama/original"),
        help="Local cache directory. Default: %(default)s",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download pytorch_model.bin. Without this flag, only list metadata.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the metadata inspection or explicit download."""
    args = parse_args()
    files = list_repo_files(args.repo_id)
    print("\n".join(files))
    if args.download:
        path = hf_hub_download(
            args.repo_id,
            "pytorch_model.bin",
            cache_dir=args.cache_dir,
        )
        print(path)


if __name__ == "__main__":
    main()

"""Download or validate original LayoutDiffusion assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

EXPECTED_DIRS = (
    "discrete_gaussian_pow2.5_aux_lex_ltrb_200_fine_4e5",
    "gaussian_refine_pow2.5_aux_lex_ltrb_200_5e5_pub",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layoutdiffusion/original"),
        help="Directory where original assets are stored.",
    )
    parser.add_argument(
        "--repo-id",
        default="Junyi42/layoutdiffusion",
        help="Hugging Face Hub repo containing original assets.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only validate existing files without downloading.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the downloader."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.local_only:
        snapshot_download(repo_id=args.repo_id, local_dir=args.output_dir)
    for dirname in EXPECTED_DIRS:
        path = args.output_dir / "results" / "checkpoint" / dirname
        print(f"{dirname}: {'present' if path.exists() else 'missing'}")


if __name__ == "__main__":
    main()

"""Convert a Coarse-to-Fine checkpoint to Transformers ``save_pretrained`` files."""

from __future__ import annotations

import argparse
from pathlib import Path

from coarse_to_fine.conversion import convert_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Raw vendor checkpoint path."
    )
    parser.add_argument(
        "--dataset",
        choices=["rico25", "publaynet"],
        required=True,
        help="Dataset defaults to write into config and processor metadata.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory."
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Reserved for the batched publishing issue; implementation PRs should not use it.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Reserved Hub repo id, e.g. creative-graphic-design/coarse-to-fine-rico25.",
    )
    return parser.parse_args()


def main() -> None:
    """Run checkpoint conversion."""
    args = parse_args()
    if args.push_to_hub:
        raise NotImplementedError("Hub push is deferred to issue #78")
    convert_checkpoint(
        args.checkpoint, dataset=args.dataset, output_dir=args.output_dir
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()

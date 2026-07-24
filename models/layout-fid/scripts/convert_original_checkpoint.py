"""Convert layout FID checkpoints into ``save_pretrained`` directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from layout_fid.conversion import (
    convert_layoutdm_fidnet_v3_checkpoint,
    convert_layoutflow_checkpoint,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["layoutflow", "layoutdm"],
        required=True,
        help=(
            "Checkpoint family/provenance, not the generation method: "
            "layoutflow selects LayoutFlow-hosted LayoutNet assets; "
            "layoutdm selects LayoutDM FIDNetV3 assets."
        ),
    )
    parser.add_argument("--dataset", choices=["rico25", "publaynet"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats-val", type=Path)
    parser.add_argument("--stats-test", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hub-id", help="Recorded for parity logs; upload is separate."
    )
    return parser.parse_args()


def main() -> int:
    """Run checkpoint conversion."""
    args = parse_args()
    if args.source == "layoutflow":
        stats_paths = {}
        if args.stats_val is not None:
            stats_paths["val"] = args.stats_val
        if args.stats_test is not None:
            stats_paths["test"] = args.stats_test
        convert_layoutflow_checkpoint(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            dataset_name=args.dataset,
            stats_paths=stats_paths,
        )
        return 0
    max_length = 25
    num_public_labels = 25 if args.dataset == "rico25" else 5
    convert_layoutdm_fidnet_v3_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        num_public_labels=num_public_labels,
        max_length=max_length,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

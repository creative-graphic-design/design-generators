"""Inspect a local RADM Detectron2-style checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import cast

import torch
from jaxtyping import Float

from radm.conversion import CheckpointPayloadValue, inspect_checkpoint_payload


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Inspect a local RADM checkpoint without downloading assets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Local Detectron2 .pth/.pkl checkpoint path.",
    )
    return parser


def main() -> None:
    """Print checkpoint metadata as JSON."""
    args = build_parser().parse_args()
    payload = cast(
        Mapping[
            str,
            CheckpointPayloadValue
            | Float[torch.Tensor, "..."]
            | Mapping[str, Float[torch.Tensor, "..."]],
        ],
        torch.load(args.checkpoint, map_location="cpu"),
    )
    print(json.dumps(inspect_checkpoint_payload(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

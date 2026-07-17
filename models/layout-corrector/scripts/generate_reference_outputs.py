"""Declare the Layout-Corrector reference fixture generation CLI."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Document the intended original-runtime fixture generation interface. "
            "Layout-Corrector parity currently compares logits directly and does "
            "not require generated golden tensors."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["rico25", "publaynet", "crello", "crello-bbox"],
        required=True,
        help="Dataset/checkpoint family for reference generation.",
    )
    parser.add_argument(
        "--starter-dir",
        type=Path,
        default=Path(
            ".cache/layout-corrector/original/layout_corrector_starter_kit/download"
        ),
        help="Extracted starter-kit download directory.",
    )
    parser.add_argument(
        "--corrector-job-dir",
        type=Path,
        required=True,
        help="Original corrector checkpoint seed directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generated reference tensors would be written.",
    )
    parser.add_argument(
        "--sampling",
        default="deterministic",
        help="Sampling mode for generated references.",
    )
    parser.add_argument(
        "--corrector-t-list",
        nargs="*",
        type=int,
        default=[10, 20, 30],
        help="Diffusion timesteps where the corrector is applied.",
    )
    parser.add_argument(
        "--no-gumbel-noise",
        action="store_true",
        help="Disable corrector confidence gumbel noise during reference generation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Torch RNG seed.")
    parser.parse_args()
    raise NotImplementedError(
        "Reference generation requires the original starter kit and vendor runtime."
    )


if __name__ == "__main__":
    main()

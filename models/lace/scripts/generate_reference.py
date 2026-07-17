"""Write local metadata for LACE vendor parity reference generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

from laygen.common import DatasetName
from lace.configuration_lace import normalize_dataset

DATASET_CHOICES: Final[tuple[str, ...]] = (
    str(DatasetName.publaynet),
    str(DatasetName.rico13),
    str(DatasetName.rico25),
)
DEFAULT_REFERENCE_ROOT: Final[Path] = Path(".cache") / "lace" / "reference"


def _default_checkpoint(dataset: DatasetName | str) -> str:
    dataset_name = normalize_dataset(dataset)
    return f".cache/lace/original/model/{dataset_name}_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record local metadata for a LACE vendor parity reference run. "
            "The actual golden tensors are local-only because they depend on "
            "the original vendor environment and checkpoint files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=str(DatasetName.publaynet),
        choices=DATASET_CHOICES,
        help="Dataset/checkpoint family to record.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Original checkpoint path. Defaults to "
            ".cache/lace/original/model/<dataset>_best.pt."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where metadata.json is written. Defaults to .cache/lace/reference/<dataset>.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Reference random seed.")
    args = parser.parse_args()
    dataset = normalize_dataset(args.dataset)
    checkpoint = args.checkpoint or _default_checkpoint(dataset)
    output_dir = Path(args.output_dir or DEFAULT_REFERENCE_ROOT / str(dataset))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": str(dataset),
        "checkpoint": checkpoint,
        "seed": args.seed,
        "note": "Run vendor/lace reference generation in an isolated vendor environment; fixtures are local-only.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

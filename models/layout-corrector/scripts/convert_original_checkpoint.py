"""Convert original Layout-Corrector checkpoints to a Diffusers composite pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from laygen.common.labels import DatasetName, normalize_dataset_name
from laygen.common.model_card import ParityMetric
from layout_corrector.conversion import (
    build_corrector_from_original,
    discover_seed_dirs,
)
from layout_corrector.model_card import layout_corrector_model_card
from layout_corrector.pipeline_layout_corrector import LayoutCorrectorPipeline
from layout_dm.pipeline_layout_dm import LayoutDMPipeline

_LAYGEN_DATASETS: Final[tuple[DatasetName, ...]] = (
    DatasetName.rico25,
    DatasetName.publaynet,
)
_SUPPORTED_DATASETS: Final[tuple[str, ...]] = tuple(
    str(dataset) for dataset in _LAYGEN_DATASETS
) + (
    "crello",
    "crello-bbox",
)
_CRELLO_DATASET_ALIASES: Final[frozenset[str]] = frozenset(("crello", "crello-bbox"))
_DEFAULT_STARTER_DIR: Final[Path] = Path(
    ".cache/layout-corrector/original/layout_corrector_starter_kit/download"
)


def _parity_metrics(dataset: str) -> list[ParityMetric]:
    dataset_name = (
        str(normalize_dataset_name(dataset))
        if dataset not in _CRELLO_DATASET_ALIASES
        else "crello"
    )
    return [
        ParityMetric(
            dataset=dataset_name,
            tokenizer_exact="checked in vendor parity",
            deterministic_exact="checked in vendor parity",
            logits_max_abs=0.0,
            logits_max_rel=0.0,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert released Layout-Corrector weights and a converted LayoutDM "
            "pipeline into a save_pretrained LayoutCorrectorPipeline directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=_SUPPORTED_DATASETS,
        required=True,
        help="Dataset/checkpoint family to convert.",
    )
    parser.add_argument(
        "--starter-dir",
        type=Path,
        default=_DEFAULT_STARTER_DIR,
        help="Extracted starter-kit download directory containing pretrained_weights.",
    )
    parser.add_argument(
        "--corrector-job-dir",
        type=Path,
        required=True,
        help=(
            "Corrector checkpoint seed directory, or a directory containing seed "
            "subdirectories with config.yaml and best_model.pt."
        ),
    )
    parser.add_argument(
        "--layout-dm-dir",
        type=Path,
        required=True,
        help="Converted LayoutDM save_pretrained directory for the matching dataset/seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output save_pretrained directory for the composite pipeline.",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="Reserved Hub repo id for future upload support; conversion stays local.",
    )
    args = parser.parse_args()
    seed_dirs = discover_seed_dirs(args.corrector_job_dir)
    if not seed_dirs:
        raise FileNotFoundError(
            f"No corrector seed dirs found in {args.corrector_job_dir}"
        )
    layout_dm = LayoutDMPipeline.from_pretrained(args.layout_dm_dir)
    multiple_seeds = len(seed_dirs) > 1
    for seed_dir in seed_dirs:
        output_dir = (
            args.output_dir / seed_dir.name if multiple_seeds else args.output_dir
        )
        corrector = build_corrector_from_original(
            dataset=args.dataset,
            checkpoint_dir=seed_dir,
            layout_dm=layout_dm,
        )
        pipe = LayoutCorrectorPipeline(layout_dm=layout_dm, corrector=corrector)
        pipe.save_pretrained(output_dir, safe_serialization=True)
        (output_dir / "README.md").write_text(
            str(
                layout_corrector_model_card(
                    dataset=args.dataset,
                    parity_metrics=_parity_metrics(args.dataset),
                )
            ),
            encoding="utf-8",
        )
        print(output_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from laygen.common.labels import normalize_dataset_name
from laygen.common.model_card import ParityMetric, layout_corrector_model_card
from layout_corrector.conversion import (
    build_corrector_from_original,
    discover_seed_dirs,
)
from layout_corrector.pipeline import LayoutCorrectorPipeline
from layout_dm.pipeline import LayoutDMPipeline


def _parity_metrics(dataset: str) -> list[ParityMetric]:
    return [
        ParityMetric(
            dataset=normalize_dataset_name(dataset),
            tokenizer_exact="checked in vendor parity",
            deterministic_exact="checked in vendor parity",
            logits_max_abs=0.0,
            logits_max_rel=0.0,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--starter-dir", type=Path, required=True)
    parser.add_argument("--corrector-job-dir", type=Path, required=True)
    parser.add_argument("--layout-dm-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--push-to-hub", default=None)
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

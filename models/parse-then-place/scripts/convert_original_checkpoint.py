"""Convert original Parse-Then-Place checkpoints into a composite directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import cast

import torch
from transformers import (
    AutoModelForSeq2SeqLM,  # ty: ignore[possibly-missing-import]
    AutoTokenizer,  # ty: ignore[possibly-missing-import]
    PreTrainedTokenizerBase,
    T5ForConditionalGeneration,  # ty: ignore[possibly-missing-import]
    T5Tokenizer,  # ty: ignore[possibly-missing-import]
)

from parse_then_place import ParseThenPlaceConfig, ParseThenPlaceProcessor


def convert_original_checkpoint(
    *,
    original_root: Path,
    output_dir: Path,
    dataset_name: str,
    stage2_mode: str,
    parser_base_model: str = "google/t5-v1_1-base",
    parser_state_path: Path | None = None,
) -> None:
    """Convert a vendor asset tree into a lightweight composite checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ParseThenPlaceConfig(
        dataset_name=dataset_name,
        stage2_mode=stage2_mode,
        parser_model_name=parser_base_model,
    )
    parser_output = output_dir / config.parser_subfolder
    placement_output = output_dir / config.placement_subfolder
    parser_output.mkdir(parents=True, exist_ok=True)
    placement_output.mkdir(parents=True, exist_ok=True)

    parser_tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(parser_base_model),
    )
    parser_tokenizer.save_pretrained(parser_output)
    if parser_state_path is not None:
        parser = AutoModelForSeq2SeqLM.from_pretrained(parser_base_model)
        state = torch.load(parser_state_path, map_location="cpu")
        parser.load_state_dict(state, strict=False)
        parser.save_pretrained(parser_output)
    else:
        shutil.rmtree(parser_output)

    stage2_root = original_root / "ckpt" / dataset_name / "stage2" / stage2_mode
    placement = T5ForConditionalGeneration.from_pretrained(stage2_root)
    placement.save_pretrained(placement_output)
    placement_tokenizer = cast(
        PreTrainedTokenizerBase,
        T5Tokenizer.from_pretrained(stage2_root),
    )
    placement_tokenizer.save_pretrained(placement_output)

    config.save_pretrained(output_dir)
    config_id2label = cast(dict[int | str, str], config.id2label or {})
    processor = ParseThenPlaceProcessor(
        parser_tokenizer=parser_tokenizer,
        placement_tokenizer=placement_tokenizer,
        dataset_name=dataset_name,
        canvas_size=cast(tuple[int, int], config.canvas_size),
        id2label={int(key): str(value) for key, value in config_id2label.items()},
    )
    processor.save_pretrained(output_dir)


def main() -> None:
    """Parse CLI arguments and run conversion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=["rico", "web"], required=True)
    parser.add_argument(
        "--stage2-mode", choices=["pretrain", "finetune"], required=True
    )
    parser.add_argument("--parser-base-model", default="google/t5-v1_1-base")
    parser.add_argument("--parser-state-path", type=Path)
    args = parser.parse_args()
    convert_original_checkpoint(
        original_root=args.original_root,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        stage2_mode=args.stage2_mode,
        parser_base_model=args.parser_base_model,
        parser_state_path=args.parser_state_path,
    )


if __name__ == "__main__":
    main()

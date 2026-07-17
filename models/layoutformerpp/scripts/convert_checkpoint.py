"""Convert a LayoutFormer++ PyTorch checkpoint to `save_pretrained` format."""

from __future__ import annotations

import argparse
from pathlib import Path

from layoutformerpp import (
    LayoutFormerPPConfig,
    LayoutFormerPPForConditionalGeneration,
    LayoutFormerPPProcessor,
    LayoutFormerPPTokenizer,
)
from layoutformerpp.conversion import load_original_state_dict
from layoutformerpp.serialization import build_default_tokens
from layout_generation_common.labels import labels_for_dataset


def main() -> None:
    """Run checkpoint conversion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=["rico", "publaynet"], required=True)
    parser.add_argument(
        "--task",
        choices=["gen_t", "gen_ts", "gen_r", "refinement", "completion", "ugen"],
        required=True,
    )
    parser.add_argument("--vocab-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.vocab_json is not None:
        tokenizer = LayoutFormerPPTokenizer(vocab_file=str(args.vocab_json))
    else:
        tokenizer = LayoutFormerPPTokenizer(
            tokens=build_default_tokens(
                labels_for_dataset(args.dataset), task=args.task, grid=128
            )
        )
    config = LayoutFormerPPConfig(
        vocab_size=tokenizer.vocab_size, dataset=args.dataset, task=args.task
    )
    model = LayoutFormerPPForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(
        load_original_state_dict(args.checkpoint), strict=False
    )
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    if missing:
        print(f"Missing keys: {missing}")
    processor = LayoutFormerPPProcessor(
        tokenizer=tokenizer, dataset=args.dataset, task=args.task
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

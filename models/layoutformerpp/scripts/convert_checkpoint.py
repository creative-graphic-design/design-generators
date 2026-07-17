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
from layoutformerpp.conversion import (
    load_original_state_dict,
    write_layoutformerpp_model_card,
)
from layoutformerpp.serialization import build_default_tokens
from layoutformerpp.tasks import (
    LayoutFormerPPTask,
    SUPPORTED_DATASETS,
    normalize_layoutformerpp_dataset,
)
from laygen.common.labels import labels_for_dataset


def resolve_vocab_json(vocab_json: Path | None, checkpoint: Path) -> Path | None:
    """Return an explicit or sibling vendor vocab file when one is published."""
    if vocab_json is not None:
        return vocab_json
    sibling = checkpoint.parent / "vocab.json"
    return sibling if sibling.exists() else None


def main() -> None:
    """Run checkpoint conversion."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a LayoutFormer++ PyTorch checkpoint into Transformers "
            "`save_pretrained` artifacts plus a Hub README model card."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the original LayoutFormer++ .pth.tar checkpoint. Required.",
    )
    parser.add_argument(
        "--dataset",
        type=normalize_layoutformerpp_dataset,
        choices=sorted(SUPPORTED_DATASETS, key=str),
        required=True,
        help="Dataset for the checkpoint. Required.",
    )
    parser.add_argument(
        "--task",
        type=LayoutFormerPPTask,
        choices=tuple(LayoutFormerPPTask),
        required=True,
        help="Task-specific checkpoint type. Required.",
    )
    parser.add_argument(
        "--vocab-json",
        type=Path,
        help=(
            "Optional original vocab.json path. When omitted, a default tokenizer "
            "vocabulary is generated from dataset labels and the selected task. "
            "Default: omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where converted artifacts are written. Required.",
    )
    args = parser.parse_args()

    vocab_json = resolve_vocab_json(args.vocab_json, args.checkpoint)
    if vocab_json is not None:
        tokenizer = LayoutFormerPPTokenizer(vocab_file=str(vocab_json))
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
    write_layoutformerpp_model_card(
        args.output_dir, dataset=args.dataset, task=args.task
    )


if __name__ == "__main__":
    main()

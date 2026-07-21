"""Convert a Flex-DM TensorFlow checkpoint to a local Transformers checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flex_dm import FlexDmForMaskedDocumentModeling, FlexDmProcessor
from flex_dm.conversion import conversion_report, map_tensor_by_rule
from flex_dm.tf_checkpoint import load_tf_checkpoint_variables


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["crello", "rico"], required=True)
    parser.add_argument("--variant", default="ours-exp-ft")
    parser.add_argument(
        "--asset-dir", type=Path, default=Path(".cache/flex-dm/original")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocabulary-json", type=Path)
    parser.add_argument("--checkpoint-name", default="best.ckpt")
    return parser.parse_args()


def resolve_variant_root(asset_dir: Path, dataset: str, variant: str) -> Path:
    """Return the extracted checkpoint variant directory."""
    direct = asset_dir / "weights" / dataset / variant
    nested = asset_dir / "weights" / dataset / dataset / variant
    return direct if direct.exists() else nested


def resolve_vocabulary_path(asset_dir: Path, dataset: str) -> Path:
    """Return the extracted vendor vocabulary path."""
    direct = asset_dir / "data" / dataset / "vocabulary.json"
    nested = asset_dir / "data" / dataset / dataset / "vocabulary.json"
    return direct if direct.exists() else nested


def main() -> None:
    """Convert variables by semantic mapping and save model/processor."""
    args = parse_args()
    vocabulary_path = args.vocabulary_json or resolve_vocabulary_path(
        args.asset_dir, args.dataset
    )
    vocabulary = (
        json.loads(vocabulary_path.read_text()) if vocabulary_path.exists() else {}
    )
    processor = FlexDmProcessor.from_vocabulary(
        dataset_name=args.dataset,
        vocabulary=vocabulary,
        checkpoint_variant=args.variant,
    )
    checkpoint_root = resolve_variant_root(args.asset_dir, args.dataset, args.variant)
    args_path = checkpoint_root / "args.json"
    if args_path.exists():
        original_args = json.loads(args_path.read_text())
        processor.config.original_args = original_args
        for key in (
            "latent_dim",
            "num_blocks",
            "block_type",
            "masking_method",
            "seq_type",
            "arch_type",
            "context",
            "input_dtype",
        ):
            if key in original_args:
                setattr(processor.config, key, original_args[key])
    model = FlexDmForMaskedDocumentModeling(processor.config)
    checkpoint = checkpoint_root / "checkpoints" / args.checkpoint_name
    variables = load_tf_checkpoint_variables(checkpoint)
    converted: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    model_source_keys: set[str] = set()
    for source_name, value in variables.items():
        if source_name.startswith("model/") and "/.OPTIMIZER_SLOT/" not in source_name:
            model_source_keys.add(source_name)
        mapped = map_tensor_by_rule(source_name, value)
        if mapped is None:
            continue
        target_name, tensor = mapped
        if target_name in model.state_dict():
            converted[target_name] = tensor
            consumed.add(source_name)
    report = conversion_report(
        converted=converted,
        target_keys=set(model.state_dict()),
        source_keys=model_source_keys,
        consumed_source_keys=consumed,
    )
    if report.missing_target_keys or report.unexpected_source_keys:
        raise ValueError(
            "Flex-DM checkpoint conversion is incomplete: "
            f"missing={list(report.missing_target_keys)} "
            f"unexpected={list(report.unexpected_source_keys)}"
        )
    model.load_state_dict(converted, strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.config.conversion_report = {
        "matched_tensor_count": report.matched_tensor_count,
        "matched_parameter_count": report.matched_parameter_count,
        "missing_target_keys": [],
        "unexpected_source_keys": [],
        "strict": True,
    }
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(model.config.conversion_report, indent=2, sort_keys=True)
    )
    print(f"wrote converted checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()

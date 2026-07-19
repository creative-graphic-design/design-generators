"""Convert an original RALF checkpoint to a local Transformers-style directory.

The converter parses the original YAML as plain data, builds the local RALF
module tree, and records strict-load results measured from that local tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch

from ralf import RalfConfig, RalfForConditionalLayoutGeneration, RalfProcessor


def _int_value(value: object, default: int) -> int:
    if value is None:
        return default
    return int(cast(int | str, value))


def _str_sequence(value: object, default: list[str]) -> list[str]:
    if value is None:
        return default
    return [str(item) for item in cast(list[object], value)]


def _int_or_str(value: object, default: int | str) -> int | str:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return str(value)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-dir",
        type=Path,
        required=True,
        help="Original job directory containing config.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Original PyTorch checkpoint path.",
    )
    parser.add_argument(
        "--dataset", choices=["cgl", "pku", "pku_posterlayout"], required=True
    )
    parser.add_argument(
        "--task", required=True, help="Canonical condition task for this checkpoint."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the converted checkpoint.",
    )
    parser.add_argument(
        "--vocabulary-json",
        type=Path,
        default=None,
        help="Dataset vocabulary.json used by the original dataset loader.",
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML to parse original YAML configs") from exc
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not contain a mapping")
    return cast(dict[str, object], data)


def _state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if not isinstance(state_dict, dict):
        raise TypeError(f"{path} did not contain a state_dict mapping")
    return cast(dict[str, torch.Tensor], state_dict)


def _labels_from_vocabulary(
    path: Path | None, dataset: str, label_count: int
) -> dict[int, str]:
    if path is not None:
        with path.open() as f:
            vocabulary = json.load(f)
        names = sorted(vocabulary["label"].keys())
    elif dataset == "pku":
        names = ["logo", "text", "underlay"]
    else:
        raise FileNotFoundError(
            "CGL conversion requires --vocabulary-json so label ids match vendor "
            "ClassLabel(sorted(vocabulary['label'].keys()))"
        )
    if len(names) != label_count:
        raise ValueError(
            f"Vocabulary label count {len(names)} does not match checkpoint {label_count}"
        )
    return dict(enumerate(names))


def _config_from_original(
    *,
    original_config: dict[str, object],
    dataset: str,
    task: str,
    id2label: dict[int, str],
) -> RalfConfig:
    dataset_cfg = cast(dict[str, object], original_config.get("dataset", {}))
    tokenizer_cfg = cast(dict[str, object], original_config.get("tokenizer", {}))
    generator_cfg = cast(dict[str, object], original_config.get("generator", {}))
    return RalfConfig(
        dataset_name=dataset,
        task=task,
        id2label=cast(dict[int | str, str], id2label),
        max_seq_length=_int_value(dataset_cfg.get("max_seq_length"), 10),
        num_bin=_int_value(tokenizer_cfg.get("num_bin"), 128),
        var_order=_str_sequence(
            tokenizer_cfg.get("var_order"),
            ["label", "width", "height", "center_x", "center_y"],
        ),
        special_tokens=_str_sequence(
            tokenizer_cfg.get("special_tokens"), ["pad", "bos", "eos"]
        ),
        is_loc_vocab_shared=bool(tokenizer_cfg.get("is_loc_vocab_shared", False)),
        geo_quantization=str(tokenizer_cfg.get("geo_quantization", "linear")),
        retrieval_backbone=str(generator_cfg.get("retrieval_backbone", "dreamsim")),
        saliency_k=_int_or_str(generator_cfg.get("saliency_k"), "None"),
        top_k=_int_value(generator_cfg.get("top_k"), 16),
        original_config=original_config,
    )


def _strict_load_report(
    model: RalfForConditionalLayoutGeneration,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, object]:
    target_state = model.state_dict()
    shape_mismatches = {
        key: {
            "source": list(value.shape),
            "target": list(target_state[key].shape),
        }
        for key, value in state_dict.items()
        if key in target_state and tuple(value.shape) != tuple(target_state[key].shape)
    }
    try:
        result = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        missing = sorted(set(target_state) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(target_state))
        return {
            "matched_keys": sorted(set(state_dict) & set(target_state)),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "skipped_shape_mismatch_keys": shape_mismatches,
            "strict_load_error": str(exc),
            "weight_parity_ready": False,
        }
    return {
        "matched_keys": sorted(set(state_dict) & set(target_state)),
        "missing_keys": sorted(result.missing_keys),
        "unexpected_keys": sorted(result.unexpected_keys),
        "skipped_shape_mismatch_keys": shape_mismatches,
        "strict_load_error": None,
        "weight_parity_ready": not result.missing_keys
        and not result.unexpected_keys
        and not shape_mismatches,
    }


def main() -> None:
    """Convert checkpoint metadata and save a local model directory."""
    args = parse_args()
    original_config = _read_yaml(args.job_dir / "config.yaml")
    state_dict = _state_dict_from_checkpoint(args.checkpoint)
    label_count = int(state_dict["layout_encoer.emb_label.weight"].shape[0])
    vocabulary_path = args.vocabulary_json
    if vocabulary_path is None:
        candidate = (
            args.job_dir.parents[1] / "dataset" / args.dataset / "vocabulary.json"
        )
        if candidate.exists():
            vocabulary_path = candidate
    id2label = _labels_from_vocabulary(vocabulary_path, args.dataset, label_count)
    config = _config_from_original(
        original_config=original_config,
        dataset=args.dataset,
        task=args.task,
        id2label=id2label,
    )
    model = RalfForConditionalLayoutGeneration(config)
    report = _strict_load_report(model, state_dict)
    if not report["weight_parity_ready"]:
        raise RuntimeError(f"Strict local RALF load failed: {report}")
    processor = RalfProcessor.from_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    report = {
        "checkpoint": str(args.checkpoint),
        "job_dir": str(args.job_dir),
        "vocabulary_json": None if vocabulary_path is None else str(vocabulary_path),
        "source_key_count": len(state_dict),
        "target_key_count": len(model.state_dict()),
        **report,
    }
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()

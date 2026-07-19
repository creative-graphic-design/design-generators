"""Convert an original RALF checkpoint to a local Transformers-style directory.

The converter reads the original `config.yaml` as plain YAML and does not use
Hydra or OmegaConf. Vendor dependency loading is intentionally avoided here;
unmatched keys are reported so parity work can refine the mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ralf import RalfConfig, RalfForConditionalLayoutGeneration, RalfProcessor


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
        "--dataset",
        choices=["cgl", "pku_posterlayout"],
        required=True,
        help="Dataset name for the converted config.",
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
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Install ralf[vendor] to parse original YAML configs"
        ) from exc
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not contain a mapping")
    return data


def main() -> None:
    """Convert checkpoint metadata and save a local model directory."""
    args = parse_args()
    original_config = _read_yaml(args.job_dir / "config.yaml")
    config = RalfConfig(
        dataset_name=args.dataset,
        task=args.task,
        original_hydra_config=original_config,
    )
    model = RalfForConditionalLayoutGeneration(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    target_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    shape_mismatches = {
        key: {
            "source": list(value.shape),
            "target": list(target_state[key].shape),
        }
        for key, value in state_dict.items()
        if key in target_state and tuple(value.shape) != tuple(target_state[key].shape)
    }
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    processor = RalfProcessor.from_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    report = {
        "checkpoint": str(args.checkpoint),
        "job_dir": str(args.job_dir),
        "matched_keys": sorted(compatible_state),
        "missing_keys": list(missing),
        "skipped_shape_mismatch_keys": shape_mismatches,
        "source_key_count": len(state_dict),
        "target_key_count": len(target_state),
        "unexpected_keys": list(unexpected),
        "weight_parity_ready": len(compatible_state) == len(target_state),
    }
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()

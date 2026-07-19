"""Convert an original RALF checkpoint to a local Transformers-style directory.

The converter reads the original `config.yaml` as plain YAML and does not use
Hydra or OmegaConf in port code. It instantiates the vendor module tree only
when strict checkpoint loading is requested through `use_vendor_modules=True`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import cast

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
        "--vendor-cache-dir",
        type=Path,
        default=Path(".cache/ralf/cache"),
        help="Unpacked authors' cache directory used by vendor-compatible modules.",
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
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    label_count = int(state_dict["layout_encoer.emb_label.weight"].shape[0])
    id2label = _vendor_id2label(args.dataset, label_count)
    config = RalfConfig(
        dataset_name=args.dataset,
        task=args.task,
        id2label=cast(dict[int | str, str], id2label),
        original_hydra_config=original_config,
        use_vendor_modules=True,
        vendor_cache_dir=str(args.vendor_cache_dir),
    )
    model = RalfForConditionalLayoutGeneration(config)
    target_state = model.state_dict()
    model.load_state_dict(state_dict, strict=True)
    processor = RalfProcessor.from_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    report = {
        "checkpoint": str(args.checkpoint),
        "job_dir": str(args.job_dir),
        "matched_keys": sorted(state_dict),
        "missing_keys": [],
        "skipped_shape_mismatch_keys": {},
        "source_key_count": len(state_dict),
        "target_key_count": len(target_state),
        "unexpected_keys": [],
        "weight_parity_ready": True,
    }
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )


def _vendor_id2label(dataset: str, label_count: int) -> dict[int, str]:
    if dataset == "cgl":
        names = ["embellishment", "logo", "text", "underlay"]
    else:
        names = ["logo", "text", "underlay"]
    if label_count != len(names):
        names = [f"label_{idx}" for idx in range(label_count)]
    return dict(enumerate(names))


if __name__ == "__main__":
    main()

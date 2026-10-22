"""Prepare a selected package checkpoint for the authors' evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast

from jaxtyping import Shaped
import torch
from omegaconf import DictConfig, OmegaConf


class LightningCheckpoint(TypedDict, total=False):
    state_dict: dict[str, Shaped[torch.Tensor, "..."]]


def package_state_for_original(
    checkpoint: LightningCheckpoint,
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    """Map Lightning's package state keys to the vendor model's state keys."""
    if not isinstance(checkpoint, dict):
        raise TypeError("Lightning checkpoint must be a mapping")

    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise TypeError("Lightning checkpoint has no state_dict mapping")

    prefix = "model.model."
    converted: dict[str, Shaped[torch.Tensor, "..."]] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Package state entry {key!r} is not a tensor")
        vendor_key = f"model.module.{key.removeprefix(prefix)}"
        if vendor_key in converted:
            raise ValueError(f"Duplicate converted state key: {vendor_key}")
        converted[vendor_key] = value.detach().cpu()

    if not converted:
        raise ValueError("Lightning checkpoint has no model.model.* state entries")

    return converted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    output_path = args.output_dir / "best_model.pt"
    config_path = args.output_dir / "config.yaml"
    if output_path.exists() or config_path.exists():
        raise FileExistsError(f"Evaluation job already exists: {args.output_dir}")

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vendor_state = package_state_for_original(raw)
    config = cast(DictConfig, OmegaConf.load(args.template_config))
    config.seed = args.seed
    config.job_dir = str(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(vendor_state, output_path)
    OmegaConf.save(config, config_path)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": _sha256(args.checkpoint),
                "vendor_checkpoint": str(output_path),
                "vendor_checkpoint_sha256": _sha256(output_path),
                "config": str(config_path),
                "tensor_count": len(vendor_state),
                "training_seed": args.seed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

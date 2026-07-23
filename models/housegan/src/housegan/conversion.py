"""Checkpoint conversion utilities for House-GAN."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .configuration_housegan import HouseGanConfig
from .modeling_housegan import HouseGanGenerator
from .pipeline_housegan import HouseGanPipeline
from .processing_housegan import HouseGanProcessor
from .vendor_state_dict import convert_state_dict


def sha256_file(path: str | Path) -> str:
    """Compute a SHA256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_original_checkpoint(
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    target_set: str = "D",
    checkpoint_step: int = 200000,
) -> dict[str, object]:
    """Convert a raw House-GAN generator state dict into HF files."""
    checkpoint_path = Path(checkpoint)
    raw_state = torch.load(checkpoint_path, map_location="cpu")
    converted, report = convert_state_dict(raw_state)
    config = HouseGanConfig(
        target_set=target_set,
        checkpoint_step=checkpoint_step,
        source_checkpoint=checkpoint_path.name,
        conversion_report={
            **report.to_dict(),
            "source_sha256": sha256_file(checkpoint_path),
        },
    )
    model = HouseGanGenerator(config)
    load_result = model.load_state_dict(converted, strict=True)
    measured = {
        **config.conversion_report,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
    model.config.conversion_report = measured
    output_path = Path(output_dir)
    processor = HouseGanProcessor(config=config)
    HouseGanPipeline(model=model, processor=processor, config=config).save_pretrained(
        output_path
    )
    (output_path / "conversion_report.json").write_text(
        json.dumps(measured, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return measured

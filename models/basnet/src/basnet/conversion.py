"""Checkpoint conversion helpers for BASNet."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Shaped

from .configuration_basnet import BASNetConfig
from .modeling_basnet import BASNetModel

CONVERSION_REPORT: Final[str] = "conversion_report.json"


def strip_module_prefix(
    state_dict: Mapping[str, Shaped[torch.Tensor, "..."]],
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    """Remove ``DataParallel`` ``module.`` prefixes.

    Args:
        state_dict: Raw PyTorch state dict.

    Returns:
        State dict with prefixes removed.

    Examples:
        >>> strip_module_prefix({"module.a": torch.tensor(1)})["a"].item()
        1
    """
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def file_sha256(path: Path) -> str:
    """Compute SHA256 for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_original_checkpoint(
    *,
    checkpoint: Path,
    output_dir: Path,
    config: BASNetConfig,
) -> dict[str, str | int | list[str]]:
    """Convert a raw BASNet checkpoint into a ``save_pretrained`` directory.

    Args:
        checkpoint: Raw checkpoint path.
        output_dir: Output model directory.
        config: BASNet config.

    Returns:
        Conversion report dictionary.

    Raises:
        RuntimeError: If converted keys do not strictly match the target model.
    """
    model = BASNetModel(config)
    state = strip_module_prefix(torch.load(checkpoint, map_location="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    report: dict[str, str | int | list[str]] = {
        "checkpoint": str(checkpoint),
        "sha256": file_sha256(checkpoint),
        "source_key_count": len(state),
        "missing_keys": [str(key) for key in missing],
        "unexpected_keys": [str(key) for key in unexpected],
    }
    if missing or unexpected:
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    config.conversion_report = dict[str, str | int | float | bool | list[str]](report)
    model.config = config
    model.save_pretrained(output_dir)
    (output_dir / CONVERSION_REPORT).write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report

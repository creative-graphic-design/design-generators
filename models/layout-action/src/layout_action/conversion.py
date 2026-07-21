"""Checkpoint conversion helpers for LayoutAction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .configuration_layout_action import LayoutActionConfig
from .modeling_layout_action import LayoutActionForCausalLM
from .processing_layout_action import LayoutActionProcessor
from .tokenization_layout_action import LayoutActionTokenizer


@dataclass(frozen=True)
class StateDictKeyReport:
    """One source-to-target state-dict mapping result."""

    source_key: str
    target_key: str
    source_shape: tuple[int, ...]
    loaded: bool


def remap_layout_action_key(key: str) -> str:
    """Map a vendor LayoutAction state-dict key to this package."""
    return key


def remap_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: LayoutActionForCausalLM,
) -> tuple[dict[str, torch.Tensor], list[StateDictKeyReport]]:
    """Remap and report checkpoint key coverage."""
    target_keys = set(model.state_dict())
    remapped: dict[str, torch.Tensor] = {}
    report: list[StateDictKeyReport] = []
    for source_key, value in state_dict.items():
        target_key = remap_layout_action_key(source_key)
        loaded = target_key in target_keys
        if loaded:
            remapped[target_key] = value
        report.append(
            StateDictKeyReport(
                source_key=source_key,
                target_key=target_key,
                source_shape=tuple(value.shape),
                loaded=loaded,
            )
        )
    return remapped, report


def convert_layout_action_checkpoint(
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    config: LayoutActionConfig,
    strict: bool = True,
) -> dict[str, object]:
    """Convert a raw vendor ``.pth`` checkpoint to HF-style files.

    Args:
        checkpoint: Raw PyTorch state-dict path.
        output_dir: Destination checkpoint directory.
        config: LayoutAction config built from dataset metadata.
        strict: Whether model loading is strict.

    Returns:
        Conversion report dictionary.
    """
    model = LayoutActionForCausalLM(config)
    raw = torch.load(checkpoint, map_location="cpu")
    if not isinstance(raw, dict):
        raise TypeError("LayoutAction checkpoint must be a state-dict mapping")
    remapped, report = remap_state_dict(raw, model)
    missing, unexpected = model.load_state_dict(remapped, strict=strict)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processor = LayoutActionProcessor(LayoutActionTokenizer(config))
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    conversion_report = {
        "checkpoint": str(checkpoint),
        "config": config.to_dict(),
        "keys": [asdict(row) for row in report],
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    return conversion_report

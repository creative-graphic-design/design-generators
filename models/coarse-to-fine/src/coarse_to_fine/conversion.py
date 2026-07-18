"""Checkpoint conversion helpers for Coarse-to-Fine."""

from __future__ import annotations

from pathlib import Path

import torch

from laygen.common.labels import DatasetName

from .configuration_coarse_to_fine import CoarseToFineConfig
from .modeling_coarse_to_fine import CoarseToFineForLayoutGeneration
from .processing_coarse_to_fine import CoarseToFineProcessor


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove optional DDP ``module.`` prefixes from checkpoint keys."""
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def convert_checkpoint(
    checkpoint: str | Path,
    *,
    dataset: DatasetName | str,
    output_dir: str | Path,
) -> CoarseToFineForLayoutGeneration:
    """Convert a raw vendor state dict into ``save_pretrained`` format.

    Args:
        checkpoint: Path to ``checkpoint.pth.tar``.
        dataset: Dataset for config defaults.
        output_dir: Destination directory.

    Returns:
        The loaded model.
    """
    config = CoarseToFineConfig(dataset=dataset)
    model = CoarseToFineForLayoutGeneration(config)
    raw_state = torch.load(checkpoint, map_location="cpu")
    state = strip_module_prefix(raw_state)
    model.load_state_dict(state, strict=True)
    output_path = Path(output_dir)
    model.save_pretrained(output_path, safe_serialization=True)
    id2label = {int(key): str(value) for key, value in (config.id2label or {}).items()}
    processor = CoarseToFineProcessor.from_config(
        dataset=config.dataset,
        x_grid=config.discrete_x_grid,
        y_grid=config.discrete_y_grid,
        max_num_elements=config.max_num_elements,
        id2label=id2label,
    )
    processor.save_pretrained(output_path)
    return model

"""Narrow RALF checkpoint and output adapter for the pinned evaluator path."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Mapping, Sequence, TypedDict, TypeAlias, cast

import torch

from laygen.common.testing import (
    load_torch_checkpoint_state_dict,
    strip_torch_state_dict_prefix,
)
from laygen.modeling_outputs import LayoutGenerationOutput


VendorConfigValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["VendorConfigValue"]
    | dict[str, "VendorConfigValue"]
)


class VendorSample(TypedDict):
    """One generated layout row accepted by the pinned evaluator."""

    label: list[int]
    center_x: list[float]
    center_y: list[float]
    width: list[float]
    height: list[float]
    id: str | int


@dataclass(frozen=True)
class VendorCheckpointBundle:
    """Files that form one vendor inference job directory."""

    job_dir: Path
    checkpoint_path: Path
    config_path: Path


def materialize_vendor_checkpoint(
    lightning_checkpoint: str | PathLike[str],
    job_dir: str | PathLike[str],
    *,
    train_config: Mapping[str, VendorConfigValue],
) -> VendorCheckpointBundle:
    """Expose a Lightning checkpoint as a raw checkpoint plus vendor config.

    The pinned inference entrypoint loads ``config.yaml`` beside ``*pt`` files
    and expects a raw model state dictionary. Lightning stores the model under
    ``state_dict`` with a ``model.`` wrapper, so this adapter removes exactly
    that wrapper and leaves the vendor model/configuration path unchanged.
    """
    normalized_job_dir = Path(job_dir)
    normalized_job_dir.mkdir(parents=True, exist_ok=True)
    lightning_state = load_torch_checkpoint_state_dict(
        lightning_checkpoint,
        state_dict_key="state_dict",
        map_location="cpu",
        weights_only=False,
    )
    if not lightning_state:
        raise ValueError("Lightning checkpoint has an empty state_dict")
    if any(not key.startswith("model.") for key in lightning_state):
        raise ValueError("Lightning state_dict contains a key without model. prefix")
    raw_state = strip_torch_state_dict_prefix(
        lightning_state,
        strip_prefix="model.",
        include_prefix="model.",
    )
    if len(raw_state) != len(lightning_state):
        raise ValueError("Lightning state_dict prefix normalization changed coverage")

    checkpoint_path = normalized_job_dir / "gen_final_model.pt"
    torch.save(dict(raw_state), checkpoint_path)

    import yaml

    config_path = normalized_job_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(dict(train_config), sort_keys=False), encoding="utf-8"
    )
    return VendorCheckpointBundle(
        job_dir=normalized_job_dir,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )


def layout_output_to_vendor_samples(
    output: LayoutGenerationOutput,
    *,
    sample_ids: Sequence[str | int],
) -> list[VendorSample]:
    """Map common package layouts to the vendor pickle geometry schema."""
    bbox = cast(torch.Tensor, output.bbox)
    labels = cast(torch.Tensor, output.labels)
    mask = cast(torch.Tensor, output.mask)
    if bbox.ndim != 3 or bbox.shape[-1] != 4:
        raise ValueError(f"expected bbox shape (batch, elements, 4), got {bbox.shape}")
    if labels.ndim != 2 or mask.ndim != 2:
        raise ValueError("expected labels and mask shapes (batch, elements)")
    if bbox.shape[:2] != labels.shape or labels.shape != mask.shape:
        raise ValueError("bbox, labels, and mask batch shapes differ")
    if len(sample_ids) != bbox.shape[0]:
        raise ValueError("sample_ids must contain one id per output batch item")

    samples: list[VendorSample] = []
    for batch_index, sample_id in enumerate(sample_ids):
        valid = mask[batch_index].bool()
        valid_bbox = bbox[batch_index][valid].detach().cpu().tolist()
        valid_labels = labels[batch_index][valid].detach().cpu().tolist()
        samples.append(
            {
                "label": cast(list[int], valid_labels),
                "center_x": [float(row[0]) for row in valid_bbox],
                "center_y": [float(row[1]) for row in valid_bbox],
                "width": [float(row[2]) for row in valid_bbox],
                "height": [float(row[3]) for row in valid_bbox],
                "id": sample_id,
            }
        )
    return samples


__all__ = [
    "VendorCheckpointBundle",
    "VendorConfigValue",
    "VendorSample",
    "layout_output_to_vendor_samples",
    "materialize_vendor_checkpoint",
]

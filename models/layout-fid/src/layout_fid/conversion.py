"""Conversion helpers for layout FID checkpoints and statistics."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import torch

from laygen.common.labels import id2label_for_dataset

from .configuration_layout_fid import LayoutFIDConfig
from .evaluation import LayoutFIDStatistics, save_reference_statistics
from .modeling_layout_fid import LayoutFIDModel
from .processing_layout_fid import LayoutFIDProcessor


class LayoutFlowDatasetSpec(TypedDict):
    """Conversion metadata for one LayoutFlow dataset."""

    num_public_labels: int
    num_label_embeddings: int
    max_length: int
    stats_suffix: str


LAYOUTFLOW_DATASET_SPECS: Final[dict[str, LayoutFlowDatasetSpec]] = {
    "rico25": {
        "num_public_labels": 25,
        "num_label_embeddings": 26,
        "max_length": 20,
        "stats_suffix": "rico",
    },
    "publaynet": {
        "num_public_labels": 5,
        "num_label_embeddings": 6,
        "max_length": 20,
        "stats_suffix": "publaynet",
    },
}


def convert_layoutflow_checkpoint(
    *,
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    dataset_name: str,
    stats_paths: Mapping[str, str | PathLike[str]] | None = None,
) -> LayoutFIDConfig:
    """Convert a LayoutFlow-style LayoutNet checkpoint directory."""
    spec = _layoutflow_spec(dataset_name)
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    state_dict = strip_module_prefix(state_dict)
    config = LayoutFIDConfig(
        dataset_name=dataset_name,
        id2label=id2label_for_dataset(dataset_name),
        architecture="layoutnet",
        source="layoutflow",
        num_public_labels=int(spec["num_public_labels"]),
        num_label_embeddings=int(state_dict["emb_label.weight"].shape[0]),
        max_length=int(state_dict["pos_token"].shape[0]),
        bbox_format_for_model="ltrb",
        label_id_offset=0,
        pad_label_id=0,
    )
    validate_state_dict_shapes(state_dict, config)
    model = LayoutFIDModel(config)
    model.load_state_dict(state_dict)
    output = Path(output_dir)
    model.save_pretrained(output, safe_serialization=True)
    LayoutFIDProcessor(config).save_pretrained(output)
    if stats_paths:
        for split, path in stats_paths.items():
            stats = load_musig_statistics(
                path,
                split=split,
                dataset_name=config.dataset_name,
                source=config.source,
            )
            save_reference_statistics(output / f"reference_stats/{split}.npz", stats)
    return config


def convert_layoutdm_fidnet_v3_checkpoint(
    *,
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    dataset_name: str,
    num_public_labels: int,
    max_length: int,
) -> LayoutFIDConfig:
    """Convert a LayoutDM FIDNetV3 checkpoint when assets are available."""
    state_dict = load_checkpoint_state_dict(
        checkpoint_path, state_dict_key="state_dict"
    )
    config = LayoutFIDConfig(
        dataset_name=dataset_name,
        id2label=id2label_for_dataset(dataset_name),
        architecture="fidnet_v3",
        source="layoutdm",
        num_public_labels=num_public_labels,
        num_label_embeddings=int(state_dict["emb_label.weight"].shape[0]),
        max_length=max_length,
        bbox_format_for_model="xywh",
        label_id_offset=0,
        pad_label_id=0,
    )
    validate_state_dict_shapes(state_dict, config)
    model = LayoutFIDModel(config)
    model.load_state_dict(state_dict)
    output = Path(output_dir)
    model.save_pretrained(output, safe_serialization=True)
    LayoutFIDProcessor(config).save_pretrained(output)
    return config


def load_musig_statistics(
    path: str | PathLike[str],
    *,
    split: str,
    dataset_name: str,
    source: str,
) -> LayoutFIDStatistics:
    """Convert a stacked ``[mu; sigma]`` tensor into typed statistics."""
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    array = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    return LayoutFIDStatistics(
        mu=array[0],
        sigma=array[1:],
        split=split,
        dataset_name=dataset_name,
        source=source,
        feature_dim=array.shape[1],
        num_samples=None,
    )


def load_checkpoint_state_dict(
    path: str | PathLike[str],
    *,
    state_dict_key: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load a torch checkpoint state dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if state_dict_key is not None:
        checkpoint = checkpoint[state_dict_key]
    return {str(key): value for key, value in checkpoint.items()}


def strip_module_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strip optional ``module.`` prefixes from checkpoint keys."""
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def validate_state_dict_shapes(
    state_dict: Mapping[str, torch.Tensor], config: LayoutFIDConfig
) -> None:
    """Validate checkpoint tensor shapes before writing artifacts."""
    expected = {
        "emb_label.weight": (config.num_label_embeddings, config.d_model),
        "fc_bbox.weight": (config.d_model, 4),
        "enc_fc_in.weight": (config.d_model, config.d_model * 2),
        "fc_out_cls.weight": (config.num_label_embeddings, config.d_model),
        "fc_out_bbox.weight": (4, config.d_model),
        "pos_token": (config.max_length, 1, config.d_model),
    }
    missing = sorted(set(expected) - set(state_dict))
    if missing:
        raise ValueError(f"checkpoint is missing expected keys: {missing}")
    mismatched = {
        key: (tuple(state_dict[key].shape), shape)
        for key, shape in expected.items()
        if tuple(state_dict[key].shape) != shape
    }
    if mismatched:
        raise ValueError(f"checkpoint tensor shapes do not match config: {mismatched}")


def _layoutflow_spec(dataset_name: str) -> LayoutFlowDatasetSpec:
    try:
        return LAYOUTFLOW_DATASET_SPECS[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported LayoutFlow dataset_name: {dataset_name}"
        ) from exc

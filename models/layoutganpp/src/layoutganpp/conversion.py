"""Conversion helpers for original LayoutGAN++ checkpoint metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from laygen.common.labels import max_elements_for_dataset

from .configuration_layoutganpp import LayoutGANPPConfig
from .datasets import id2label_for_dataset, normalize_dataset_name


def config_from_checkpoint_args(args: object) -> LayoutGANPPConfig:
    """Build a config from original LayoutGAN++ checkpoint arguments.

    Args:
        args: Mapping or argparse-style namespace with upstream checkpoint fields.

    Returns:
        A `LayoutGANPPConfig` populated from the checkpoint metadata.

    Raises:
        KeyError: If a required upstream field is missing.
        ValueError: If the dataset name is unsupported.

    Examples:
        >>> config_from_checkpoint_args(
        ...     {
        ...         "dataset": "rico",
        ...         "latent_size": 4,
        ...         "G_d_model": 512,
        ...         "G_nhead": 8,
        ...         "G_num_layers": 4,
        ...     }
        ... ).dataset_name
        'rico'
    """
    values = _checkpoint_values(args)
    dataset_name = str(values["dataset"])
    canonical_dataset = normalize_dataset_name(dataset_name)
    id2label = id2label_for_dataset(dataset_name)
    return LayoutGANPPConfig(
        dataset_name=dataset_name,
        latent_size=_required_int(values, "latent_size"),
        num_labels=len(id2label),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        d_model=_required_int(values, "G_d_model"),
        nhead=_required_int(values, "G_nhead"),
        num_layers=_required_int(values, "G_num_layers"),
        max_position_embeddings=max_elements_for_dataset(canonical_dataset),
    )


def _checkpoint_values(args: object) -> Mapping[str, object]:
    if isinstance(args, Mapping):
        return cast(Mapping[str, object], args)
    return vars(args)


def _required_int(values: Mapping[str, object], key: str) -> int:
    value = values[key]
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"{key} must be an int-compatible checkpoint value")

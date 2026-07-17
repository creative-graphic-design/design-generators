"""Conversion helpers for original LayoutGAN++ checkpoint metadata."""

from __future__ import annotations

from collections.abc import Mapping

from .configuration_layoutganpp import LayoutGANPPConfig
from .datasets import dataset_metadata, id2label_for_dataset


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
    metadata = dataset_metadata(dataset_name)
    id2label = id2label_for_dataset(dataset_name)
    return LayoutGANPPConfig(
        dataset_name=dataset_name,
        latent_size=int(values["latent_size"]),
        num_labels=len(id2label),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        d_model=int(values["G_d_model"]),
        nhead=int(values["G_nhead"]),
        num_layers=int(values["G_num_layers"]),
        max_position_embeddings=int(metadata["max_elements"]),
    )


def _checkpoint_values(args: object) -> Mapping[str, object]:
    if isinstance(args, Mapping):
        return args
    return vars(args)

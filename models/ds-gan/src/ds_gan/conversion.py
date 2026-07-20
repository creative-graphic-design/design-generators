"""Conversion helpers for original PosterLayout DS-GAN checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch

from .configuration_ds_gan import DSGANConfig


def convert_vendor_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert vendor DS-GAN generator keys to ``DSGANModel`` keys.

    The released checkpoint was commonly saved from ``torch.nn.DataParallel``;
    this helper strips the leading ``module.`` prefix and keeps all generator
    module names otherwise unchanged.

    Args:
        state_dict: Original checkpoint mapping.

    Returns:
        Converted state dictionary.

    Examples:
        >>> convert_vendor_state_dict({"module.fc1.weight": torch.zeros(1)})["fc1.weight"].shape
        torch.Size([1])
    """
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        name = key.removeprefix("module.")
        name = name.removeprefix("generator.")
        converted[name] = value
    return converted


def config_from_vendor_args(args: object | None = None) -> DSGANConfig:
    """Build a DS-GAN config from vendor args or defaults."""
    if args is None:
        return DSGANConfig()
    values = _values(args)
    max_elem = _int_value(values, "max_elem", 32)
    return DSGANConfig(
        backbone=str(values.get("backbone", "resnet50")),
        max_elem=max_elem,
        in_channels=_int_value(values, "in_channels", 8),
        out_channels=_int_value(values, "out_channels", 32),
        hidden_size=_int_value(values, "hidden_size", max_elem * 8),
        num_layers=_int_value(values, "num_layers", 4),
        output_size=_int_value(values, "output_size", 8),
    )


def _values(args: object) -> Mapping[str, object]:
    if isinstance(args, Mapping):
        return cast(Mapping[str, object], args)
    return vars(args)


def _int_value(values: Mapping[str, object], key: str, default: int) -> int:
    value = values.get(key, default)
    if isinstance(value, int):
        return value
    return int(str(value))

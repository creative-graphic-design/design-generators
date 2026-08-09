"""State-dict validation helpers for original House-GAN checkpoints."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from jaxtyping import Shaped

from .configuration_housegan import HouseGanConversionReport


@dataclass(frozen=True)
class ConversionReport:
    """Measured state-dict conversion metadata."""

    key_count: int
    tensor_shapes: dict[str, tuple[int, ...]]
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()

    def to_dict(self) -> HouseGanConversionReport:
        """Serialize report to JSON-compatible values."""
        return {
            "key_count": self.key_count,
            "tensor_shapes": self.tensor_shapes,
            "missing_keys": self.missing_keys,
            "unexpected_keys": self.unexpected_keys,
        }


EXPECTED_PREFIXES = (
    "l1.",
    "upsample_1.",
    "upsample_2.",
    "cmp_1.",
    "cmp_2.",
    "decoder.",
)


def convert_state_dict(
    source: Mapping[str, Shaped[torch.Tensor, "..."]],
) -> tuple[OrderedDict[str, Shaped[torch.Tensor, "..."]], ConversionReport]:
    """Validate and copy an original raw generator state dict."""
    converted: OrderedDict[str, Shaped[torch.Tensor, "..."]] = OrderedDict()
    for key, tensor in source.items():
        if not key.startswith(EXPECTED_PREFIXES):
            raise KeyError(key)
        converted[key] = tensor
    report = ConversionReport(
        key_count=len(converted),
        tensor_shapes={key: tuple(value.shape) for key, value in converted.items()},
    )
    return converted, report

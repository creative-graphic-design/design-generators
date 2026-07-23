"""State-dict validation helpers for original House-GAN checkpoints."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ConversionReport:
    """Measured state-dict conversion metadata."""

    key_count: int
    tensor_shapes: dict[str, tuple[int, ...]]
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize report to JSON-compatible values."""
        return asdict(self)


EXPECTED_PREFIXES = (
    "l1.",
    "upsample_1.",
    "upsample_2.",
    "cmp_1.",
    "cmp_2.",
    "decoder.",
)


def convert_state_dict(
    source: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], ConversionReport]:
    """Validate and copy an original raw generator state dict."""
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, tensor in source.items():
        if not key.startswith(EXPECTED_PREFIXES):
            raise KeyError(key)
        converted[key] = tensor
    report = ConversionReport(
        key_count=len(converted),
        tensor_shapes={key: tuple(value.shape) for key, value in converted.items()},
    )
    return converted, report

"""Shared field definitions for layout-generation output classes."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Final, NamedTuple

import torch


class OutputFieldSpec(NamedTuple):
    """Field specification used to build matching output dataclasses."""

    name: str
    annotation: object
    default: object


LAYOUT_GENERATION_OUTPUT_FIELDS: Final[tuple[OutputFieldSpec, ...]] = (
    OutputFieldSpec("bbox", torch.Tensor, MISSING),
    OutputFieldSpec("labels", torch.Tensor, None),
    OutputFieldSpec("mask", torch.Tensor, None),
    OutputFieldSpec("id2label", dict[int, str], None),
    OutputFieldSpec("sequences", torch.Tensor | None, None),
    OutputFieldSpec("scores", torch.Tensor | None, None),
    OutputFieldSpec("trajectory", object | None, None),
    OutputFieldSpec("intermediates", object | None, None),
)


def dataclass_fields() -> list[tuple[str, object] | tuple[str, object, object]]:
    """Return ``make_dataclass`` field tuples for layout output classes."""
    fields: list[tuple[str, object] | tuple[str, object, object]] = []
    for name, annotation, default in LAYOUT_GENERATION_OUTPUT_FIELDS:
        if default is MISSING:
            fields.append((name, annotation))
        else:
            fields.append((name, annotation, default))
    return fields

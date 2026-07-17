"""Shared field definitions for layout-generation output classes."""

from __future__ import annotations

from dataclasses import MISSING
from enum import StrEnum, auto
from typing import Final, NamedTuple

import torch


class OutputField(StrEnum):
    """Canonical field names shared by layout-generation outputs."""

    bbox = auto()
    labels = auto()
    mask = auto()
    id2label = auto()
    sequences = auto()
    scores = auto()
    trajectory = auto()
    intermediates = auto()


class OutputFieldSpec(NamedTuple):
    """Field specification used to build matching output dataclasses."""

    name: OutputField
    annotation: object
    default: object


LAYOUT_GENERATION_OUTPUT_FIELDS: Final[tuple[OutputFieldSpec, ...]] = (
    OutputFieldSpec(OutputField.bbox, torch.Tensor, MISSING),
    OutputFieldSpec(OutputField.labels, torch.Tensor, None),
    OutputFieldSpec(OutputField.mask, torch.Tensor, None),
    OutputFieldSpec(OutputField.id2label, dict[int, str], None),
    OutputFieldSpec(OutputField.sequences, torch.Tensor | None, None),
    OutputFieldSpec(OutputField.scores, torch.Tensor | None, None),
    OutputFieldSpec(OutputField.trajectory, object | None, None),
    OutputFieldSpec(OutputField.intermediates, object | None, None),
)


def dataclass_fields() -> list[tuple[str, object] | tuple[str, object, object]]:
    """Return ``make_dataclass`` field tuples for layout output classes."""
    fields: list[tuple[str, object] | tuple[str, object, object]] = []
    for name, annotation, default in LAYOUT_GENERATION_OUTPUT_FIELDS:
        if default is MISSING:
            fields.append((str(name), annotation))
        else:
            fields.append((str(name), annotation, default))
    return fields

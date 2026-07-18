"""Typed record keys used by LayoutPrompter serializers and selectors."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from enum import StrEnum, auto
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict

NumericArray: TypeAlias = (
    NDArray[np.int64] | NDArray[np.int32] | NDArray[np.float32] | NDArray[np.float64]
)


class LayoutRecordKey(StrEnum):
    """Closed key set for dict-like layout records."""

    id = auto()
    labels = auto()
    bboxes = auto()
    discrete_bboxes = auto()
    discrete_gold_bboxes = auto()
    discrete_content_bboxes = auto()
    relations = auto()
    text = auto()
    embedding = auto()


class LayoutRecord(TypedDict, total=False):
    """Structured LayoutPrompter record accepted by prompt and selector code."""

    id: NotRequired[str]
    labels: NDArray[np.int64] | Sequence[int]
    bboxes: NumericArray | Sequence[Sequence[int | float]]
    discrete_bboxes: NotRequired[NumericArray | Sequence[Sequence[int | float]]]
    discrete_gold_bboxes: NumericArray | Sequence[Sequence[int | float]]
    discrete_content_bboxes: NotRequired[NumericArray | Sequence[Sequence[int | float]]]
    relations: NotRequired[NumericArray | Sequence[Sequence[int | float]]]
    text: NotRequired[str]
    embedding: NotRequired[NumericArray | Sequence[Sequence[int | float]]]


LayoutRecordInput: TypeAlias = LayoutRecord | Mapping[str, object]


def record_value(data: LayoutRecordInput, key: LayoutRecordKey) -> object:
    """Return a layout-record value by enum key."""
    return data[key.value]


def optional_record_value(
    data: LayoutRecordInput, key: LayoutRecordKey, default: object
) -> object:
    """Return a layout-record value by enum key, or a default."""
    return data.get(key.value, default)

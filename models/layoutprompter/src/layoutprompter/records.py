"""Typed record keys used by LayoutPrompter serializers and selectors."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from enum import StrEnum, auto
from typing import TYPE_CHECKING, cast

import numpy as np
from jaxtyping import Bool, Float, Int
from typing_extensions import NotRequired, TypedDict


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
    labels: Int[np.ndarray, "elements"] | Sequence[int]
    bboxes: (
        Int[np.ndarray, "..."]
        | Float[np.ndarray, "..."]
        | Sequence[Sequence[int | float]]
    )
    discrete_bboxes: NotRequired[
        Int[np.ndarray, "..."]
        | Float[np.ndarray, "..."]
        | Sequence[Sequence[int | float]]
    ]
    discrete_gold_bboxes: (
        Int[np.ndarray, "..."]
        | Float[np.ndarray, "..."]
        | Sequence[Sequence[int | float]]
    )
    discrete_content_bboxes: NotRequired[
        Int[np.ndarray, "..."]
        | Float[np.ndarray, "..."]
        | Sequence[Sequence[int | float]]
    ]
    relations: NotRequired[
        Int[np.ndarray, "..."]
        | Float[np.ndarray, "..."]
        | Sequence[Sequence[int | float]]
    ]
    text: NotRequired[str]
    embedding: NotRequired[
        Int[np.ndarray, "..."]
        | Float[np.ndarray, "..."]
        | Sequence[Sequence[int | float]]
    ]


LayoutRecordScalar = str | int | float | bool | None
if TYPE_CHECKING:
    LayoutRecordPayload = (
        Int[np.ndarray, "..."]
        | Float[np.ndarray, "..."]
        | Bool[np.ndarray, "..."]
        | Sequence["LayoutRecordPayload"]
    )
else:
    LayoutRecordPayload = object

LayoutRecordValue = LayoutRecordScalar | LayoutRecordPayload
LayoutRecordInput = (
    LayoutRecord | Mapping[str, LayoutRecordValue] | Mapping[str, object]
)


def record_value(data: LayoutRecordInput, key: LayoutRecordKey) -> LayoutRecordValue:
    """Return a layout-record value by enum key."""
    return cast(LayoutRecordValue, data[key.value])


def optional_record_value(
    data: LayoutRecordInput, key: LayoutRecordKey, default: LayoutRecordValue
) -> LayoutRecordValue:
    """Return a layout-record value by enum key, or a default."""
    return cast(LayoutRecordValue, data.get(key.value, default))

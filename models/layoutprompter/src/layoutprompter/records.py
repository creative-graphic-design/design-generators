"""Typed record keys used by LayoutPrompter serializers and selectors."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from enum import StrEnum, auto
from typing import TypeAlias

import numpy as np
from jaxtyping import Float, Int
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


LayoutRecordInput: TypeAlias = LayoutRecord | Mapping[str, object]


def record_value(data: LayoutRecordInput, key: LayoutRecordKey) -> object:
    """Return a layout-record value by enum key."""
    return data[key.value]


def optional_record_value(
    data: LayoutRecordInput, key: LayoutRecordKey, default: object
) -> object:
    """Return a layout-record value by enum key, or a default."""
    return data.get(key.value, default)

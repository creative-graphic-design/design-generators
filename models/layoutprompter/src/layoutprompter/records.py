"""Typed record keys used by LayoutPrompter serializers and selectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum, auto
from typing import NotRequired, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from jaxtyping import Bool, Float, Int
from typing_extensions import TypedDict


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
    labels: npt.ArrayLike | Sequence[int]
    bboxes: npt.ArrayLike | Sequence[Sequence[int | float]]
    discrete_bboxes: NotRequired[npt.ArrayLike | Sequence[Sequence[int | float]]]
    discrete_gold_bboxes: npt.ArrayLike | Sequence[Sequence[int | float]]
    discrete_content_bboxes: NotRequired[
        npt.ArrayLike | Sequence[Sequence[int | float]]
    ]
    relations: NotRequired[npt.ArrayLike | Sequence[Sequence[int | float]]]
    text: NotRequired[str]
    embedding: NotRequired[npt.ArrayLike | Sequence[Sequence[int | float]]]


LayoutRecordScalar: TypeAlias = str | int | float | bool | None
LayoutRecordPayload: TypeAlias = npt.ArrayLike
LayoutRecordInput: TypeAlias = LayoutRecord | Mapping[str, LayoutRecordPayload]


def record_value(
    data: LayoutRecordInput, key: LayoutRecordKey
) -> (
    LayoutRecordScalar
    | Int[np.ndarray, ...]
    | Float[np.ndarray, ...]
    | Bool[np.ndarray, ...]
    | Sequence[int | float | str | bool | None]
    | Sequence[Sequence[int | float | str | bool | None]]
):
    """Return a layout-record value by enum key."""
    return cast(
        LayoutRecordScalar
        | Int[np.ndarray, "..."]
        | Float[np.ndarray, "..."]
        | Bool[np.ndarray, "..."]
        | Sequence[int | float | str | bool | None]
        | Sequence[Sequence[int | float | str | bool | None]],
        data[key.value],
    )


def optional_record_value(
    data: LayoutRecordInput,
    key: LayoutRecordKey,
    default: LayoutRecordScalar
    | Int[np.ndarray, ...]
    | Float[np.ndarray, ...]
    | Bool[np.ndarray, ...]
    | Sequence[int | float | str | bool | None]
    | Sequence[Sequence[int | float | str | bool | None]],
) -> (
    LayoutRecordScalar
    | Int[np.ndarray, ...]
    | Float[np.ndarray, ...]
    | Bool[np.ndarray, ...]
    | Sequence[int | float | str | bool | None]
    | Sequence[Sequence[int | float | str | bool | None]]
):
    """Return a layout-record value by enum key, or a default."""
    return cast(
        LayoutRecordScalar
        | Int[np.ndarray, "..."]
        | Float[np.ndarray, "..."]
        | Bool[np.ndarray, "..."]
        | Sequence[int | float | str | bool | None]
        | Sequence[Sequence[int | float | str | bool | None]],
        data.get(key.value, default),
    )

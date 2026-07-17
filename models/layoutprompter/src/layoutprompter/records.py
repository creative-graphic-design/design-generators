"""Typed record keys used by LayoutPrompter serializers and selectors."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum, auto
from typing import TypeAlias

import torch
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
    labels: torch.Tensor
    bboxes: torch.Tensor
    discrete_bboxes: NotRequired[torch.Tensor]
    discrete_gold_bboxes: torch.Tensor
    discrete_content_bboxes: NotRequired[torch.Tensor]
    relations: NotRequired[torch.Tensor]
    text: NotRequired[str]
    embedding: NotRequired[torch.Tensor]


LayoutRecordInput: TypeAlias = LayoutRecord | Mapping[str, object]


def record_value(data: LayoutRecordInput, key: LayoutRecordKey) -> object:
    """Return a layout-record value by enum key."""
    return data[key.value]


def optional_record_value(
    data: LayoutRecordInput, key: LayoutRecordKey, default: object
) -> object:
    """Return a layout-record value by enum key, or a default."""
    return data.get(key.value, default)

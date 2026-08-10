"""Shared key names for Hugging Face layout sample extraction."""

from __future__ import annotations

from typing import Final

LAYOUT_BBOX_KEYS: Final[tuple[str, ...]] = ("bbox", "bboxes", "boxes")
LAYOUT_LABEL_KEYS: Final[tuple[str, ...]] = (
    "labels",
    "label",
    "category",
    "categories",
    "type",
    "class_id",
)
LAYOUT_ANNOTATION_KEYS: Final[tuple[str, ...]] = (
    "annotations",
    "objects",
    "elements",
    "children",
)

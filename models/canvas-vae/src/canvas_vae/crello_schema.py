"""Shared Crello field names used by dataset and training records."""

from __future__ import annotations

from typing import Final, Literal

CrelloContextField = Literal[
    "length",
    "group",
    "format",
    "canvas_width",
    "canvas_height",
    "category",
]

CRELLO_CONTEXT_FIELDS: Final[tuple[CrelloContextField, ...]] = (
    "length",
    "group",
    "format",
    "canvas_width",
    "canvas_height",
    "category",
)
CRELLO_SEQUENCE_FIELDS: Final[tuple[str, ...]] = (
    "type",
    "left",
    "top",
    "width",
    "height",
    "opacity",
    "color",
    "image_embedding",
)

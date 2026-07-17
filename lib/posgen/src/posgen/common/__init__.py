"""Common building blocks for future position-generation packages."""

from .content import PositionContent
from .labels import normalize_label
from .testing import assert_position_content_schema
from .visualization import render_position_summary

__all__ = [
    "PositionContent",
    "assert_position_content_schema",
    "normalize_label",
    "render_position_summary",
]

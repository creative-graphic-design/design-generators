"""Configuration enums for LayoutFlow training."""

from __future__ import annotations

from enum import StrEnum, auto


class LayoutFlowSeedMode(StrEnum):
    """Seed modes for regular and deterministic LayoutFlow training."""

    default = auto()
    deterministic = auto()

"""Configuration enums for LayoutFlow training."""

from __future__ import annotations

from enum import StrEnum, auto


class LayoutFlowSeedMode(StrEnum):
    """Seed modes for vendor-compatible and strict parity training."""

    vendor_compat = auto()
    strict_deterministic = auto()

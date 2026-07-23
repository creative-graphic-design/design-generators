"""Small constrained training configuration types for DLT."""

from __future__ import annotations

from enum import StrEnum, auto


class DLTSeedMode(StrEnum):
    """Closed set of DLT training seed modes."""

    default = auto()
    deterministic = auto()

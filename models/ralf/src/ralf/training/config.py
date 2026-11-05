"""Stage definitions for RALF reproduction training."""

from __future__ import annotations

from enum import StrEnum, auto


class RalfTrainingStage(StrEnum):
    """Ordered training reproduction stages."""

    s0 = auto()
    s1 = auto()
    s2 = auto()
    s3 = auto()
    s4 = auto()
    s5 = auto()

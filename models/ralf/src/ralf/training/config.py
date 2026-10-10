"""Stage definitions for RALF reproduction training."""

from __future__ import annotations

from enum import StrEnum


class RalfTrainingStage(StrEnum):
    """Ordered training reproduction stages."""

    s0 = "S0"
    s1 = "S1"
    s2 = "S2"
    s3 = "S3"
    s4 = "S4"
    s5 = "S5"

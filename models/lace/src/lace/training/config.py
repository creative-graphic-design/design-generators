"""Configuration types for package-local LACE training."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Literal, TypeAlias

LaceTrainingDatasetName: TypeAlias = Literal["publaynet", "rico25"]
LaceTrainingSplit: TypeAlias = Literal["train", "validation", "test"]


class LaceSeedMode(StrEnum):
    """Regular and deterministic seed policies."""

    default = auto()
    deterministic = auto()

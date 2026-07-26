"""Configuration enums for LayoutDM training."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Literal, TypeAlias

LayoutDMTrainingDatasetName: TypeAlias = Literal["rico25", "publaynet"]
"""Dataset names supported by package-local LayoutDM training data."""

LayoutDMTrainingSplit: TypeAlias = Literal["train", "validation", "test"]
"""Split names supported by package-local LayoutDM training data."""

LayoutDMTrainingScheduler: TypeAlias = Literal["reduce_on_plateau"]
"""Scheduler names supported by package-local LayoutDM training."""

LayoutDMTimeSampler: TypeAlias = Literal["importance", "uniform"]
"""Timestep-sampling strategies used by the categorical diffusion loss."""


class LayoutDMSeedMode(StrEnum):
    """Seed modes for regular and deterministic LayoutDM training."""

    default = auto()
    deterministic = auto()

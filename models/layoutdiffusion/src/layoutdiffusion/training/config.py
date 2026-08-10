"""Configuration enums for LayoutDiffusion training."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Literal, TypeAlias

LayoutDiffusionTrainingDatasetName: TypeAlias = Literal["rico25", "publaynet"]
"""Dataset names supported by package-local LayoutDiffusion training data."""

LayoutDiffusionTrainingDatasetSource: TypeAlias = Literal["hf", "processed"]
"""Dataset source modes supported by package-local LayoutDiffusion training data."""

LayoutDiffusionTrainingSplit: TypeAlias = Literal["train", "validation", "test"]
"""Split names supported by package-local LayoutDiffusion training data."""

LayoutDiffusionTrainingTransform: TypeAlias = Literal["LexicographicOrder"]
"""Training-only layout transforms supported by package-local data."""

LayoutDiffusionTrainingScheduler: TypeAlias = Literal["linear_anneal"]
"""Scheduler names supported by package-local LayoutDiffusion training."""

LayoutDiffusionTimeSampler: TypeAlias = Literal["importance", "uniform"]
"""Timestep-sampling strategies used by the categorical diffusion loss."""


class LayoutDiffusionSeedMode(StrEnum):
    """Seed modes for regular and deterministic LayoutDiffusion training."""

    default = auto()
    deterministic = auto()

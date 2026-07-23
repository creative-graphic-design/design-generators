"""Configuration enums for LayoutFlow training."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Literal, TypeAlias


LayoutFlowTrainingDatasetName: TypeAlias = Literal["rico25", "publaynet"]
"""Dataset names supported by package-local LayoutFlow training data."""

LayoutFlowTrainingSplit: TypeAlias = Literal["train", "validation", "test"]
"""HDF5 split names supported by package-local LayoutFlow training data."""

LayoutFlowTrainingScheduler: TypeAlias = Literal["reduce_on_plateau"]
"""Scheduler names supported by package-local LayoutFlow training."""

LayoutFlowConditionPolicy: TypeAlias = Literal["random4"]
"""Condition-mask policy names supported by package-local LayoutFlow training."""


class LayoutFlowSeedMode(StrEnum):
    """Seed modes for regular and deterministic LayoutFlow training."""

    default = auto()
    deterministic = auto()

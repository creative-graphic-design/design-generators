"""Configuration types for Layout-Corrector training."""

from __future__ import annotations

from typing import Literal, TypeAlias

LayoutCorrectorTrainingDatasetName: TypeAlias = Literal["rico25", "publaynet"]
LayoutCorrectorTrainingDatasetSource: TypeAlias = Literal["processed"]
LayoutCorrectorTrainingSplit: TypeAlias = Literal["train", "validation", "test"]
LayoutCorrectorTrainingScheduler: TypeAlias = Literal["reduce_on_plateau"]

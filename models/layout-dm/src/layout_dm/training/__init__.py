"""Training entry points for LayoutDM."""

from __future__ import annotations

from .config import (
    LayoutDMSeedMode,
    LayoutDMTimeSampler,
    LayoutDMTrainingDatasetName,
    LayoutDMTrainingDatasetSource,
    LayoutDMTrainingScheduler,
    LayoutDMTrainingSplit,
)
from .dataset import LayoutDMDataset, LayoutDMProcessedDataset, LayoutDMSyntheticDataset

_LIGHTNING_EXPORTS: tuple[type[object], ...] = ()
try:
    from .datamodule import LayoutDMDataModule
    from .lightning_module import LayoutDMTrainingModule

    _LIGHTNING_EXPORTS = (LayoutDMDataModule, LayoutDMTrainingModule)
except ModuleNotFoundError as exc:
    if exc.name != "lightning":
        raise

__all__ = [
    "LayoutDMDataModule",
    "LayoutDMDataset",
    "LayoutDMProcessedDataset",
    "LayoutDMSeedMode",
    "LayoutDMSyntheticDataset",
    "LayoutDMTimeSampler",
    "LayoutDMTrainingDatasetName",
    "LayoutDMTrainingDatasetSource",
    "LayoutDMTrainingModule",
    "LayoutDMTrainingScheduler",
    "LayoutDMTrainingSplit",
]
__all__.extend(symbol.__name__ for symbol in _LIGHTNING_EXPORTS)

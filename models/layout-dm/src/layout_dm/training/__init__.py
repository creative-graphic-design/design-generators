"""Training entry points for LayoutDM."""

from __future__ import annotations

from .config import (
    LayoutDMSeedMode,
    LayoutDMTimeSampler,
    LayoutDMTrainingDatasetName,
    LayoutDMTrainingScheduler,
    LayoutDMTrainingSplit,
)
from .dataset import LayoutDMDataset, LayoutDMSyntheticDataset

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
    "LayoutDMSeedMode",
    "LayoutDMSyntheticDataset",
    "LayoutDMTimeSampler",
    "LayoutDMTrainingDatasetName",
    "LayoutDMTrainingModule",
    "LayoutDMTrainingScheduler",
    "LayoutDMTrainingSplit",
]
__all__.extend(symbol.__name__ for symbol in _LIGHTNING_EXPORTS)

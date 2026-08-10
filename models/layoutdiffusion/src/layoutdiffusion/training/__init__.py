"""Training entry points for LayoutDiffusion."""

from __future__ import annotations

from .config import (
    LayoutDiffusionSeedMode,
    LayoutDiffusionTimeSampler,
    LayoutDiffusionTrainingDatasetName,
    LayoutDiffusionTrainingDatasetSource,
    LayoutDiffusionTrainingScheduler,
    LayoutDiffusionTrainingSplit,
    LayoutDiffusionTrainingTransform,
)
from .dataset import (
    LayoutDiffusionDataset,
    LayoutDiffusionProcessedDataset,
    LayoutDiffusionSyntheticDataset,
)

_LIGHTNING_EXPORTS: tuple[type[object], ...] = ()
try:
    from .datamodule import LayoutDiffusionDataModule
    from .lightning_module import LayoutDiffusionTrainingModule

    _LIGHTNING_EXPORTS = (LayoutDiffusionDataModule, LayoutDiffusionTrainingModule)
except ModuleNotFoundError as exc:
    if exc.name != "lightning":
        raise

__all__ = [
    "LayoutDiffusionDataset",
    "LayoutDiffusionDataModule",
    "LayoutDiffusionProcessedDataset",
    "LayoutDiffusionSeedMode",
    "LayoutDiffusionSyntheticDataset",
    "LayoutDiffusionTimeSampler",
    "LayoutDiffusionTrainingDatasetName",
    "LayoutDiffusionTrainingDatasetSource",
    "LayoutDiffusionTrainingScheduler",
    "LayoutDiffusionTrainingSplit",
    "LayoutDiffusionTrainingTransform",
    "LayoutDiffusionTrainingModule",
]
__all__.extend(symbol.__name__ for symbol in _LIGHTNING_EXPORTS)

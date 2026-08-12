"""Training entry points for LayoutDM."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datamodule import LayoutDMDataModule
    from .lightning_module import LayoutDMTrainingModule

from .config import (
    LayoutDMSeedMode,
    LayoutDMTimeSampler,
    LayoutDMTrainingDatasetName,
    LayoutDMTrainingDatasetSource,
    LayoutDMTrainingScheduler,
    LayoutDMTrainingSplit,
)
from .dataset import LayoutDMDataset, LayoutDMProcessedDataset, LayoutDMSyntheticDataset

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


def __getattr__(name: str) -> type[LayoutDMDataModule | LayoutDMTrainingModule]:
    """Resolve one LayoutDM training class on first access."""
    resolved: type[LayoutDMDataModule | LayoutDMTrainingModule]
    try:
        if name == "LayoutDMDataModule":
            from .datamodule import LayoutDMDataModule

            resolved = LayoutDMDataModule
        elif name == "LayoutDMTrainingModule":
            from .lightning_module import LayoutDMTrainingModule

            resolved = LayoutDMTrainingModule
        else:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    except ModuleNotFoundError as error:
        missing_root = error.name.partition(".")[0] if error.name is not None else None
        if missing_root != "lightning":
            raise
        raise ImportError(
            f"{__name__}.{name} requires the optional 'lightning' dependency; "
            f"install the training extra with `pip install 'layout-dm[training]'`."
        ) from error

    globals()[name] = resolved
    return resolved


def __dir__() -> list[str]:
    """Return stable eager and lazy training namespace names."""
    return sorted(set(globals()) | set(__all__))

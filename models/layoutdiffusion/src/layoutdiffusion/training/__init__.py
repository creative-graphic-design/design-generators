"""Training entry points for LayoutDiffusion."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datamodule import LayoutDiffusionDataModule
    from .lightning_module import LayoutDiffusionTrainingModule

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


def __getattr__(
    name: str,
) -> type[LayoutDiffusionDataModule | LayoutDiffusionTrainingModule]:
    """Resolve one LayoutDiffusion training class on first access."""
    resolved: type[LayoutDiffusionDataModule | LayoutDiffusionTrainingModule]
    try:
        if name == "LayoutDiffusionDataModule":
            from .datamodule import LayoutDiffusionDataModule

            resolved = LayoutDiffusionDataModule
        elif name == "LayoutDiffusionTrainingModule":
            from .lightning_module import LayoutDiffusionTrainingModule

            resolved = LayoutDiffusionTrainingModule
        else:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    except ModuleNotFoundError as error:
        missing_root = error.name.partition(".")[0] if error.name is not None else None
        if missing_root != "lightning":
            raise
        raise ImportError(
            f"{__name__}.{name} requires the optional 'lightning' dependency; "
            f"install the training extra with `pip install 'layoutdiffusion[training]'`."
        ) from error

    globals()[name] = resolved
    return resolved


def __dir__() -> list[str]:
    """Return stable eager and lazy training namespace names."""
    return sorted(set(globals()) | set(__all__))

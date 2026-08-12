"""Training entry points for LayoutDiffusion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laygen.common.import_utils import (
    LazyClassExport,
    build_module_dir,
    resolve_lazy_class,
)

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

_LAZY_EXPORTS: dict[
    str, LazyClassExport[LayoutDiffusionDataModule | LayoutDiffusionTrainingModule]
] = {
    "LayoutDiffusionDataModule": LazyClassExport(
        module="layoutdiffusion.training.datamodule",
        attribute="LayoutDiffusionDataModule",
        optional_roots=frozenset({"lightning"}),
    ),
    "LayoutDiffusionTrainingModule": LazyClassExport(
        module="layoutdiffusion.training.lightning_module",
        attribute="LayoutDiffusionTrainingModule",
        optional_roots=frozenset({"lightning"}),
    ),
}

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
    """Resolve a lazy LayoutDiffusion training class."""
    return resolve_lazy_class(
        name,
        module_name=__name__,
        distribution_name="layoutdiffusion",
        exports=_LAZY_EXPORTS,
    )


def __dir__() -> list[str]:
    """Return stable eager and lazy training namespace names."""
    return build_module_dir(globals(), _LAZY_EXPORTS)

"""Training entry points for LayoutDM."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laygen.common.import_utils import (
    LazyClassExport,
    build_module_dir,
    resolve_lazy_class,
)

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

_LAZY_EXPORTS: dict[
    str, LazyClassExport[LayoutDMDataModule | LayoutDMTrainingModule]
] = {
    "LayoutDMDataModule": LazyClassExport(
        module="layout_dm.training.datamodule",
        attribute="LayoutDMDataModule",
        optional_roots=frozenset({"lightning"}),
    ),
    "LayoutDMTrainingModule": LazyClassExport(
        module="layout_dm.training.lightning_module",
        attribute="LayoutDMTrainingModule",
        optional_roots=frozenset({"lightning"}),
    ),
}

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
    """Resolve a lazy LayoutDM training class."""
    return resolve_lazy_class(
        name,
        module_name=__name__,
        distribution_name="layout-dm",
        exports=_LAZY_EXPORTS,
    )


def __dir__() -> list[str]:
    """Return stable eager and lazy training namespace names."""
    return build_module_dir(globals(), _LAZY_EXPORTS)

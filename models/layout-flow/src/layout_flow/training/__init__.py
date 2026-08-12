"""Training entry points for LayoutFlow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laygen.common.import_utils import (
    LazyClassExport,
    build_module_dir,
    resolve_lazy_class,
)

if TYPE_CHECKING:
    from .datamodule import LayoutFlowDataModule
    from .lightning_module import LayoutFlowTrainingModule

from .config import (
    LayoutFlowConditionPolicy,
    LayoutFlowSeedMode,
    LayoutFlowTrainingDatasetName,
    LayoutFlowTrainingScheduler,
    LayoutFlowTrainingSplit,
)
from .dataset import LayoutFlowH5Dataset, collate_layout_flow_batch

_LAZY_EXPORTS: dict[
    str, LazyClassExport[LayoutFlowDataModule | LayoutFlowTrainingModule]
] = {
    "LayoutFlowDataModule": LazyClassExport(
        module="layout_flow.training.datamodule",
        attribute="LayoutFlowDataModule",
        optional_roots=frozenset({"lightning"}),
    ),
    "LayoutFlowTrainingModule": LazyClassExport(
        module="layout_flow.training.lightning_module",
        attribute="LayoutFlowTrainingModule",
        optional_roots=frozenset({"lightning"}),
    ),
}


__all__ = [
    "LayoutFlowConditionPolicy",
    "LayoutFlowDataModule",
    "LayoutFlowH5Dataset",
    "LayoutFlowSeedMode",
    "LayoutFlowTrainingDatasetName",
    "LayoutFlowTrainingModule",
    "LayoutFlowTrainingScheduler",
    "LayoutFlowTrainingSplit",
    "collate_layout_flow_batch",
]


def __getattr__(name: str) -> type[LayoutFlowDataModule | LayoutFlowTrainingModule]:
    """Resolve a lazy LayoutFlow training class."""
    return resolve_lazy_class(
        name,
        module_name=__name__,
        distribution_name="layout-flow",
        exports=_LAZY_EXPORTS,
    )


def __dir__() -> list[str]:
    """Return stable eager and lazy training namespace names."""
    return build_module_dir(globals(), _LAZY_EXPORTS)

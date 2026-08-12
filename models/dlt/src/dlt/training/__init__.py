"""Training utilities for DLT."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laygen.common.import_utils import (
    LazyClassExport,
    build_module_dir,
    resolve_lazy_class,
)

if TYPE_CHECKING:
    from .callbacks import DLTReferenceEpochSamplingCallback
    from .datamodule import DLTDataModule
    from .lightning_module import DLTTrainingModule, DLTWarmupCosineSchedulerFactory

from .config import DLTSeedMode

_LAZY_EXPORTS: dict[
    str,
    LazyClassExport[
        DLTDataModule
        | DLTReferenceEpochSamplingCallback
        | DLTTrainingModule
        | DLTWarmupCosineSchedulerFactory
    ],
] = {
    "DLTDataModule": LazyClassExport(
        module="dlt.training.datamodule",
        attribute="DLTDataModule",
        optional_roots=frozenset({"lightning", "h5py"}),
    ),
    "DLTReferenceEpochSamplingCallback": LazyClassExport(
        module="dlt.training.callbacks",
        attribute="DLTReferenceEpochSamplingCallback",
        optional_roots=frozenset({"lightning", "h5py"}),
    ),
    "DLTTrainingModule": LazyClassExport(
        module="dlt.training.lightning_module",
        attribute="DLTTrainingModule",
        optional_roots=frozenset({"lightning", "h5py"}),
    ),
    "DLTWarmupCosineSchedulerFactory": LazyClassExport(
        module="dlt.training.lightning_module",
        attribute="DLTWarmupCosineSchedulerFactory",
        optional_roots=frozenset({"lightning", "h5py"}),
    ),
}

__all__ = [
    "DLTSeedMode",
    "DLTDataModule",
    "DLTReferenceEpochSamplingCallback",
    "DLTTrainingModule",
    "DLTWarmupCosineSchedulerFactory",
]


def __getattr__(
    name: str,
) -> type[
    DLTDataModule
    | DLTReferenceEpochSamplingCallback
    | DLTTrainingModule
    | DLTWarmupCosineSchedulerFactory
]:
    """Resolve a lazy DLT training class or factory."""
    return resolve_lazy_class(
        name,
        module_name=__name__,
        distribution_name="dlt",
        exports=_LAZY_EXPORTS,
    )


def __dir__() -> list[str]:
    """Return stable eager and lazy training namespace names."""
    return build_module_dir(globals(), _LAZY_EXPORTS)

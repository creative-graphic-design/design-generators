"""Training utilities for CGB-DM."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laygen.common.import_utils import (
    LazyClassExport,
    build_module_dir,
    resolve_lazy_class,
)

if TYPE_CHECKING:
    from .datamodule import CGBDMDataModule
    from .lightning_module import CGBDMTrainingModule

from .config import CGBDMSeedMode

_LAZY_EXPORTS: dict[str, LazyClassExport[CGBDMDataModule | CGBDMTrainingModule]] = {
    "CGBDMDataModule": LazyClassExport(
        module="cgb_dm.training.datamodule",
        attribute="CGBDMDataModule",
        optional_roots=frozenset({"lightning"}),
    ),
    "CGBDMTrainingModule": LazyClassExport(
        module="cgb_dm.training.lightning_module",
        attribute="CGBDMTrainingModule",
        optional_roots=frozenset({"lightning"}),
    ),
}

__all__ = ["CGBDMDataModule", "CGBDMSeedMode", "CGBDMTrainingModule"]


def __getattr__(name: str) -> type[CGBDMDataModule | CGBDMTrainingModule]:
    """Resolve a lazy CGB-DM training class."""
    return resolve_lazy_class(
        name,
        module_name=__name__,
        distribution_name="cgb-dm",
        exports=_LAZY_EXPORTS,
    )


def __dir__() -> list[str]:
    """Return stable eager and lazy training namespace names."""
    return build_module_dir(globals(), _LAZY_EXPORTS)

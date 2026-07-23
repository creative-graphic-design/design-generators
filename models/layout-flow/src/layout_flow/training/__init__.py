"""Training entry points for LayoutFlow."""

from __future__ import annotations

from .config import (
    LayoutFlowConditionPolicy,
    LayoutFlowSeedMode,
    LayoutFlowTrainingDatasetName,
    LayoutFlowTrainingScheduler,
    LayoutFlowTrainingSplit,
)
from .dataset import LayoutFlowH5Dataset, collate_layout_flow_batch

_LIGHTNING_EXPORTS: tuple[type[object], ...] = ()
try:
    from .datamodule import LayoutFlowDataModule
    from .lightning_module import LayoutFlowTrainingModule

    _LIGHTNING_EXPORTS = (LayoutFlowDataModule, LayoutFlowTrainingModule)
except ModuleNotFoundError as exc:
    if exc.name != "lightning":
        raise


__all__ = [
    "LayoutFlowConditionPolicy",
    "LayoutFlowDataModule",
    "LayoutFlowH5Dataset",
    "LayoutFlowSeedMode",
    "LayoutFlowTrainingDatasetName",
    "LayoutFlowTrainingScheduler",
    "LayoutFlowTrainingSplit",
    "collate_layout_flow_batch",
]
__all__.extend(symbol.__name__ for symbol in _LIGHTNING_EXPORTS)

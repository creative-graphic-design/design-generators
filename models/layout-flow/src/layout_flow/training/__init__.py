"""Training entry points for LayoutFlow."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    """Resolve one LayoutFlow training class on first access."""
    resolved: type[LayoutFlowDataModule | LayoutFlowTrainingModule]
    try:
        if name == "LayoutFlowDataModule":
            from .datamodule import LayoutFlowDataModule

            resolved = LayoutFlowDataModule
        elif name == "LayoutFlowTrainingModule":
            from .lightning_module import LayoutFlowTrainingModule

            resolved = LayoutFlowTrainingModule
        else:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    except ModuleNotFoundError as error:
        missing_root = error.name.partition(".")[0] if error.name is not None else None
        if missing_root != "lightning":
            raise
        raise ImportError(
            f"{__name__}.{name} requires the optional 'lightning' dependency; "
            f"install the training extra with `pip install 'layout-flow[training]'`."
        ) from error

    globals()[name] = resolved
    return resolved


def __dir__() -> list[str]:
    """Return stable eager and lazy training namespace names."""
    return sorted(set(globals()) | set(__all__))

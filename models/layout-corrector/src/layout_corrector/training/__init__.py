"""Training entry points for Layout-Corrector."""

from importlib.util import find_spec as _find_spec
from typing import TYPE_CHECKING

from .config import (
    LayoutCorrectorTrainingDatasetName as LayoutCorrectorTrainingDatasetName,
    LayoutCorrectorTrainingDatasetSource as LayoutCorrectorTrainingDatasetSource,
    LayoutCorrectorTrainingScheduler as LayoutCorrectorTrainingScheduler,
    LayoutCorrectorTrainingSplit as LayoutCorrectorTrainingSplit,
)
from .dataset import LayoutCorrectorDataModule as LayoutCorrectorDataModule
from .reference import (
    FROZEN_LAYOUT_DM_SHA256 as FROZEN_LAYOUT_DM_SHA256,
    FrozenLayoutDMReference as FrozenLayoutDMReference,
)

if TYPE_CHECKING:
    from .lightning_module import LayoutCorrectorTrainingModule

if _find_spec("lightning") is not None:
    from .lightning_module import (
        LayoutCorrectorTrainingModule as LayoutCorrectorTrainingModule,
    )

__all__ = [
    "FROZEN_LAYOUT_DM_SHA256",
    "FrozenLayoutDMReference",
    "LayoutCorrectorDataModule",
    "LayoutCorrectorTrainingDatasetName",
    "LayoutCorrectorTrainingDatasetSource",
    "LayoutCorrectorTrainingScheduler",
    "LayoutCorrectorTrainingSplit",
]

if _find_spec("lightning") is not None:
    __all__.append("LayoutCorrectorTrainingModule")

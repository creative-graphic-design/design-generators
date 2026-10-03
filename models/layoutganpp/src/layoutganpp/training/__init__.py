"""Training utilities for LayoutGAN++."""

from importlib.util import find_spec as _find_spec
from typing import TYPE_CHECKING

from .config import LayoutGANPPSeedMode as LayoutGANPPSeedMode

if TYPE_CHECKING:
    from .datamodule import LayoutGANPPDataModule as LayoutGANPPDataModule
    from .lightning_module import LayoutGANPPTrainingModule as LayoutGANPPTrainingModule

if _find_spec("lightning") is not None:
    from .datamodule import LayoutGANPPDataModule as LayoutGANPPDataModule
    from .lightning_module import (
        LayoutGANPPTrainingModule as LayoutGANPPTrainingModule,
    )

__all__ = [
    "LayoutGANPPSeedMode",
]

if "LayoutGANPPDataModule" in globals():
    __all__ += ["LayoutGANPPDataModule", "LayoutGANPPTrainingModule"]

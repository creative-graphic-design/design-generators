"""Training utilities for CGB-DM."""

from importlib.util import find_spec as _find_spec

from .config import CGBDMSeedMode as CGBDMSeedMode

if _find_spec("lightning") is not None:
    from .datamodule import CGBDMDataModule as CGBDMDataModule
    from .lightning_module import CGBDMTrainingModule as CGBDMTrainingModule

"""RADM training entry points."""

from importlib.util import find_spec as _find_spec

if _find_spec("lightning") is not None:
    from .datamodule import RADMDataModule as RADMDataModule
    from .lightning_module import RADMTrainingModule as RADMTrainingModule

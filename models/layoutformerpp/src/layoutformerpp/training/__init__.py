"""Package-local LayoutFormer++ training entry points."""

# ruff: noqa: F401

from importlib.util import find_spec as _find_spec

from .recipes import (
    LayoutFormerPPTrainingRecipe,
    TRAINING_RECIPES,
    TRAINING_RECIPES_BY_NAME,
    get_training_recipe,
)
from .scheduler import LayoutFormerPPWarmupLR

if _find_spec("lightning") is not None:
    from .lightning_module import LayoutFormerPPDataModule, LayoutFormerPPTrainingModule

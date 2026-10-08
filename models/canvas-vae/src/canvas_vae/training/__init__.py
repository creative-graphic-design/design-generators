"""Training utilities for CanvasVAE."""

from importlib.util import find_spec as _find_spec

from .optim import (
    KerasAdam as KerasAdam,
    clip_gradients_by_norm as clip_gradients_by_norm,
    l2_penalty as l2_penalty,
)
from .sampling import (
    CrossEpochBatchSampler as CrossEpochBatchSampler,
    sequential_batches as sequential_batches,
    wrapping_batches as wrapping_batches,
)

if _find_spec("lightning") is not None:
    from .datamodule import CanvasVAEDataModule as CanvasVAEDataModule
    from .lightning_module import CanvasVAETrainingModule as CanvasVAETrainingModule

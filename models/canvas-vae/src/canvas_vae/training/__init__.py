"""Training utilities for CanvasVAE."""

from importlib.util import find_spec as _find_spec

from .optim import (
    clip_gradients_by_norm as clip_gradients_by_norm,
    l2_penalty as l2_penalty,
)
from .sampling import (
    CrossEpochBatchSampler as CrossEpochBatchSampler,
    sequential_batches as sequential_batches,
    wrapping_batches as wrapping_batches,
)

if _find_spec("lightning") is not None:
    from .datamodule import (
        CanvasVAECrelloDataModule as CanvasVAECrelloDataModule,
        CanvasVAEDataModule as CanvasVAEDataModule,
    )
    from .lightning_module import (
        CanvasVAECrelloTrainingModule as CanvasVAECrelloTrainingModule,
        CanvasVAETrainingModule as CanvasVAETrainingModule,
    )

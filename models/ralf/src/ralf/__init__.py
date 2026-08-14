"""RALF Transformers-style package."""

from .configuration_ralf import RalfConfig
from .image_processing_ralf import RalfImageProcessor
from .modeling_ralf import RalfForConditionalLayoutGeneration
from .pipeline_ralf import RalfPipeline
from .processing_ralf import RalfProcessor
from .retrieval import RalfRetrievalTable, RalfRetrievedBatch
from .tokenization_ralf import RalfLayoutTokenizer

try:
    from .training.config import RalfTrainingConfig, RalfTrainingStage
    from .training.datamodule import RalfDataModule, RalfTrainingDataset
    from .training.lightning_module import RalfTrainingModule
except ImportError:
    # The optional training extra keeps inference imports lightweight.
    pass

__all__ = [
    "RalfConfig",
    "RalfForConditionalLayoutGeneration",
    "RalfImageProcessor",
    "RalfLayoutTokenizer",
    "RalfPipeline",
    "RalfProcessor",
    "RalfRetrievalTable",
    "RalfRetrievedBatch",
]

if "RalfTrainingModule" in globals():
    __all__ += [
        "RalfDataModule",
        "RalfTrainingConfig",
        "RalfTrainingDataset",
        "RalfTrainingModule",
        "RalfTrainingStage",
    ]

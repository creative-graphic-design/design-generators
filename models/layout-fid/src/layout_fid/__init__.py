"""Public API for the layout FID evaluator package."""

from .configuration_layout_fid import (
    LayoutFIDArchitecture,
    LayoutFIDConfig,
    LayoutFIDSource,
    LayoutFIDStatsSplit,
    normalize_architecture,
    normalize_source,
    normalize_stats_split,
)
from .evaluation import (
    LayoutFIDStatistics,
    calculate_frechet_distance,
    compute_feature_statistics,
    compute_layout_fid,
    compute_layout_fid_from_statistics,
    load_reference_statistics,
    save_reference_statistics,
)
from .modeling_layout_fid import LayoutFIDModel, LayoutFIDOutput
from .pipeline_layout_fid import LayoutFIDEvaluator
from .processing_layout_fid import LayoutFIDBatch, LayoutFIDProcessor

FIDNetFeatureExtractor = LayoutFIDProcessor
FIDNetModel = LayoutFIDModel
LayoutFIDCalculator = LayoutFIDEvaluator
LayoutFIDFeatureExtractor = LayoutFIDProcessor

__all__ = [
    "FIDNetFeatureExtractor",
    "FIDNetModel",
    "LayoutFIDArchitecture",
    "LayoutFIDBatch",
    "LayoutFIDCalculator",
    "LayoutFIDConfig",
    "LayoutFIDEvaluator",
    "LayoutFIDFeatureExtractor",
    "LayoutFIDModel",
    "LayoutFIDOutput",
    "LayoutFIDProcessor",
    "LayoutFIDSource",
    "LayoutFIDStatistics",
    "LayoutFIDStatsSplit",
    "calculate_frechet_distance",
    "compute_feature_statistics",
    "compute_layout_fid",
    "compute_layout_fid_from_statistics",
    "load_reference_statistics",
    "normalize_architecture",
    "normalize_source",
    "normalize_stats_split",
    "save_reference_statistics",
]

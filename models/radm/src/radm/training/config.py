"""Configuration constants for RADM training."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal


class RADMTrainingDataset(StrEnum):
    """Datasets supported by the RADM training adapter."""

    cgl_v2 = "cgl_v2"


class RADMTextFeaturePolicy(StrEnum):
    """Policy for text features in content-aware RADM training."""

    hf = "hf"
    zeros = "zeros"


RADMTrainingSplit = Literal["train", "validation", "test", "no_annotation"]

DEFAULT_DATA_ROOT: Final[str] = "data/radm/cgl-dataset-v2"
DEFAULT_VENDOR_BACKBONE_URL: Final[str] = (
    "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl"
)
DEFAULT_NUM_PROPOSALS: Final[int] = 100
DEFAULT_NUM_CLASSES: Final[int] = 5
DEFAULT_HIDDEN_DIM: Final[int] = 256
DEFAULT_TEXT_FEATURE_DIM: Final[int] = 768
DEFAULT_MAX_TEXT_NUM: Final[int] = 20
DEFAULT_NUM_TIMESTEPS: Final[int] = 1000
DEFAULT_SNR_SCALE: Final[float] = 2.0
DEFAULT_BATCH_SIZE: Final[int] = 16
DEFAULT_NUM_WORKERS: Final[int] = 8
DEFAULT_PREFETCH_FACTOR: Final[int] = 2
DEFAULT_BASE_LR: Final[float] = 0.000025
DEFAULT_WEIGHT_DECAY: Final[float] = 0.0001
DEFAULT_MAX_ITER: Final[int] = 250000
DEFAULT_LR_STEPS: Final[tuple[int, int]] = (150000, 220000)
DEFAULT_GRAD_CLIP_NORM: Final[float] = 1.0
DEFAULT_CLASS_WEIGHT: Final[float] = 5.0
DEFAULT_L1_WEIGHT: Final[float] = 1.0
DEFAULT_GIOU_WEIGHT: Final[float] = 1.0
DEFAULT_ALPHA: Final[float] = 0.25
DEFAULT_GAMMA: Final[float] = 2.0
DEFAULT_OTA_K: Final[int] = 5

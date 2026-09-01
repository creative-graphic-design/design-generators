"""Configuration and stage definitions for RALF reproduction training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from ..configuration_ralf import RalfConfig

RalfTrainingDatasetName = Literal["cgl", "cgl_v2", "pku", "pku_posterlayout"]
RalfTrainingScheduler = Literal["multi_step", "none"]


class RalfTrainingStage(StrEnum):
    """Ordered training reproduction stages."""

    s0 = "S0"
    s1 = "S1"
    s2 = "S2"
    s3 = "S3"
    s4 = "S4"
    s5 = "S5"


@dataclass(frozen=True)
class RalfTrainingConfig:
    """Runtime settings shared by the package training components."""

    dataset_name: RalfTrainingDatasetName = "cgl"
    condition_type: str = "unconditional"
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    clip_max_norm: float = 0.1
    epochs: int = 70
    scheduler: RalfTrainingScheduler = "multi_step"
    scheduler_milestones: tuple[float, ...] = (0.7,)
    seed: int = 1
    data_root: str | None = None
    retrieval_index_path: str | None = None
    max_seq_length: int = 10
    top_k: int = 16

    @staticmethod
    def stage_order() -> tuple[RalfTrainingStage, ...]:
        """Return the only accepted stage order."""
        return tuple(RalfTrainingStage)

    def model_config(self, config: RalfConfig) -> RalfConfig:
        """Validate that data settings agree with a model config."""
        if config.dataset_name != self.dataset_name:
            raise ValueError(
                "training dataset and model dataset differ: "
                f"{self.dataset_name!r} != {config.dataset_name!r}"
            )

        if config.max_seq_length != self.max_seq_length:
            raise ValueError(
                "training max_seq_length and model max_seq_length differ: "
                f"{self.max_seq_length} != {config.max_seq_length}"
            )

        if config.top_k != self.top_k:
            raise ValueError(
                f"training top_k and model top_k differ: {self.top_k} != {config.top_k}"
            )

        return config

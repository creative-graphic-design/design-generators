"""Effective training configuration captured from the released RADM recipe."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Final

from laygen.common.serialization import YamlValue, sanitize_for_yaml


RADM_CLASS_ID_TO_LABEL: Final[dict[int, str]] = {
    0: "Logo",
    1: "文字",
    2: "衬底",
    3: "符号元素",
    4: "强调突出子部分文字",
}
RADM_PREDICTED_CLASS_ID_TO_LABEL: Final[dict[int, str]] = {
    key: value for key, value in RADM_CLASS_ID_TO_LABEL.items() if key < 4
}


@dataclass(frozen=True)
class RADMEffectiveConfig:
    """Static state observed from the pinned RADM training configuration."""

    num_classes: int = 4
    vocabulary_size: int = 5
    class_id_to_label: dict[int, str] = field(
        default_factory=lambda: dict(RADM_CLASS_ID_TO_LABEL)
    )
    predicted_class_id_to_label: dict[int, str] = field(
        default_factory=lambda: dict(RADM_PREDICTED_CLASS_ID_TO_LABEL)
    )
    num_proposals: int = 100
    hidden_dim: int = 256
    text_feature_dim: int = 768
    max_text_num: int = 20
    num_heads: int = 6
    num_attention_heads: int = 8
    dim_feedforward: int = 2048
    num_dynamic: int = 2
    dim_dynamic: int = 64
    num_cls: int = 1
    num_reg: int = 3
    roi_resolution: int = 7
    roi_sampling_ratio: int = 2
    backbone_depth: int = 50
    backbone_freeze_at: int = 2
    pixel_mean: tuple[float, float, float] = (123.675, 116.28, 103.53)
    pixel_std: tuple[float, float, float] = (58.395, 57.12, 57.375)
    with_vtram: bool = True
    with_gram: bool = True
    deep_supervision: bool = True
    use_focal: bool = True
    use_fed_loss: bool = False
    class_weight: float = 5.0
    giou_weight: float = 1.0
    l1_weight: float = 1.0
    no_object_weight: float = 0.1
    prior_prob: float = 0.01
    alpha: float = 0.25
    gamma: float = 2.0
    ota_k: int = 5
    num_train_timesteps: int = 1000
    snr_scale: float = 2.0
    sample_step: int = 1
    optimizer: str = "ADAMW"
    learning_rate: float = 2.5e-5
    weight_decay: float = 1.0e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    batch_size: int = 16
    backbone_multiplier: float = 1.0
    gradient_clip_norm: float = 1.0
    warmup_factor: float = 0.01
    warmup_iters: int = 1000
    milestones: tuple[int, int] = (150000, 220000)
    max_iter: int = 250000
    scheduler_gamma: float = 0.1
    scheduler_interval: str = "step"
    num_gpus: int = 1
    world_size: int = 1
    gradient_accumulation_steps: int = 1
    eval_period: int = 5000
    num_workers: int = 0
    filter_empty_annotations: bool = False
    seed: int = 1
    min_size_train: tuple[int, ...] = (
        480,
        512,
        544,
        576,
        608,
        640,
        672,
        704,
        736,
        768,
        800,
    )
    max_size_train: int = 1333
    min_size_train_sampling: str = "choice"
    crop_enabled: bool = False
    crop_size: tuple[int, int] = (384, 600)
    crop_type: str = "absolute_range"
    box_renewal: bool = True
    use_ensemble: bool = True
    random_repeat_permutation: bool = True
    ema_enabled: bool = False
    amp_enabled: bool = False
    ddp_enabled: bool = False
    simple_trainer: bool = True
    transform_names: tuple[str, ...] = ("RandomFlip", "ResizeShortestEdge")
    crop_transform_names: tuple[str, ...] = ()
    text_padding_value: float = 0.0
    text_valid_mask_value: bool = True
    missing_text_fallback: str = "zero_features_all_padding"

    def as_dict(self) -> YamlValue:
        """Return JSON/YAML-friendly static configuration metadata."""
        return sanitize_for_yaml(asdict(self))


def effective_radm_config() -> RADMEffectiveConfig:
    """Return the checked effective configuration for the one-GPU recipe."""
    return RADMEffectiveConfig()

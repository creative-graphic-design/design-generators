from __future__ import annotations

from dataclasses import dataclass, field

from diffusers.configuration_utils import ConfigMixin, register_to_config

from laygen.common.labels import id2label_for_dataset, normalize_dataset_name


@dataclass
class LayoutDMConfig(ConfigMixin):
    config_name = "layout_dm_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        dataset_name: str,
        id2label: dict[int | str, str] | None = None,
        max_seq_length: int = 25,
        num_bin_bboxes: int = 32,
        var_order: str = "c-x-y-w-h",
        shared_bbox_vocab: str = "x-y-w-h",
        bbox_quantization: str = "kmeans",
        special_tokens: tuple[str, ...] = ("pad", "mask"),
        cluster_centers: dict[str, list[float]] | None = None,
        hidden_size: int = 464,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 4,
        intermediate_size: int = 1856,
        dropout: float = 0.0,
        timestep_type: str | None = "adalayernorm",
        num_timesteps: int = 100,
        q_type: str = "constrained",
        att_1: float = 0.99999,
        att_T: float = 0.000009,
        ctt_1: float = 0.000009,
        ctt_T: float = 0.99999,
    ) -> None:
        self.dataset_name = normalize_dataset_name(dataset_name)
        raw_id2label = id2label or id2label_for_dataset(self.dataset_name)
        self.id2label = {int(k): v for k, v in raw_id2label.items()}
        self.max_seq_length = max_seq_length
        self.num_bin_bboxes = num_bin_bboxes
        self.var_order = var_order
        self.shared_bbox_vocab = shared_bbox_vocab
        self.bbox_quantization = bbox_quantization
        self.special_tokens = tuple(special_tokens)
        self.cluster_centers = cluster_centers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.timestep_type = timestep_type
        self.num_timesteps = num_timesteps
        self.q_type = q_type
        self.att_1 = att_1
        self.att_T = att_T
        self.ctt_1 = ctt_1
        self.ctt_T = ctt_T

    @property
    def label2id(self) -> dict[str, int]:
        return {v: k for k, v in self.id2label.items()}

    @property
    def num_categories(self) -> int:
        return len(self.id2label)

    @property
    def num_bbox_tokens(self) -> int:
        return self.num_bin_bboxes * len(self.shared_bbox_vocab.split("-"))

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.num_categories + self.num_bbox_tokens + self.num_special_tokens

    @property
    def pad_token_id(self) -> int:
        return (
            self.num_categories
            + self.num_bbox_tokens
            + self.special_tokens.index("pad")
        )

    @property
    def mask_token_id(self) -> int:
        return (
            self.num_categories
            + self.num_bbox_tokens
            + self.special_tokens.index("mask")
        )

    @property
    def num_attributes_per_element(self) -> int:
        return len(self.var_order.split("-"))

    @property
    def max_token_length(self) -> int:
        return self.max_seq_length * self.num_attributes_per_element

    @property
    def bbox_slices(self) -> dict[str, tuple[int, int]]:
        slices: dict[str, tuple[int, int]] = {}
        for i, key in enumerate(("x", "y", "w", "h")):
            start = self.num_categories + i * self.num_bin_bboxes
            slices[key] = (start, start + self.num_bin_bboxes)
        return slices


@dataclass
class LayoutDMRuntimeConfig:
    config: LayoutDMConfig = field(
        default_factory=lambda: LayoutDMConfig(dataset_name="publaynet")
    )

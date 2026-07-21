"""Configuration objects for Flex-DM masked document modeling."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Literal, NotRequired, TypedDict, cast

from transformers import PretrainedConfig


class FlexDmDatasetName(StrEnum):
    """Dataset names supported by the Flex-DM MFP checkpoints."""

    crello = auto()
    rico = auto()


class FlexDmColumnType(StrEnum):
    """Internal column storage type."""

    categorical = auto()
    numerical = auto()


class FlexDmLossCondition(TypedDict):
    """Vendor loss-condition filter for conditionally valid fields."""

    key: str
    mask: tuple[bool, ...]


class FlexDmColumnSpec(TypedDict):
    """Tensor specification for one Flex-DM input/output column."""

    type: Literal["categorical", "numerical"]
    input_dim: int | None
    shape: tuple[int, ...]
    is_sequence: bool
    primary_label: int | None
    loss_condition: NotRequired[FlexDmLossCondition | None]


def _normalize_id2label(id2label: dict[int | str, str] | None) -> dict[int, str]:
    if id2label is None:
        return {
            0: "coloredBackground",
            1: "imageElement",
            2: "maskElement",
            3: "svgElement",
            4: "textElement",
        }
    return {int(key): str(value) for key, value in id2label.items()}


def _normalize_columns(
    input_columns: dict[str, FlexDmColumnSpec] | None,
) -> dict[str, FlexDmColumnSpec]:
    if input_columns is None:
        from .data_specs import build_column_specs

        return build_column_specs(dataset_name=FlexDmDatasetName.crello, vocabulary={})
    normalized: dict[str, FlexDmColumnSpec] = {}
    for key, value in input_columns.items():
        item = dict(value)
        item["shape"] = tuple(cast(tuple[int, ...], item.get("shape", (1,))))
        if "loss_condition" in item and item["loss_condition"] is not None:
            cond = dict(cast(FlexDmLossCondition, item["loss_condition"]))
            raw_mask = cast(tuple[bool, ...] | list[bool], cond["mask"])
            cond["mask"] = tuple(bool(v) for v in raw_mask)
            item["loss_condition"] = cast(FlexDmLossCondition, cond)
        normalized[key] = cast(FlexDmColumnSpec, item)
    return normalized


class FlexDmConfig(PretrainedConfig):
    """Configuration for a converted Flex-DM MFP model.

    Args:
        dataset_name: Vendor dataset name.
        checkpoint_variant: Released checkpoint variant name.
        id2label: Public dataset-local label mapping.
        input_columns: Heterogeneous vendor column specs.
        attribute_groups: Vendor feature groups used for infilling.
        max_seq_length: Maximum document elements.
        latent_dim: Transformer hidden dimension.
        num_blocks: Number of DeepSVG-style transformer blocks.
        block_type: Vendor block type. Only ``deepsvg`` is implemented.
        masking_method: Vendor masking task selector.
        seq_type: Vendor sequence model type. ``default`` is the released path.
        arch_type: Vendor architecture type. ``oneshot`` is the released path.
        context: Optional vendor context embedding mode.
        input_dtype: Vendor input ordering mode.
        use_elemwise_noise: Whether element-wise noise was enabled.
        dropout: Dropout probability.
        layer_norm_epsilon: LayerNorm epsilon matching Keras defaults.
        l2: Original L2 setting, stored for provenance.
        original_args: Raw vendor ``args.json`` values.
        conversion_report: Checkpoint conversion diagnostics.
        kwargs: Extra ``PretrainedConfig`` fields.
    """

    model_type = "flex-dm"

    def __init__(
        self,
        dataset_name: str = "crello",
        checkpoint_variant: str = "ours-exp-ft",
        id2label: dict[int | str, str] | None = None,
        input_columns: dict[str, FlexDmColumnSpec] | None = None,
        attribute_groups: dict[str, tuple[str, ...] | list[str]] | None = None,
        max_seq_length: int = 50,
        latent_dim: int = 256,
        num_blocks: int = 4,
        block_type: str = "deepsvg",
        masking_method: str = "random",
        seq_type: str = "default",
        arch_type: str = "oneshot",
        context: str | None = None,
        input_dtype: str = "set",
        use_elemwise_noise: bool = False,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-3,
        l2: float | None = 1e-2,
        original_args: dict[str, object] | None = None,
        conversion_report: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize a Flex-DM config."""
        normalized_id2label = _normalize_id2label(id2label)
        kwargs.pop("label2id", None)
        super().__init__(
            id2label=normalized_id2label,
            label2id={label: idx for idx, label in normalized_id2label.items()},
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
        self.dataset_name = dataset_name
        self.checkpoint_variant = checkpoint_variant
        self.input_columns = _normalize_columns(input_columns)
        if attribute_groups is None:
            from .data_specs import attribute_groups_for_dataset

            attribute_groups = dict(attribute_groups_for_dataset(dataset_name))
        groups = attribute_groups
        self.attribute_groups = {key: tuple(value) for key, value in groups.items()}
        self.max_seq_length = max_seq_length
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.block_type = block_type
        self.masking_method = masking_method
        self.seq_type = seq_type
        self.arch_type = arch_type
        self.context = context
        self.input_dtype = input_dtype
        self.use_elemwise_noise = use_elemwise_noise
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.l2 = l2
        self.original_args = original_args or {}
        self.conversion_report = conversion_report or {}

    @property
    def max_seq_length_with_length_lookup(self) -> int:
        """Return the max length used by the vendor zero-based length lookup."""
        return self.max_seq_length

    @property
    def valid_sequence_keys(self) -> tuple[str, ...]:
        """Return non-demo sequence fields modeled by Flex-DM."""
        return tuple(
            key for key, column in self.input_columns.items() if column["is_sequence"]
        )

    @property
    def categorical_keys(self) -> tuple[str, ...]:
        """Return sequence fields with categorical heads."""
        return tuple(
            key
            for key, column in self.input_columns.items()
            if column["is_sequence"] and column["type"] == "categorical"
        )

    @property
    def numerical_keys(self) -> tuple[str, ...]:
        """Return sequence fields with numerical heads."""
        return tuple(
            key
            for key, column in self.input_columns.items()
            if column["is_sequence"] and column["type"] == "numerical"
        )

    @property
    def task_names(self) -> tuple[str, ...]:
        """Return vendor task names in sampler order."""
        return ("random", "elem", *self.attribute_groups.keys())

    def mask_token_id_for(self, key: str) -> int:
        """Return the categorical mask token id for ``key``."""
        input_dim = self.input_columns[key]["input_dim"]
        if input_dim is None:
            raise ValueError(f"{key} is not categorical")
        return input_dim

    def unused_token_id_for(self, key: str) -> int:
        """Return the categorical unused token id for ``key``."""
        return self.mask_token_id_for(key) + 1

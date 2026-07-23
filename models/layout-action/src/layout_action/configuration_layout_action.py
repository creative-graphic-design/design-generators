"""Configuration for converted LayoutAction checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum, auto
from typing import Final, cast

from transformers import PretrainedConfig

from .data import (
    layout_action_labels,
    max_elements_for_layout_action_dataset,
    normalize_vendor_dataset_name,
)

ELEMENT_TOKEN_WIDTH: Final[int] = 13


class LayoutActionSamplingMode(StrEnum):
    """Supported token sampling modes."""

    greedy = auto()
    multinomial = auto()
    top_k = auto()


def normalize_sampling_mode(
    value: LayoutActionSamplingMode | str,
) -> LayoutActionSamplingMode:
    """Normalize a sampling-mode value."""
    if isinstance(value, LayoutActionSamplingMode):
        return value
    try:
        return LayoutActionSamplingMode(str(value).lower().replace("-", "_"))
    except ValueError as exc:
        raise ValueError(f"Unsupported LayoutAction sampling mode: {value}") from exc


class LayoutActionConfig(PretrainedConfig):
    """Architecture and tokenizer metadata for LayoutAction checkpoints.

    Args:
        dataset_name: Dataset slug. ``rico`` and ``layout_action_rico13`` map to
            the released RICO13 label order.
        id2label: Dataset-local label mapping.
        precision: Coordinate precision; checkpoint defaults to 8 bits.
        max_elements: Maximum number of layout elements.
        block_size: GPT context length. Defaults to released max token length.
        vocab_size: Token vocabulary size. Defaults to the reference formula.
        n_layer: Number of GPT blocks.
        n_head: Attention heads.
        n_embd: Hidden size.
        embd_pdrop: Embedding dropout.
        resid_pdrop: Residual dropout.
        attn_pdrop: Attention dropout.
        default_sampling: Default pipeline sampling mode.
        default_top_k: Default top-k value.
        default_temperature: Default sampling temperature.
        original_dataset_name: Original dataset name.
        original_asset_manifest: Optional asset manifest.
        kwargs: Additional ``PretrainedConfig`` fields.

    Examples:
        >>> config = LayoutActionConfig(dataset_name="publaynet")
        >>> config.bos_token_id == config.vocab_size - 3
        True
    """

    model_type = "layout-action"

    def __init__(
        self,
        *,
        dataset_name: str = "rico13",
        id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        precision: int = 8,
        max_elements: int | None = None,
        block_size: int | None = None,
        vocab_size: int | None = None,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 512,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        default_sampling: LayoutActionSamplingMode
        | str = LayoutActionSamplingMode.top_k,
        default_top_k: int = 5,
        default_temperature: float = 1.0,
        original_dataset_name: str | None = None,
        original_asset_manifest: Mapping[str, object] | None = None,
        model_type: str | None = None,
        transformers_version: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize LayoutAction metadata and derived token ids."""
        _ = (model_type, transformers_version)
        for derived_key in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "label2id",
            "size",
            "element_token_width",
            "max_token_length",
            "no_value_token_id",
            "label_token_offset",
            "copy_token_id",
            "margin_token_id",
            "generate_token_id",
            "no_obj_token_id",
        ):
            kwargs.pop(derived_key, None)
        _ = kwargs
        super().__init__()
        dataset = normalize_vendor_dataset_name(dataset_name)
        labels = layout_action_labels(dataset)
        normalized_id2label = (
            {int(key): str(value) for key, value in id2label.items()}
            if id2label is not None
            else dict(enumerate(labels))
        )
        self.dataset_name = dataset
        self.precision = int(precision)
        self.max_elements = int(
            max_elements
            if max_elements is not None
            else max_elements_for_layout_action_dataset(dataset)
        )
        self.n_layer = int(n_layer)
        self.n_head = int(n_head)
        self.n_embd = int(n_embd)
        self.embd_pdrop = float(embd_pdrop)
        self.resid_pdrop = float(resid_pdrop)
        self.attn_pdrop = float(attn_pdrop)
        self.default_sampling = str(normalize_sampling_mode(default_sampling))
        self.default_top_k = int(default_top_k)
        self.default_temperature = float(default_temperature)
        self.original_dataset_name = original_dataset_name or dataset
        self.original_asset_manifest = dict(original_asset_manifest or {})
        self.id2label: dict[int, str] = normalized_id2label
        self.label2id: dict[str, int] = {
            value: key for key, value in self.id2label.items()
        }
        self.size: int = 2**self.precision
        self.element_token_width: int = ELEMENT_TOKEN_WIDTH
        self.max_token_length: int = self.max_elements * self.element_token_width + 2
        self.block_size = int(
            block_size if block_size is not None else self.max_token_length
        )
        self.no_value_token_id: int = self.size
        self.label_token_offset: int = self.size + 1
        self.copy_token_id: int = self.label_token_offset + len(self.id2label)
        self.margin_token_id: int = self.copy_token_id + 1
        self.generate_token_id: int = self.margin_token_id + 1
        self.no_obj_token_id: int = self.generate_token_id + 1
        resolved_vocab_size = (
            int(vocab_size)
            if vocab_size is not None
            else self.size + 1 + len(self.id2label) + 3 + 1 + self.max_elements + 3
        )
        self.vocab_size = resolved_vocab_size
        self.bos_token_id = self.vocab_size - 3
        self.eos_token_id = self.vocab_size - 2
        self.pad_token_id = self.vocab_size - 1

    def label_token_id(self, label_id: int) -> int:
        """Return the synthetic token id for a dataset-local label id."""
        id2label = cast(dict[int, str], self.id2label)
        if label_id not in id2label:
            raise ValueError(f"Unknown LayoutAction label id: {label_id}")
        return self.label_token_offset + int(label_id)

    def label_id_from_token(self, token_id: int) -> int | None:
        """Return a dataset-local label id for a label token id."""
        id2label = cast(dict[int, str], self.id2label)
        label_id = int(token_id) - self.label_token_offset
        if 0 <= label_id < len(id2label):
            return label_id
        return None

    def object_token_id(self, back_reference: int) -> int:
        """Return the token id for a previous-object back reference."""
        if not 1 <= back_reference <= self.max_elements:
            raise ValueError("back_reference must be in [1, max_elements]")
        return self.no_obj_token_id + back_reference

    def back_reference_from_token(self, token_id: int) -> int | None:
        """Return a previous-object back reference from a token id."""
        value = int(token_id) - self.no_obj_token_id
        if 1 <= value <= self.max_elements:
            return value
        return None

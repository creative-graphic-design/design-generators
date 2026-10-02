"""Configuration for CanvasVAE layout models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum, auto
from typing import Final

from transformers import PretrainedConfig


class CanvasVAEField(StrEnum):
    """Per-element fields of a CanvasVAE layout, in model column order."""

    left = auto()
    top = auto()
    width = auto()
    height = auto()
    clickable = auto()
    component = auto()
    icon = auto()
    text_button = auto()


GEOMETRY_FIELDS: Final[tuple[CanvasVAEField, ...]] = (
    CanvasVAEField.left,
    CanvasVAEField.top,
    CanvasVAEField.width,
    CanvasVAEField.height,
)
VOCABULARY_FIELDS: Final[tuple[CanvasVAEField, ...]] = (
    CanvasVAEField.component,
    CanvasVAEField.icon,
    CanvasVAEField.text_button,
)
NUM_CLICKABLE_CLASSES: Final[int] = 2
OOV_TOKEN: Final[str] = "[UNK]"
PRIMARY_LABEL_TOKEN: Final[str] = ""


class CanvasVAEConfig(PretrainedConfig):
    """Configuration for a CanvasVAE model trained on RICO.

    A layout has a document-level element count and eight categorical
    per-element fields. Box coordinates and sizes are discretized into
    ``num_bins`` bins over ``[0, 1]``; ``component``, ``icon``, and
    ``text_button`` use the lookup tables in ``vocabularies``, whose index 0 is
    the out-of-vocabulary token.

    Args:
        vocabularies: Lookup tables for ``component``, ``icon``, and
            ``text_button``, each starting with the out-of-vocabulary token.
            Required for a usable model; the default only exists so that
            Transformers can instantiate an empty config while serializing.
        max_length: Maximum number of elements per layout.
        num_bins: Number of discretization bins for box coordinates and sizes.
        latent_dim: Hidden and latent dimension.
        num_blocks: Number of transformer blocks in the encoder and decoder.
        num_heads: Number of attention heads.
        dropout: Dropout rate of position embeddings and blocks.
        kl_weight: Weight of the KL divergence term.
        l2_weight: Weight of the squared-norm penalty on dense and embedding
            weights.
        layer_norm_epsilon: Layer-normalization epsilon.
        batch_norm_epsilon: Batch-normalization epsilon.
        batch_norm_momentum: Batch-normalization moving-average momentum.
        **kwargs: Extra ``PretrainedConfig`` fields.

    Examples:
        >>> config = CanvasVAEConfig(
        ...     vocabularies={
        ...         "component": ["[UNK]", "", "Text"],
        ...         "icon": ["[UNK]", ""],
        ...         "text_button": ["[UNK]", ""],
        ...     },
        ...     latent_dim=16,
        ...     num_heads=2,
        ... )
        >>> config.field_sizes["component"], config.primary_label_id
        (3, 1)
    """

    model_type = "canvas-vae"

    def __init__(
        self,
        vocabularies: Mapping[str, Sequence[str]] | None = None,
        max_length: int = 50,
        num_bins: int = 64,
        latent_dim: int = 256,
        num_blocks: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        kl_weight: float = 16.0,
        l2_weight: float = 1e-6,
        layer_norm_epsilon: float = 1e-3,
        batch_norm_epsilon: float = 1e-3,
        batch_norm_momentum: float = 0.99,
        **kwargs: str | float | bool | None,
    ) -> None:
        """Initialize the config; see the class docstring for arguments."""
        tables = {
            str(key): [str(token) for token in tokens]
            for key, tokens in (vocabularies or {}).items()
        }
        components = tables.get(CanvasVAEField.component, [])
        kwargs.pop("id2label", None)
        kwargs.pop("label2id", None)
        super().__init__(
            id2label=dict(enumerate(components)),
            label2id={token: index for index, token in enumerate(components)},
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
        self.vocabularies = tables
        self.max_length = max_length
        self.num_bins = num_bins
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.kl_weight = kl_weight
        self.l2_weight = l2_weight
        self.layer_norm_epsilon = layer_norm_epsilon
        self.batch_norm_epsilon = batch_norm_epsilon
        self.batch_norm_momentum = batch_norm_momentum

    @property
    def field_sizes(self) -> dict[str, int]:
        """Return the number of classes of every per-element field.

        Raises:
            ValueError: If a required vocabulary is missing.
        """
        missing = [key for key in VOCABULARY_FIELDS if key not in self.vocabularies]
        if missing:
            raise ValueError(f"CanvasVAEConfig is missing vocabularies: {missing}")

        sizes = {str(key): self.num_bins for key in GEOMETRY_FIELDS}
        sizes[CanvasVAEField.clickable] = NUM_CLICKABLE_CLASSES
        for key in VOCABULARY_FIELDS:
            sizes[key] = len(self.vocabularies[key])
        return sizes

    @property
    def primary_label_id(self) -> int:
        """Return the ``component`` id of the empty label used as background."""
        return self.vocabularies[CanvasVAEField.component].index(PRIMARY_LABEL_TOKEN)

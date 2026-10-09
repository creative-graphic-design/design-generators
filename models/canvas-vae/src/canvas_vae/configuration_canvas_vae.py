"""Configuration for CanvasVAE layout models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum, auto
from typing import ClassVar, Final

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


class CanvasVAECrelloConfig(PretrainedConfig):
    """Configuration for CanvasVAE on the original Crello document schema.

    Args:
        vocabularies: Source-ordered Crello lookup tables. String tables start
            with the empty mask token; ``length`` is the inclusive range 1–50.
        latent_dim: Hidden and latent dimension. The released Crello command
            uses 512.
        num_blocks: Number of encoder and decoder sequence blocks.
        num_heads: Number of attention heads.
        dropout: Position and sequence-block dropout.
        kl_weight: KL divergence coefficient. The released Crello command uses
            32.
        l2_weight: Squared-norm penalty on Dense and Embedding parameters.
        layer_norm_epsilon: Layer-normalization epsilon.
        batch_norm_epsilon: Batch-normalization epsilon.
        batch_norm_momentum: Previous-statistic weight in batch normalization.
        **kwargs: Extra ``PretrainedConfig`` fields.
    """

    model_type = "canvas-vae-crello"
    context_fields: ClassVar[tuple[str, ...]] = (
        "length",
        "group",
        "format",
        "canvas_width",
        "canvas_height",
        "category",
    )
    sequence_fields: ClassVar[tuple[str, ...]] = (
        "type",
        "left",
        "top",
        "width",
        "height",
        "opacity",
        "color",
        "image_embedding",
    )
    conditional_types: ClassVar[dict[str, tuple[str, ...]]] = {
        "color": ("textElement", "coloredBackground"),
        "image_embedding": ("svgElement", "imageElement", "maskElement"),
    }

    def __init__(
        self,
        vocabularies: Mapping[str, Sequence[str | int]] | None = None,
        max_length: int = 50,
        latent_dim: int = 512,
        num_blocks: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        kl_weight: float = 32.0,
        l2_weight: float = 1e-6,
        layer_norm_epsilon: float = 1e-3,
        batch_norm_epsilon: float = 1e-3,
        batch_norm_momentum: float = 0.99,
        **kwargs: str | float | bool | None,
    ) -> None:
        """Initialize the configuration with the Crello v1 recipe defaults."""
        tables = {
            str(key): [
                str(token) if isinstance(token, str) else int(token) for token in values
            ]
            for key, values in (vocabularies or {}).items()
        }
        type_values = [str(value) for value in tables.get("type", [])]
        kwargs.pop("id2label", None)
        kwargs.pop("label2id", None)
        super().__init__(
            id2label=dict(enumerate(type_values)),
            label2id={token: index for index, token in enumerate(type_values)},
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
        self.vocabularies = tables
        self.max_length = max_length
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
    def context_field_sizes(self) -> dict[str, int]:
        """Return categorical context sizes, including the length head."""
        missing = [
            field for field in self.context_fields[1:] if field not in self.vocabularies
        ]
        if missing:
            raise ValueError(
                f"CanvasVAECrelloConfig is missing vocabularies: {missing}"
            )

        if self.vocabularies.get("length", list(range(1, self.max_length + 1))) != list(
            range(1, self.max_length + 1)
        ):
            raise ValueError(
                f"Crello length vocabulary must be the inclusive range 1–{self.max_length}"
            )

        return {
            "length": self.max_length,
            **{
                field: len(self.vocabularies[field])
                for field in self.context_fields[1:]
            },
        }

    @property
    def sequence_field_sizes(self) -> dict[str, int]:
        """Return the categorical class count of every Crello sequence field."""
        missing = [field for field in ("type",) if field not in self.vocabularies]
        if missing:
            raise ValueError(
                f"CanvasVAECrelloConfig is missing vocabularies: {missing}"
            )

        return {
            "type": len(self.vocabularies["type"]),
            "left": 64,
            "top": 64,
            "width": 64,
            "height": 64,
            "opacity": 8,
            "color": 16,
        }

    @property
    def primary_label_id(self) -> int:
        """Return the empty ``type`` id used to paint the background grid."""
        return self.vocabularies["type"].index(PRIMARY_LABEL_TOKEN)

    def conditional_type_ids(self, field: str) -> tuple[int, ...]:
        """Return source-vocabulary ids that enable a conditional field."""
        try:
            values = self.conditional_types[field]
        except KeyError as error:
            raise ValueError(f"unknown conditional Crello field: {field}") from error

        vocabulary = self.vocabularies.get("type", [])
        missing = [value for value in values if value not in vocabulary]
        if missing:
            raise ValueError(
                f"Crello type vocabulary is missing conditional types: {missing}"
            )

        return tuple(vocabulary.index(value) for value in values)


__all__ = [
    "CanvasVAEConfig",
    "CanvasVAECrelloConfig",
    "CanvasVAEField",
    "GEOMETRY_FIELDS",
    "VOCABULARY_FIELDS",
]

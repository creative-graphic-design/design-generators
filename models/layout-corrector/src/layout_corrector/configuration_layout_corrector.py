"""Configuration objects for Layout-Corrector models."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Final

from diffusers.configuration_utils import ConfigMixin, register_to_config

from laygen.common.labels import (
    DatasetName,
    id2label_for_dataset,
    normalize_dataset_name,
)

from .sampling import CorrectorMaskMode, normalize_corrector_mask_mode

CRELLO_BBOX_DATASET: Final[str] = "crello-bbox"


class CorrectorReconType(StrEnum):
    """Supported Layout-Corrector reconstruction targets."""

    x_0 = auto()
    x_t_minus_1 = "x_t-1"


class CorrectorTarget(StrEnum):
    """Supported Layout-Corrector confidence targets."""

    mask = auto()
    recon_acc = auto()


class CorrectorPositionEmbedding(StrEnum):
    """Supported original position-embedding modes."""

    default = auto()
    none = auto()
    pos_enc = auto()
    shuffle_pos_enc = auto()
    shuffle = auto()


class CorrectorTransformerType(StrEnum):
    """Supported corrector transformer variants."""

    aggregated = auto()


def normalize_corrector_core_options(
    recon_type: CorrectorReconType | str,
    target: CorrectorTarget | str,
    transformer_type: CorrectorTransformerType | str,
    pos_emb: CorrectorPositionEmbedding | str,
) -> tuple[
    CorrectorReconType,
    CorrectorTarget,
    CorrectorTransformerType,
    CorrectorPositionEmbedding,
]:
    """Normalize shared Layout-Corrector enum options."""
    try:
        normalized_recon_type = CorrectorReconType(recon_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported recon_type: {recon_type}") from exc
    try:
        normalized_target = CorrectorTarget(target)
    except ValueError as exc:
        raise ValueError(f"Unsupported target: {target}") from exc
    try:
        normalized_transformer_type = CorrectorTransformerType(transformer_type)
    except ValueError as exc:
        raise ValueError("Only transformer_type='aggregated' is supported") from exc
    try:
        normalized_pos_emb = CorrectorPositionEmbedding(pos_emb)
    except ValueError as exc:
        raise ValueError(f"Unsupported pos_emb: {pos_emb}") from exc
    return (
        normalized_recon_type,
        normalized_target,
        normalized_transformer_type,
        normalized_pos_emb,
    )


class LayoutCorrectorConfig(ConfigMixin):
    """Configuration for the Layout-Corrector transformer.

    Args:
        dataset_name: Dataset key or alias used for labels.
        vocab_size: LayoutDM vocabulary size expected by the corrector.
        id2label: Optional class-id mapping. When omitted, the shared registry is used.
        max_seq_length: Maximum number of layout elements.
        num_attributes_per_element: Number of token attributes per element.
        hidden_size: Transformer hidden dimension.
        num_attention_heads: Number of attention heads.
        num_hidden_layers: Number of transformer layers.
        intermediate_size: Feed-forward hidden dimension.
        dropout: Dropout probability.
        timestep_type: Timestep conditioning type.
        num_timesteps: Number of diffusion training timesteps.
        recon_type: Reconstruction target used by the corrector.
        target: Confidence target type.
        attr_loss_weights: Per-attribute loss weights.
        use_padding_as_vocab: Whether padding is part of the modeled vocabulary.
        pos_emb: Position embedding mode from the original implementation.
        transformer_type: Corrector transformer variant.
        corrector_steps: Number of correction passes per selected timestep.
        corrector_t_list: Explicit timesteps where the corrector is applied.
        corrector_mask_mode: Strategy for selecting tokens to remask.
        corrector_mask_threshold: Confidence threshold for threshold masking.
        corrector_temperature: Temperature used for corrector resampling.
        use_gumbel_noise: Whether to perturb confidence logits.
        gumbel_temperature: Temperature for confidence Gumbel noise.
        time_adaptive_temperature: Whether to scale noise by timestep ratio.

    Raises:
        ValueError: If a supplied dataset, shape, or option is unsupported.

    Examples:
        >>> cfg = LayoutCorrectorConfig(dataset_name="publaynet", vocab_size=100)
        >>> cfg.max_token_length
        125
    """

    config_name = "corrector_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        dataset_name: DatasetName | str,
        vocab_size: int,
        id2label: dict[int | str, str] | None = None,
        max_seq_length: int = 25,
        num_attributes_per_element: int = 5,
        hidden_size: int = 464,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 4,
        intermediate_size: int = 1856,
        dropout: float = 0.0,
        timestep_type: str | None = "adalayernorm",
        num_timesteps: int = 100,
        recon_type: CorrectorReconType | str = CorrectorReconType.x_t_minus_1,
        target: CorrectorTarget | str = CorrectorTarget.recon_acc,
        attr_loss_weights: tuple[float, float, float, float, float] = (
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ),
        use_padding_as_vocab: bool = True,
        pos_emb: CorrectorPositionEmbedding | str = CorrectorPositionEmbedding.none,
        transformer_type: CorrectorTransformerType | str = (
            CorrectorTransformerType.aggregated
        ),
        corrector_steps: int = 1,
        corrector_t_list: tuple[int, ...] = (10, 20, 30),
        corrector_mask_mode: CorrectorMaskMode | str = CorrectorMaskMode.thresh,
        corrector_mask_threshold: float = 0.7,
        corrector_temperature: float = 1.0,
        use_gumbel_noise: bool = True,
        gumbel_temperature: float = 1.0,
        time_adaptive_temperature: bool = False,
    ) -> None:
        """Initialize a Layout-Corrector config.

        Args:
            dataset_name: Dataset key or alias used for labels.
            vocab_size: LayoutDM vocabulary size expected by the corrector.
            id2label: Optional class-id mapping.
            max_seq_length: Maximum number of layout elements.
            num_attributes_per_element: Number of token attributes per element.
            hidden_size: Transformer hidden dimension.
            num_attention_heads: Number of attention heads.
            num_hidden_layers: Number of transformer layers.
            intermediate_size: Feed-forward hidden dimension.
            dropout: Dropout probability.
            timestep_type: Timestep conditioning type.
            num_timesteps: Number of diffusion training timesteps.
            recon_type: Reconstruction target used by the corrector.
            target: Confidence target type.
            attr_loss_weights: Per-attribute loss weights.
            use_padding_as_vocab: Whether padding is part of the modeled vocabulary.
            pos_emb: Position embedding mode.
            transformer_type: Corrector transformer variant.
            corrector_steps: Number of correction passes per selected timestep.
            corrector_t_list: Explicit timesteps where the corrector is applied.
            corrector_mask_mode: Strategy for selecting tokens to remask.
            corrector_mask_threshold: Confidence threshold for threshold masking.
            corrector_temperature: Temperature used for corrector resampling.
            use_gumbel_noise: Whether to perturb confidence logits.
            gumbel_temperature: Temperature for confidence Gumbel noise.
            time_adaptive_temperature: Whether to scale noise by timestep ratio.

        Raises:
            ValueError: If a supplied dataset, shape, or option is unsupported.

        Examples:
            >>> cfg = LayoutCorrectorConfig(dataset_name="publaynet", vocab_size=100)
            >>> cfg.max_token_length
            125
        """
        try:
            dataset_name = str(normalize_dataset_name(dataset_name))
        except ValueError:
            if id2label is None:
                raise
            dataset_name = str(dataset_name)
        self.register_to_config(dataset_name=dataset_name)
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if num_attributes_per_element != 5:
            raise ValueError("Layout-Corrector supports 5 attributes per element")
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        recon_type, target, transformer_type, pos_emb = (
            normalize_corrector_core_options(
                recon_type,
                target,
                transformer_type,
                pos_emb,
            )
        )
        if len(attr_loss_weights) != num_attributes_per_element:
            raise ValueError("attr_loss_weights must match num_attributes_per_element")
        if corrector_steps <= 0:
            raise ValueError("corrector_steps must be positive")
        try:
            corrector_mask_mode = normalize_corrector_mask_mode(corrector_mask_mode)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported corrector_mask_mode: {corrector_mask_mode}"
            ) from exc
        if not 0.0 <= corrector_mask_threshold <= 1.0:
            raise ValueError("corrector_mask_threshold must be in [0, 1]")
        self.register_to_config(
            dataset_name=dataset_name,
            recon_type=str(recon_type),
            target=str(target),
            pos_emb=str(pos_emb),
            transformer_type=str(transformer_type),
            corrector_mask_mode=str(corrector_mask_mode),
        )

        self.dataset_name = dataset_name
        raw_id2label = id2label or id2label_for_dataset(dataset_name)
        self.id2label = {int(k): v for k, v in raw_id2label.items()}
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.num_attributes_per_element = num_attributes_per_element
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.timestep_type = timestep_type
        self.num_timesteps = num_timesteps
        self.recon_type = str(recon_type)
        self.target = str(target)
        self.attr_loss_weights = tuple(float(v) for v in attr_loss_weights)
        self.use_padding_as_vocab = use_padding_as_vocab
        self.pos_emb = str(pos_emb)
        self.transformer_type = str(transformer_type)
        self.corrector_steps = corrector_steps
        self.corrector_t_list = tuple(int(v) for v in corrector_t_list)
        self.corrector_mask_mode = str(corrector_mask_mode)
        self.corrector_mask_threshold = corrector_mask_threshold
        self.corrector_temperature = corrector_temperature
        self.use_gumbel_noise = use_gumbel_noise
        self.gumbel_temperature = gumbel_temperature
        self.time_adaptive_temperature = time_adaptive_temperature

    @property
    def max_token_length(self) -> int:
        """Return the flattened token length for one layout sequence.

        Returns:
            Maximum token count after flattening element attributes.

        Examples:
            >>> LayoutCorrectorConfig(dataset_name="publaynet", vocab_size=100).max_token_length
            125
        """
        return self.max_seq_length * self.num_attributes_per_element

"""Layout-Corrector confidence model components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from jaxtyping import Bool, Float, Int
from torch import nn

from laygen.nn import (
    Block,
    ElementPositionalEmbedding,
    TimestepEmbeddingType,
    TransformerEncoder,
)
from laygen.common.labels import id2label_for_dataset, normalize_dataset_name

from .configuration_layout_corrector import (
    CorrectorPositionEmbedding,
    CorrectorReconType,
    CorrectorTarget,
    CorrectorTransformerType,
)
from .sampling import CorrectorMaskMode, normalize_corrector_mask_mode


@dataclass
class LayoutCorrectorOutput(BaseOutput):
    """Output container for Layout-Corrector confidence logits.

    Args:
        logits: Token confidence logits shaped `(batch, tokens)`.
    """

    logits: Float[torch.Tensor, "batch tokens"]


class AggregatedCategoricalTransformer(nn.Module):
    """Aggregated-token transformer used by the original Layout-Corrector model."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_token_length: int,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        intermediate_size: int,
        dropout: float,
        timestep_type: TimestepEmbeddingType | str | None,
        pos_emb: CorrectorPositionEmbedding | str,
        num_attributes_per_element: int,
        num_timesteps: int,
    ) -> None:
        """Initialize the aggregated categorical transformer.

        Args:
            vocab_size: Number of token ids.
            max_token_length: Flattened token sequence length.
            hidden_size: Transformer hidden dimension.
            num_attention_heads: Number of attention heads.
            num_hidden_layers: Number of transformer layers.
            intermediate_size: Feed-forward hidden dimension.
            dropout: Dropout probability.
            timestep_type: Timestep conditioning type.
            pos_emb: Position embedding mode.
            num_attributes_per_element: Number of attributes per layout element.
            num_timesteps: Diffusion timestep count.

        Raises:
            ValueError: If `max_token_length` is not divisible by attributes per
                element.
        """
        super().__init__()
        if max_token_length % num_attributes_per_element:
            raise ValueError(
                "max_token_length must divide by num_attributes_per_element"
            )
        self.num_attributes_per_element = num_attributes_per_element
        self.cat_emb = nn.Embedding(vocab_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.enc = nn.Sequential(
            nn.Linear(num_attributes_per_element * hidden_size, hidden_size),
            nn.ReLU(),
        )
        layer = Block(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            diffusion_step=num_timesteps,
            timestep_type=timestep_type,
        )
        self.backbone = TransformerEncoder(layer, num_hidden_layers)
        self.dec = nn.Sequential(
            nn.Linear(hidden_size, num_attributes_per_element * hidden_size),
            nn.ReLU(),
        )
        self.pos_emb = None
        if CorrectorPositionEmbedding(pos_emb) is not CorrectorPositionEmbedding.none:
            self.pos_emb = ElementPositionalEmbedding(
                hidden_size,
                max_token_length // num_attributes_per_element,
                n_attr_per_elem=1,
            )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        *,
        timestep: Int[torch.Tensor, "batch"] | None = None,
        src_key_padding_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
    ) -> Float[torch.Tensor, "batch tokens 1"]:
        """Predict token confidence logits.

        Args:
            input_ids: Flattened token ids.
            timestep: Optional diffusion timestep tensor.
            src_key_padding_mask: Optional padding mask.

        Returns:
            Confidence logits shaped `(batch, tokens, 1)`.
        """
        batch_size, token_length = input_ids.shape
        step = self.num_attributes_per_element
        hidden = self.drop(self.cat_emb(input_ids))
        hidden = hidden.reshape(
            batch_size, token_length // step, step * hidden.size(-1)
        )
        hidden = self.enc(hidden)
        if self.pos_emb is not None:
            hidden = hidden + self.pos_emb(hidden)
        if src_key_padding_mask is not None:
            element_padding_mask = src_key_padding_mask.reshape(
                batch_size, token_length // step, step
            ).any(dim=-1)
        else:
            element_padding_mask = None
        hidden = self.backbone(
            hidden,
            src_key_padding_mask=element_padding_mask,
            timestep=timestep,
        )
        hidden = self.dec(hidden)
        hidden = hidden.reshape(batch_size, token_length, -1)
        return self.head(hidden)


class LayoutCorrectorModel(ModelMixin, ConfigMixin):
    """Diffusers-compatible Layout-Corrector confidence model."""

    config_name = "corrector_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        dataset_name: str,
        vocab_size: int,
        id2label: dict[int | str, str] | None = None,
        max_seq_length: int = 25,
        num_attributes_per_element: int = 5,
        hidden_size: int = 464,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 4,
        intermediate_size: int = 1856,
        dropout: float = 0.0,
        timestep_type: TimestepEmbeddingType | str | None = "adalayernorm",
        num_timesteps: int = 100,
        recon_type: CorrectorReconType | str = CorrectorReconType.x_t_minus_1,
        target: CorrectorTarget | str = CorrectorTarget.recon_acc,
        attr_loss_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
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
        """Initialize a Layout-Corrector model.

        Args:
            dataset_name: Dataset key or alias used for labels.
            vocab_size: LayoutDM vocabulary size.
            id2label: Optional class-id mapping.
            max_seq_length: Maximum number of elements.
            num_attributes_per_element: Number of token attributes per element.
            hidden_size: Transformer hidden dimension.
            num_attention_heads: Number of attention heads.
            num_hidden_layers: Number of transformer layers.
            intermediate_size: Feed-forward hidden dimension.
            dropout: Dropout probability.
            timestep_type: Timestep conditioning type.
            num_timesteps: Number of diffusion timesteps.
            recon_type: Reconstruction target.
            target: Confidence target type.
            attr_loss_weights: Per-attribute loss weights.
            use_padding_as_vocab: Whether padding is modeled as a vocabulary token.
            pos_emb: Position embedding mode.
            transformer_type: Corrector transformer type.
            corrector_steps: Number of correction passes.
            corrector_t_list: Explicit correction timesteps.
            corrector_mask_mode: Token remasking mode.
            corrector_mask_threshold: Threshold for confidence remasking.
            corrector_temperature: Corrector sampling temperature.
            use_gumbel_noise: Whether confidence logits receive Gumbel noise.
            gumbel_temperature: Confidence-noise temperature.
            time_adaptive_temperature: Whether to scale noise by timestep ratio.

        Raises:
            ValueError: If reconstruction, target, or transformer options are
                unsupported.
        """
        super().__init__()
        try:
            recon_type = CorrectorReconType(recon_type)
        except ValueError as exc:
            raise ValueError(f"Unsupported recon_type: {recon_type}") from exc
        try:
            target = CorrectorTarget(target)
        except ValueError as exc:
            raise ValueError(f"Unsupported target: {target}") from exc
        try:
            transformer_type = CorrectorTransformerType(transformer_type)
        except ValueError as exc:
            raise ValueError("Only transformer_type='aggregated' is supported") from exc
        try:
            pos_emb = CorrectorPositionEmbedding(pos_emb)
        except ValueError as exc:
            raise ValueError(f"Unsupported pos_emb: {pos_emb}") from exc
        try:
            corrector_mask_mode = normalize_corrector_mask_mode(corrector_mask_mode)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported corrector_mask_mode: {corrector_mask_mode}"
            ) from exc
        try:
            dataset_name = str(normalize_dataset_name(dataset_name))
        except ValueError:
            if id2label is None:
                raise
            dataset_name = str(dataset_name)
        normalized_id2label = {
            int(k): v
            for k, v in (id2label or id2label_for_dataset(dataset_name)).items()
        }
        self.register_to_config(
            dataset_name=dataset_name,
            id2label=normalized_id2label,
            corrector_t_list=tuple(corrector_t_list),
            attr_loss_weights=tuple(attr_loss_weights),
            recon_type=str(recon_type),
            target=str(target),
            pos_emb=str(pos_emb),
            transformer_type=str(transformer_type),
            corrector_mask_mode=str(corrector_mask_mode),
        )
        self.vocab_size = vocab_size
        self.id2label = normalized_id2label
        self.recon_type = recon_type
        self.corrector_steps = corrector_steps
        self.corrector_t_list = tuple(corrector_t_list)
        self.corrector_mask_mode = corrector_mask_mode
        self.corrector_mask_threshold = corrector_mask_threshold
        self.corrector_temperature = corrector_temperature
        self.use_padding_as_vocab = use_padding_as_vocab
        self.use_gumbel_noise = use_gumbel_noise
        self.gumbel_temperature = gumbel_temperature
        self.time_adaptive_temperature = time_adaptive_temperature
        self.model = AggregatedCategoricalTransformer(
            vocab_size=vocab_size,
            max_token_length=max_seq_length * num_attributes_per_element,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            dropout=dropout,
            timestep_type=timestep_type,
            pos_emb=pos_emb,
            num_attributes_per_element=num_attributes_per_element,
            num_timesteps=num_timesteps,
        )

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        timesteps: Int[torch.Tensor, "batch"],
        padding_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
    ) -> LayoutCorrectorOutput:
        """Run the corrector model.

        Args:
            input_ids: Flattened token ids.
            timesteps: Diffusion timestep tensor.
            padding_mask: Optional padding mask.

        Returns:
            `LayoutCorrectorOutput` containing token confidence logits.
        """
        src_key_padding_mask = None if self.use_padding_as_vocab else padding_mask
        logits = self.model(
            input_ids,
            timestep=timesteps,
            src_key_padding_mask=src_key_padding_mask,
        ).squeeze(-1)
        if not self.use_padding_as_vocab and padding_mask is not None:
            logits = logits.masked_fill(padding_mask, 1000.0)
        return LayoutCorrectorOutput(logits=logits)

    def calc_confidence_score(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        timesteps: Int[torch.Tensor, "batch"],
        padding_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
    ) -> Float[torch.Tensor, "batch tokens"]:
        """Return confidence logits for token remasking.

        Args:
            input_ids: Flattened token ids.
            timesteps: Diffusion timestep tensor.
            padding_mask: Optional padding mask.

        Returns:
            Confidence logits shaped `(batch, tokens)`.
        """
        return self(
            input_ids=input_ids,
            timesteps=timesteps,
            padding_mask=padding_mask,
        ).logits

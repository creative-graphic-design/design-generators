"""CanvasVAE encoder-decoder model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from laygen.common.randomness import randn
from laygen.nn.attention import MultiHeadSelfAttention
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_canvas_vae import CanvasVAEConfig, CanvasVAEField

EMBEDDING_INIT_RANGE = 0.05


@dataclass
class CanvasVAEModelOutput(ModelOutput):
    """Outputs of :class:`CanvasVAEModel`.

    Attributes:
        loss: Reconstruction plus weighted KL loss; present when layout inputs
            are given and the model is in training mode.
        length_logits: Logits over the zero-based element count.
        element_logits: Logits of every per-element field, keyed by field name.
        mask: Valid-element mask used by the decoder.
        latents: Latent codes fed to the decoder.
        z_mean: Posterior mean; present when layout inputs are given.
        z_log_var: Posterior log-variance; present when layout inputs are given.
        kl_divergence: Unweighted KL divergence to the standard normal prior.
        reconstruction_losses: Per-field reconstruction losses, including
            ``length``; present together with ``loss``.
    """

    loss: Float[torch.Tensor, ""] | None = None
    length_logits: Float[torch.Tensor, "batch lengths"] | None = None
    element_logits: dict[str, Float[torch.Tensor, "batch elements classes"]] | None = (
        None
    )
    mask: Bool[torch.Tensor, "batch elements"] | None = None
    latents: Float[torch.Tensor, "batch latent"] | None = None
    z_mean: Float[torch.Tensor, "batch latent"] | None = None
    z_log_var: Float[torch.Tensor, "batch latent"] | None = None
    kl_divergence: Float[torch.Tensor, ""] | None = None
    reconstruction_losses: dict[str, Float[torch.Tensor, ""]] | None = None


def length_mask(
    length_ids: Int[torch.Tensor, batch], max_elements: int | None = None
) -> Bool[torch.Tensor, "batch elements"]:
    """Build a valid-element mask from zero-based element-count ids.

    Args:
        length_ids: Element count minus one for every layout.
        max_elements: Mask width. Defaults to the longest layout in the batch.

    Returns:
        Mask where ``True`` marks a valid element.

    Examples:
        >>> length_mask(torch.tensor([0, 2])).tolist()
        [[True, False, False], [True, True, True]]
    """
    lengths = length_ids + 1
    width = int(lengths.max()) if max_elements is None else max_elements
    positions = torch.arange(width, device=length_ids.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


class CanvasVAEBatchNorm(nn.Module):
    """Batch normalization over latent features with population variance.

    Training mode normalizes with the batch mean and the biased batch variance
    and moves both moving statistics toward them; evaluation mode uses the
    moving statistics.

    Args:
        num_features: Feature dimension.
        eps: Variance epsilon.
        momentum: Moving-average momentum of the previous statistic.
    """

    def __init__(self, num_features: int, eps: float, momentum: float) -> None:
        """Create parameters and moving statistics."""
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(
        self, inputs: Float[torch.Tensor, "batch features"]
    ) -> Float[torch.Tensor, "batch features"]:
        """Normalize ``inputs``.

        Args:
            inputs: Pooled features.

        Returns:
            Normalized features.
        """
        if self.training:
            mean = inputs.mean(dim=0)
            variance = (inputs - mean.detach()).square().mean(dim=0)
            with torch.no_grad():
                decay = 1.0 - self.momentum
                self.running_mean.sub_((self.running_mean - mean) * decay)
                self.running_var.sub_((self.running_var - variance) * decay)
        else:
            mean = self.running_mean
            variance = self.running_var

        inverse = torch.rsqrt(variance + self.eps) * self.weight
        return inputs * inverse + (self.bias - mean * inverse)


class CanvasVAEBlock(nn.Module):
    """Pre-norm transformer block conditioned on a global vector.

    Args:
        config: Model config.
        pooling: Whether to return the masked mean of ReLU outputs.
    """

    def __init__(self, config: CanvasVAEConfig, *, pooling: bool) -> None:
        """Create attention, conditioning, and feed-forward layers."""
        super().__init__()
        dim = config.latent_dim
        self.pooling = pooling
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_epsilon)
        self.attention = MultiHeadSelfAttention(dim, config.num_heads)
        self.dropout1 = nn.Dropout(config.dropout)
        self.conditional = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim)
        )
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch elements dim"],
        condition: Float[torch.Tensor, "batch dim"],
        mask: Bool[torch.Tensor, "batch elements"],
    ) -> Float[torch.Tensor, "batch elements dim"] | Float[torch.Tensor, "batch dim"]:
        """Apply the block.

        Args:
            hidden_states: Element sequence.
            condition: Global conditioning vector.
            mask: Valid-element mask.

        Returns:
            Updated sequence, or its masked mean when pooling.
        """
        attended = self.attention(self.norm1(hidden_states), mask)
        hidden_states = hidden_states + self.dropout1(attended)
        hidden_states = hidden_states + self.conditional(condition).unsqueeze(1)
        hidden_states = hidden_states + self.dropout2(
            self.mlp(self.norm2(hidden_states))
        )
        if not self.pooling:
            return hidden_states

        weights = mask.to(hidden_states.dtype).unsqueeze(-1)
        pooled = (torch.relu(hidden_states) * weights).sum(dim=1)
        return pooled / weights.sum(dim=1)


class CanvasVAEPositionEmbedding(nn.Module):
    """Learned position embeddings with dropout.

    Args:
        config: Model config.
    """

    def __init__(self, config: CanvasVAEConfig) -> None:
        """Create the embedding table."""
        super().__init__()
        self.embeddings = nn.Embedding(config.max_length, config.latent_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, mask: Bool[torch.Tensor, "batch elements"]
    ) -> Float[torch.Tensor, "batch elements dim"]:
        """Return dropout-applied position embeddings shaped like ``mask``.

        Args:
            mask: Valid-element mask whose shape selects the output shape.

        Returns:
            Position embeddings tiled over the batch.
        """
        batch, elements = mask.shape
        weight = self.embeddings.weight[:elements]
        return self.dropout(weight.unsqueeze(0).expand(batch, -1, -1))


class CanvasVAEEncoder(nn.Module):
    """Encode a layout into posterior statistics.

    Args:
        config: Model config.
    """

    def __init__(self, config: CanvasVAEConfig) -> None:
        """Create input embeddings, blocks, normalization, and the head."""
        super().__init__()
        dim = config.latent_dim
        self.length_embedding = nn.Embedding(config.max_length, dim)
        self.position_embedding = CanvasVAEPositionEmbedding(config)
        self.field_embeddings = nn.ModuleDict(
            {key: nn.Embedding(size, dim) for key, size in config.field_sizes.items()}
        )
        self.blocks = nn.ModuleList(
            CanvasVAEBlock(config, pooling=index == config.num_blocks - 1)
            for index in range(config.num_blocks)
        )
        self.norm = CanvasVAEBatchNorm(
            dim, config.batch_norm_epsilon, config.batch_norm_momentum
        )
        self.z_mean = nn.Linear(dim, dim)
        self.z_log_var = nn.Linear(dim, dim)

    def forward(
        self,
        length_ids: Int[torch.Tensor, batch],
        element_ids: Int[torch.Tensor, "batch elements fields"],
    ) -> tuple[Float[torch.Tensor, "batch dim"], Float[torch.Tensor, "batch dim"]]:
        """Return the posterior mean and log-variance.

        Args:
            length_ids: Zero-based element counts.
            element_ids: Field ids in :class:`CanvasVAEField` order.

        Returns:
            Posterior mean and log-variance.
        """
        mask = length_mask(length_ids, element_ids.shape[1])
        condition = self.length_embedding(length_ids)
        hidden_states = self.position_embedding(mask)
        for index, embedding in enumerate(self.field_embeddings.values()):
            hidden_states = hidden_states + embedding(element_ids[..., index])

        for block in self.blocks:
            hidden_states = block(hidden_states, condition, mask)

        pooled = self.norm(hidden_states)
        return self.z_mean(pooled), self.z_log_var(pooled)


class CanvasVAEDecoder(nn.Module):
    """Decode latent codes into per-element field logits in one pass.

    Args:
        config: Model config.
    """

    def __init__(self, config: CanvasVAEConfig) -> None:
        """Create position embeddings, blocks, and output heads."""
        super().__init__()
        dim = config.latent_dim
        self.position_embedding = CanvasVAEPositionEmbedding(config)
        self.blocks = nn.ModuleList(
            CanvasVAEBlock(config, pooling=False) for _ in range(config.num_blocks)
        )
        self.length_head = nn.Linear(dim, config.max_length)
        self.field_heads = nn.ModuleDict(
            {key: nn.Linear(dim, size) for key, size in config.field_sizes.items()}
        )

    def forward(
        self,
        latents: Float[torch.Tensor, "batch dim"],
        mask: Bool[torch.Tensor, "batch elements"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "batch lengths"],
        dict[str, Float[torch.Tensor, "batch elements classes"]],
        Bool[torch.Tensor, "batch elements"],
    ]:
        """Decode latent codes.

        Args:
            latents: Latent codes.
            mask: Element mask to decode with. Defaults to the mask of the
                predicted element counts.

        Returns:
            Length logits, per-field logits, and the decoding mask.
        """
        length_logits = self.length_head(latents)
        if mask is None:
            mask = length_mask(length_logits.argmax(dim=-1))

        hidden_states = self.position_embedding(mask)
        for block in self.blocks:
            hidden_states = block(hidden_states, latents, mask)

        logits = {key: head(hidden_states) for key, head in self.field_heads.items()}
        return length_logits, logits, mask


def _sequence_cross_entropy(
    logits: Float[torch.Tensor, "batch elements classes"],
    targets: Int[torch.Tensor, "batch elements"],
    mask: Bool[torch.Tensor, "batch elements"],
) -> Float[torch.Tensor, ""]:
    losses = nn.functional.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none"
    )
    return (losses * mask.to(losses.dtype)).sum(dim=1).mean()


class CanvasVAEPreTrainedModel(PreTrainedModel):
    """Base class handling CanvasVAE weight initialization."""

    config_class = CanvasVAEConfig
    base_model_prefix = "canvas_vae"
    main_input_name = "element_ids"

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize Glorot-uniform kernels, zero biases, and uniform embeddings.

        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -EMBEDDING_INIT_RANGE, EMBEDDING_INIT_RANGE)
        elif isinstance(module, (nn.LayerNorm, CanvasVAEBatchNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        if isinstance(module, CanvasVAEBatchNorm):
            module.running_mean.zero_()
            module.running_var.fill_(1.0)


class CanvasVAEModel(CanvasVAEPreTrainedModel):
    """CanvasVAE variational autoencoder over layout element sequences.

    Given layout ids, the model encodes them, takes a posterior sample in
    training mode (or when ``posterior_noise`` is given) and the posterior mean
    otherwise, and decodes with the ground-truth element mask in training mode
    or the predicted element count in evaluation mode. Given only
    ``latents``, it decodes them with the predicted element count.

    Args:
        config: Model config with vocabularies.

    Examples:
        >>> config = CanvasVAEConfig(
        ...     vocabularies={
        ...         "component": ["[UNK]", "", "Text"],
        ...         "icon": ["[UNK]", ""],
        ...         "text_button": ["[UNK]", ""],
        ...     },
        ...     max_length=4,
        ...     num_bins=8,
        ...     latent_dim=16,
        ...     num_heads=2,
        ... )
        >>> model = CanvasVAEModel(config).eval()
        >>> output = model(latents=torch.zeros(2, 16))
        >>> output.length_logits.shape
        torch.Size([2, 4])
    """

    def __init__(self, config: CanvasVAEConfig) -> None:
        """Create the encoder and decoder."""
        super().__init__(config)
        self.encoder = CanvasVAEEncoder(config)
        self.decoder = CanvasVAEDecoder(config)
        self.post_init()

    def forward(
        self,
        num_elements: Int[torch.Tensor, batch] | None = None,
        element_ids: Int[torch.Tensor, "batch elements fields"] | None = None,
        *,
        latents: Float[torch.Tensor, "batch dim"] | None = None,
        posterior_noise: Float[torch.Tensor, "batch dim"] | None = None,
    ) -> CanvasVAEModelOutput:
        """Encode and decode a layout batch, or decode latent codes.

        Args:
            num_elements: Element count of every layout.
            element_ids: Field ids in :class:`CanvasVAEField` order, padded to
                the longest layout in the batch.
            latents: Latent codes to decode instead of encoding a layout.
            posterior_noise: Standard-normal noise for the posterior sample.
                Drawn from the global generator in training mode when omitted.

        Returns:
            Model outputs; ``loss`` is set only in training mode.

        Raises:
            ValueError: If neither a layout nor latent codes are given.
        """
        if latents is not None:
            length_logits, logits, mask = self.decoder(latents)
            return CanvasVAEModelOutput(
                length_logits=length_logits,
                element_logits=logits,
                mask=mask,
                latents=latents,
            )

        if num_elements is None or element_ids is None:
            raise ValueError("pass num_elements and element_ids, or latents")

        length_ids = num_elements - 1
        z_mean, z_log_var = self.encoder(length_ids, element_ids)
        kl_divergence = -0.5 * torch.mean(
            1 + z_log_var - z_mean.square() - torch.exp(z_log_var)
        )
        if posterior_noise is None and self.training:
            posterior_noise = randn(
                z_mean.shape, device=z_mean.device, dtype=z_mean.dtype
            )

        latents = z_mean
        if posterior_noise is not None:
            latents = z_mean + torch.exp(0.5 * z_log_var) * posterior_noise

        mask = length_mask(length_ids, element_ids.shape[1]) if self.training else None
        length_logits, logits, mask = self.decoder(latents, mask)
        output = CanvasVAEModelOutput(
            length_logits=length_logits,
            element_logits=logits,
            mask=mask,
            latents=latents,
            z_mean=z_mean,
            z_log_var=z_log_var,
            kl_divergence=kl_divergence,
        )
        if not self.training:
            return output

        losses = {
            "length": nn.functional.cross_entropy(length_logits, length_ids),
            **{
                key: _sequence_cross_entropy(logits[key], element_ids[..., index], mask)
                for index, key in enumerate(CanvasVAEField)
            },
        }
        reconstruction = sum(losses.values(), torch.zeros((), device=z_mean.device))
        output.reconstruction_losses = losses
        output.loss = reconstruction + self.config.kl_weight * kl_divergence
        return output


__all__ = ["CanvasVAEModel", "CanvasVAEModelOutput", "length_mask"]

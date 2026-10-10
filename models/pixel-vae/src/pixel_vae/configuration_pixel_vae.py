"""PixelVAE model configuration."""

from __future__ import annotations

from transformers import PretrainedConfig


class PixelVAEConfig(PretrainedConfig):
    """Configuration for the 256 × 256 RGBA MobileNetV2 VAE.

    Args:
        latent_dim: Dimension of each posterior vector.
        quantize_factor: Number of low bits removed from each target channel.
        kl_weight: Multiplier on the mean KL divergence.
        l2_weight: L2 coefficient for the two variational-head dense layers.
        batch_norm_epsilon: Variance epsilon used by batch normalization.
        backbone_batch_norm_momentum: Moving-average momentum in MobileNetV2.
        decoder_batch_norm_momentum: Moving-average momentum in the decoder.
        **kwargs: Extra Transformers configuration fields.

    Examples:
        >>> PixelVAEConfig().latent_dim
        256
    """

    model_type = "pixel-vae"

    def __init__(
        self,
        *,
        latent_dim: int = 256,
        quantize_factor: int = 4,
        kl_weight: float = 100.0,
        l2_weight: float = 1e-6,
        batch_norm_epsilon: float = 1e-3,
        backbone_batch_norm_momentum: float = 0.999,
        decoder_batch_norm_momentum: float = 0.99,
        **kwargs: str | int | float | bool | None,
    ) -> None:
        """Initialize the configuration with the production model defaults."""
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        if quantize_factor < 0 or quantize_factor > 8:
            raise ValueError("quantize_factor must be between 0 and 8")

        if kl_weight < 0 or l2_weight < 0:
            raise ValueError("loss weights must be non-negative")

        if batch_norm_epsilon <= 0:
            raise ValueError("batch_norm_epsilon must be positive")

        if not 0 <= backbone_batch_norm_momentum < 1:
            raise ValueError("backbone_batch_norm_momentum must be in [0, 1)")

        if not 0 <= decoder_batch_norm_momentum < 1:
            raise ValueError("decoder_batch_norm_momentum must be in [0, 1)")

        super().__init__(**kwargs)  # ty: ignore[invalid-argument-type]
        self.latent_dim = latent_dim
        self.quantize_factor = quantize_factor
        self.kl_weight = kl_weight
        self.l2_weight = l2_weight
        self.batch_norm_epsilon = batch_norm_epsilon
        self.backbone_batch_norm_momentum = backbone_batch_norm_momentum
        self.decoder_batch_norm_momentum = decoder_batch_norm_momentum

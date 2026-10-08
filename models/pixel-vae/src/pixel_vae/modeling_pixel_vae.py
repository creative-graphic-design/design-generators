"""MobileNetV2 variational autoencoder for RGBA visual elements."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Final

import torch
import torch.nn.functional as F
from jaxtyping import Float
from laygen.common.randomness import randn
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_pixel_vae import PixelVAEConfig

IMAGE_SIDE: Final[int] = 256
IMAGE_CHANNELS: Final[int] = 4
QUANTIZATION_LEVELS: Final[int] = 16


class MobileNetLayerKind(StrEnum):
    """Weighted layer kinds needed by the checkpoint converter."""

    conv = auto()
    depthwise = auto()
    batch_norm = auto()


@dataclass
class PixelVAEEncoderOutput(ModelOutput):
    """Posterior statistics emitted by :class:`PixelVAEEncoder`."""

    posterior_mean: Float[torch.Tensor, "batch latent"] | None = None
    posterior_log_variance: Float[torch.Tensor, "batch latent"] | None = None


@dataclass
class PixelVAEOutput(ModelOutput):
    """Reconstruction, posterior statistics, and losses from PixelVAE."""

    loss: Float[torch.Tensor, ""] | None = None
    reconstruction_loss: Float[torch.Tensor, ""] | None = None
    kl_divergence: Float[torch.Tensor, ""] | None = None
    logits: Float[torch.Tensor, "batch height width channels levels"] | None = None
    posterior_mean: Float[torch.Tensor, "batch latent"] | None = None
    posterior_log_variance: Float[torch.Tensor, "batch latent"] | None = None
    latents: Float[torch.Tensor, "batch latent"] | None = None


class PixelVAEBatchNorm2d(nn.Module):
    """Batch normalization with population variance and moving averages."""

    running_mean: Float[torch.Tensor, "channels"]
    running_var: Float[torch.Tensor, "channels"]

    def __init__(self, channels: int, eps: float, momentum: float) -> None:
        """Create affine parameters and moving statistics."""
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(
        self, inputs: Float[torch.Tensor, "batch channels height width"]
    ) -> Float[torch.Tensor, "batch channels height width"]:
        """Normalize with population variance and Keras EMA semantics."""
        if self.training:
            mean = inputs.mean(dim=(0, 2, 3))
            variance = (
                (inputs - mean.detach()[None, :, None, None])
                .square()
                .mean(dim=(0, 2, 3))
            )
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(
                    mean.detach(), alpha=1 - self.momentum
                )
                self.running_var.mul_(self.momentum).add_(
                    variance.detach(), alpha=1 - self.momentum
                )
        else:
            mean = self.running_mean
            variance = self.running_var

        scale = self.weight * torch.rsqrt(variance + self.eps)
        offset = self.bias - mean * scale
        return inputs * scale[None, :, None, None] + offset[None, :, None, None]


class PixelVAEMobileNetV2(nn.Module):
    """MobileNetV2 feature extractor configured for four input channels."""

    def __init__(self, config: PixelVAEConfig) -> None:
        """Build the source MobileNetV2 layer topology."""
        super().__init__()
        self.layers = nn.ModuleDict()
        self.keras_layer_kinds: dict[str, MobileNetLayerKind] = {}
        self._layer_strides: dict[str, int] = {}

        self._add_conv("Conv1", 4, 32, stride=2, kernel_size=3)
        self._add_bn("bn_Conv1", 32, config, config.backbone_batch_norm_momentum)

        settings: tuple[tuple[int, int, int, int], ...] = (
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        )
        in_channels = 32
        block_id = 0
        for expansion, channels, repeats, first_stride in settings:
            for repeat in range(repeats):
                stride = first_stride if repeat == 0 else 1
                prefix = "expanded_conv_" if block_id == 0 else f"block_{block_id}_"
                expanded_channels = in_channels * expansion
                if expansion != 1:
                    self._add_conv(
                        f"{prefix}expand", in_channels, expanded_channels, kernel_size=1
                    )
                    self._add_bn(
                        f"{prefix}expand_BN",
                        expanded_channels,
                        config,
                        config.backbone_batch_norm_momentum,
                    )

                self._add_depthwise(f"{prefix}depthwise", expanded_channels, stride)
                self._add_bn(
                    f"{prefix}depthwise_BN",
                    expanded_channels,
                    config,
                    config.backbone_batch_norm_momentum,
                )
                self._add_conv(
                    f"{prefix}project", expanded_channels, channels, kernel_size=1
                )
                self._add_bn(
                    f"{prefix}project_BN",
                    channels,
                    config,
                    config.backbone_batch_norm_momentum,
                )
                in_channels = channels
                block_id += 1

        self._add_conv("Conv_1", 320, 1280, kernel_size=1)
        self._add_bn("Conv_1_bn", 1280, config, config.backbone_batch_norm_momentum)
        self.output_channels = 1280

    def _add_conv(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        self.layers[name] = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.keras_layer_kinds[name] = MobileNetLayerKind.conv
        self._layer_strides[name] = stride

    def _add_depthwise(self, name: str, channels: int, stride: int) -> None:
        self.layers[name] = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=0,
            groups=channels,
            bias=False,
        )
        self.keras_layer_kinds[name] = MobileNetLayerKind.depthwise
        self._layer_strides[name] = stride

    def _add_bn(
        self, name: str, channels: int, config: PixelVAEConfig, momentum: float
    ) -> None:
        self.layers[name] = PixelVAEBatchNorm2d(
            channels, eps=config.batch_norm_epsilon, momentum=momentum
        )
        self.keras_layer_kinds[name] = MobileNetLayerKind.batch_norm

    def _apply_conv(
        self, name: str, inputs: Float[torch.Tensor, "batch channels height width"]
    ) -> Float[torch.Tensor, "batch channels height width"]:
        layer = self.layers[name]
        if layer.kernel_size == (3, 3):
            stride = self._layer_strides[name]
            height, width = inputs.shape[-2:]
            out_height = (height + stride - 1) // stride
            out_width = (width + stride - 1) // stride
            pad_height = max((out_height - 1) * stride + 3 - height, 0)
            pad_width = max((out_width - 1) * stride + 3 - width, 0)
            inputs = F.pad(
                inputs,
                (
                    pad_width // 2,
                    pad_width - pad_width // 2,
                    pad_height // 2,
                    pad_height - pad_height // 2,
                ),
            )

        return layer(inputs)

    def forward(
        self, inputs: Float[torch.Tensor, "batch channels height width"]
    ) -> Float[torch.Tensor, "batch features"]:
        """Return global-average-pooled MobileNetV2 features."""
        x = self._apply_conv("Conv1", inputs)
        x = F.relu6(self.layers["bn_Conv1"](x))

        for block_id in range(17):
            prefix = "expanded_conv_" if block_id == 0 else f"block_{block_id}_"
            expand_name = f"{prefix}expand"
            if expand_name in self.layers:
                x = self._apply_conv(expand_name, x)
                x = F.relu6(self.layers[f"{prefix}expand_BN"](x))

            depthwise_name = f"{prefix}depthwise"
            residual = x
            stride = self._layer_strides[depthwise_name]
            x = self._apply_conv(depthwise_name, x)
            x = F.relu6(self.layers[f"{prefix}depthwise_BN"](x))
            x = self._apply_conv(f"{prefix}project", x)
            x = self.layers[f"{prefix}project_BN"](x)
            if stride == 1 and x.shape[1] == residual.shape[1]:
                x = x + residual

        x = self._apply_conv("Conv_1", x)
        x = F.relu6(self.layers["Conv_1_bn"](x))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class PixelVAEEncoder(PreTrainedModel):
    """MobileNetV2 encoder that exports posterior-mean embeddings."""

    config_class = PixelVAEConfig
    base_model_prefix = "pixel_vae_encoder"
    main_input_name = "pixel_values"

    def __init__(self, config: PixelVAEConfig) -> None:
        """Create the feature extractor and posterior heads."""
        super().__init__(config)
        self.backbone = PixelVAEMobileNetV2(config)
        self.z_mean = nn.Linear(self.backbone.output_channels, config.latent_dim)
        self.z_log_variance = nn.Linear(
            self.backbone.output_channels, config.latent_dim
        )
        self.post_init()

    def forward(
        self, pixel_values: Float[torch.Tensor, "batch 4 256 256"]
    ) -> PixelVAEEncoderOutput:
        """Return posterior mean and log variance without sampling."""
        _validate_pixel_values(pixel_values)
        features = self.backbone(pixel_values)
        return PixelVAEEncoderOutput(
            posterior_mean=self.z_mean(features),
            posterior_log_variance=self.z_log_variance(features),
        )

    def encode(
        self, pixel_values: Float[torch.Tensor, "batch 4 256 256"]
    ) -> Float[torch.Tensor, "batch latent"]:
        """Return the float32 posterior mean used by Crello documents."""
        result = self(pixel_values)
        assert result.posterior_mean is not None
        return result.posterior_mean

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, PixelVAEBatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class PixelVAEDecoder(nn.Module):
    """Upsample latent vectors into per-channel 16-level logits."""

    def __init__(self, config: PixelVAEConfig) -> None:
        """Build the dense projection and five upsampling stages."""
        super().__init__()
        self.dense = nn.Linear(config.latent_dim, 8 * 8 * 32)
        self.batch_norms = nn.ModuleList(
            [
                PixelVAEBatchNorm2d(
                    32, config.batch_norm_epsilon, config.decoder_batch_norm_momentum
                ),
                PixelVAEBatchNorm2d(
                    256, config.batch_norm_epsilon, config.decoder_batch_norm_momentum
                ),
                PixelVAEBatchNorm2d(
                    256, config.batch_norm_epsilon, config.decoder_batch_norm_momentum
                ),
                PixelVAEBatchNorm2d(
                    128, config.batch_norm_epsilon, config.decoder_batch_norm_momentum
                ),
                PixelVAEBatchNorm2d(
                    128, config.batch_norm_epsilon, config.decoder_batch_norm_momentum
                ),
                PixelVAEBatchNorm2d(
                    64, config.batch_norm_epsilon, config.decoder_batch_norm_momentum
                ),
            ]
        )
        channels = (32, 256, 256, 128, 128, 64)
        self.transpose_convolutions = nn.ModuleList(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            for in_channels, out_channels in zip(
                channels[:-1], channels[1:], strict=True
            )
        )
        self.output_convolution = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.config = config

    def forward(
        self, latents: Float[torch.Tensor, "batch latent"]
    ) -> Float[torch.Tensor, "batch height width channels levels"]:
        """Return logits in NHWC channel and quantization order."""
        batch_size = latents.shape[0]
        x = (
            F.relu(self.dense(latents))
            .reshape(batch_size, 8, 8, 32)
            .permute(0, 3, 1, 2)
        )
        for batch_norm, transpose_convolution in zip(
            self.batch_norms[:-1], self.transpose_convolutions, strict=True
        ):
            x = F.relu(transpose_convolution(batch_norm(x)))

        x = self.batch_norms[-1](x)
        x = F.pad(x, (1, 1, 1, 1))
        x = self.output_convolution(x)
        return x.reshape(
            batch_size, IMAGE_CHANNELS, QUANTIZATION_LEVELS, IMAGE_SIDE, IMAGE_SIDE
        ).permute(0, 3, 4, 1, 2)


class PixelVAEModel(PreTrainedModel):
    """Transformers model implementing the RGBA PixelVAE objective."""

    config_class = PixelVAEConfig
    base_model_prefix = "pixel_vae"
    main_input_name = "pixel_values"

    def __init__(self, config: PixelVAEConfig) -> None:
        """Create the posterior encoder and autoregressive image decoder."""
        super().__init__(config)
        self.encoder = PixelVAEEncoder(config)
        self.decoder = PixelVAEDecoder(config)
        self.post_init()

    def encode(
        self, pixel_values: Float[torch.Tensor, "batch 4 256 256"]
    ) -> Float[torch.Tensor, "batch latent"]:
        """Export the encoder posterior mean without sampling."""
        return self.encoder.encode(pixel_values)

    def forward(
        self,
        pixel_values: Float[torch.Tensor, "batch 4 256 256"],
        *,
        sample: bool | None = None,
        epsilon: Float[torch.Tensor, "batch latent"] | None = None,
    ) -> PixelVAEOutput:
        """Compute logits and the source reconstruction, KL, and L2 losses.

        Args:
            pixel_values: Float32 RGBA images scaled to ``[0, 1]``.
            sample: Whether to sample the posterior; defaults to module mode.
            epsilon: Optional fixed standard-normal noise for deterministic runs.

        Returns:
            Decoder logits, posterior statistics, latent vectors, and losses.

        Raises:
            ValueError: If image dimensions or sampling-noise shape are invalid.
            TypeError: If inputs are not float32 tensors.

        Examples:
            >>> model = PixelVAEModel(PixelVAEConfig(latent_dim=8)).eval()
            >>> image = torch.zeros(1, 4, 256, 256)
            >>> model(image).posterior_mean.shape
            torch.Size([1, 8])
        """
        encoded = self.encoder(pixel_values)
        assert encoded.posterior_mean is not None
        assert encoded.posterior_log_variance is not None
        z_mean, z_log_variance = encoded.posterior_mean, encoded.posterior_log_variance
        should_sample = self.training if sample is None else sample
        if should_sample:
            if epsilon is None:
                epsilon = randn(
                    z_mean.shape,
                    device=z_mean.device,
                    dtype=z_mean.dtype,
                )

            if epsilon.shape != z_mean.shape:
                raise ValueError("epsilon must have the same shape as the posterior")

            latents = z_mean + z_log_variance.mul(0.5).exp() * epsilon
        else:
            if epsilon is not None:
                raise ValueError("epsilon is only valid when sample=True")

            latents = z_mean

        logits = self.decoder(latents)
        reconstruction_loss = pixelvae_reconstruction_loss(
            pixel_values, logits, self.config.quantize_factor
        )
        kl_divergence = (
            -0.5 * (1 + z_log_variance - z_mean.square() - z_log_variance.exp()).mean()
        )
        l2_loss = self.config.l2_weight * (
            self.encoder.z_mean.weight.square().sum()
            + self.encoder.z_mean.bias.square().sum()
            + self.encoder.z_log_variance.weight.square().sum()
            + self.encoder.z_log_variance.bias.square().sum()
        )
        loss = reconstruction_loss + self.config.kl_weight * kl_divergence + l2_loss
        return PixelVAEOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_divergence=kl_divergence,
            logits=logits,
            posterior_mean=z_mean,
            posterior_log_variance=z_log_variance,
            latents=latents,
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d | nn.ConvTranspose2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, PixelVAEBatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def pixelvae_reconstruction_loss(
    pixel_values: Float[torch.Tensor, "batch 4 256 256"],
    logits: Float[torch.Tensor, "batch height width channels levels"],
    quantize_factor: int = 4,
) -> Float[torch.Tensor, ""]:
    """Compute the original alpha-masked RGB and unmasked alpha loss.

    Args:
        pixel_values: Float32 RGBA images scaled from source bytes.
        logits: Per-channel categorical logits.
        quantize_factor: Number of low source bits removed before CE.

    Returns:
        Reconstruction loss summed over rows and averaged over batch, columns,
        and channels.
    """
    if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != (
        4,
        IMAGE_SIDE,
        IMAGE_SIDE,
    ):
        raise ValueError("pixel_values must have shape [batch, 4, 256, 256]")

    targets = torch.round(pixel_values * 255).to(torch.uint8).permute(0, 2, 3, 1)
    targets = torch.bitwise_right_shift(targets, quantize_factor).long()
    if logits.shape != (*targets.shape, QUANTIZATION_LEVELS):
        raise ValueError("logits must have shape [batch, 256, 256, 4, 16]")

    per_channel = F.cross_entropy(
        logits.permute(0, 4, 1, 2, 3), targets, reduction="none"
    )
    rgb_mask = (targets[..., 3:4] > 0).to(per_channel.dtype)
    rgb_loss = per_channel[..., :3] * rgb_mask
    alpha_loss = per_channel[..., 3:4]
    return (rgb_loss + alpha_loss).sum(dim=1).mean()


def _validate_pixel_values(
    pixel_values: Float[torch.Tensor, "..."],
) -> None:
    if pixel_values.dtype != torch.float32:
        raise TypeError("pixel_values must have dtype torch.float32")

    if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != (4, 256, 256):
        raise ValueError("pixel_values must have shape [batch, 4, 256, 256]")

"""Stable Diffusion pipelines for TKG-DM inference."""

from __future__ import annotations

import math
from typing import ClassVar, Final

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from jaxtyping import Float
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from transformers.models.clip.modeling_clip import CLIPTextModelWithProjection

DEFAULT_SD15_TARGET_SHIFT: Final[float] = 0.07
DEFAULT_SDXL_TARGET_SHIFT: Final[float] = 0.11


def _positive_ratio(
    values: Float[torch.Tensor, "height width"],
) -> Float[torch.Tensor, ""]:
    return (values > 0).float().mean()


def _shift_positive_ratio(
    latents: Float[torch.Tensor, "batch channel height width"],
    *,
    target_shift: float,
) -> Float[torch.Tensor, "batch channel height width"]:
    shifted_latents = latents.clone()
    for channel_index in (1, 2):
        channel = latents[:, channel_index, :, :]
        target_ratio = _positive_ratio(channel) + target_shift
        if bool(target_ratio > 1.0):
            raise ValueError(
                "target_shift cannot raise the channel positive ratio above 1"
            )

        offset = 0.0
        while True:
            shifted_channel = channel + offset
            if _positive_ratio(shifted_channel) >= target_ratio:
                break

            # The reference's 0.01 increments set the parity-critical threshold.
            offset += 0.01

        shifted_latents[:, channel_index, :, :] = shifted_channel

    return shifted_latents


def _validate_tkg_noise_inputs(
    latents: Float[torch.Tensor, "batch channel height width"],
    target_shift: float,
    *,
    sigma: float | None = None,
) -> tuple[int, int, int, int]:
    if latents.ndim != 4:
        raise ValueError("latents must have batch, channel, height, and width axes")

    if not latents.is_floating_point():
        raise TypeError("latents must use a floating-point dtype")

    batch_size, channel_count, height, width = latents.shape
    if batch_size < 1 or channel_count < 3:
        raise ValueError("latents must contain a batch and at least three channels")

    if height < 8 or width < 8:
        raise ValueError("latent height and width must each be at least 8")

    if not math.isfinite(target_shift) or target_shift < 0:
        raise ValueError("target_shift must be finite and non-negative")

    if sigma is not None and (not math.isfinite(sigma) or sigma <= 0):
        raise ValueError("sigma must be finite and positive")

    return batch_size, channel_count, height, width


def shift_tkg_noise_channels(
    latents: Float[torch.Tensor, "batch channel height width"],
    target_shift: float,
) -> Float[torch.Tensor, "batch channel height width"]:
    """Shift the TKG-DM noise channels without applying its spatial mask.

    Args:
        latents: Four-dimensional initial noise with at least three channels and
            spatial dimensions of at least eight pixels.
        target_shift: Positive-value-ratio increase applied to channels 1 and 2.

    Returns:
        Noise with TKG-DM's channel shift and no spatial blending.

    Raises:
        TypeError: If ``latents`` is not floating-point.
        ValueError: If tensor dimensions or ``target_shift`` are invalid.

    Examples:
        The shift changes channels 1 and 2 while retaining the other channels:

        >>> noise = torch.zeros((1, 4, 8, 8))
        >>> shifted = shift_tkg_noise_channels(noise, target_shift=0.07)
        >>> torch.equal(shifted[:, 0], noise[:, 0]) and torch.equal(shifted[:, 3], noise[:, 3])
        True
    """
    _validate_tkg_noise_inputs(latents, target_shift)
    if target_shift == 0:
        return latents

    shifted_latents = latents.clone()
    for channel_index in (1, 2):
        channel = latents[:, channel_index, :, :]
        initial_ratio = float(_positive_ratio(channel).item())
        target_ratio = initial_ratio + target_shift
        if target_ratio > 1.0:
            raise ValueError(
                "target_shift cannot raise the channel positive ratio above 1"
            )

        delta_min, delta_max = -10.0, 10.0
        delta = 0.0
        for _ in range(100):
            delta = (delta_min + delta_max) / 2
            shifted_channel = latents.clone()[:, channel_index, :, :]
            shifted_channel += delta
            current_ratio = float(_positive_ratio(shifted_channel).item())
            if abs(current_ratio - target_ratio) < 1e-4:
                break

            if current_ratio < target_ratio:
                delta_min = delta
            else:
                delta_max = delta

        shifted_latents[:, channel_index, :, :] += delta

    return shifted_latents


def _gaussian_mask(
    height: int,
    width: int,
    *,
    sigma: float,
    device: torch.device,
) -> Float[torch.Tensor, "batch channel height width"]:
    y = torch.linspace(-1, 1, height // 8, device=device)
    x = torch.linspace(-1, 1, width // 8, device=device)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    gaussian = torch.exp(-((x_grid**2 + y_grid**2) / (2 * sigma**2)))
    return gaussian[None, None, :, :]


def apply_tkg_noise(
    latents: Float[torch.Tensor, "batch channel height width"],
    target_shift: float,
    sigma: float = 0.5,
) -> Float[torch.Tensor, "batch channel height width"]:
    """Apply TKG-DM's color-channel shift and spatial mask to initial noise.

    Args:
        latents: Four-dimensional initial noise with at least three channels and
            spatial dimensions of at least eight pixels.
        target_shift: Positive-value-ratio increase applied to channels 1 and 2.
        sigma: Standard deviation of the Gaussian mask in normalized coordinates.

    Returns:
        Noise blended between its input and the shifted channels.

    Raises:
        TypeError: If ``latents`` is not floating-point.
        ValueError: If the tensor dimensions or scalar controls are invalid.

    Examples:
        A zero shift leaves the noise unchanged:

        >>> noise = torch.zeros((1, 4, 8, 8))
        >>> torch.equal(apply_tkg_noise(noise, target_shift=0.0), noise)
        True
    """
    batch_size, channel_count, height, width = _validate_tkg_noise_inputs(
        latents,
        target_shift,
        sigma=sigma,
    )

    if target_shift == 0:
        return latents

    shifted_latents = _shift_positive_ratio(latents, target_shift=target_shift)
    mask = _gaussian_mask(
        height,
        width,
        sigma=sigma,
        device=latents.device,
    )
    mask = F.interpolate(
        mask,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    mask = mask.expand(batch_size, channel_count, -1, -1)
    mask = mask.to(device=latents.device, dtype=latents.dtype)

    return mask * latents + (1 - mask) * shifted_latents


class TKGStableDiffusionPipeline(StableDiffusionPipeline):
    """Stable Diffusion 1.5 with TKG-DM initial-noise processing.

    The default target shift is ``0.07``, matching the paper because no author
    SD 1.5 pipeline is available.

    Examples:
        Load the released Stable Diffusion 1.5 checkpoint and run on CUDA:

        >>> pipe = TKGStableDiffusionPipeline.from_pretrained(
        ...     "stable-diffusion-v1-5/stable-diffusion-v1-5"
        ... )
        >>> pipe.to("cuda")
    """

    scheduler: DDIMScheduler

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker | None,
        feature_extractor: CLIPImageProcessor | None,
        image_encoder: CLIPVisionModelWithProjection | None = None,
        requires_safety_checker: bool = True,
    ) -> None:
        """Initialize the pipeline with a supported scheduler.

        Args:
            vae: Variational autoencoder from the base checkpoint.
            text_encoder: Text encoder from the base checkpoint.
            tokenizer: Tokenizer matching ``text_encoder``.
            unet: Denoising network from the base checkpoint.
            scheduler: DDIM scheduler used for denoising.
            safety_checker: Optional image safety checker.
            feature_extractor: Image processor required by the safety checker.
            image_encoder: Optional image encoder from the base checkpoint.
            requires_safety_checker: Whether the pipeline requires a safety checker.

        Raises:
            TypeError: If ``scheduler`` is not a ``DDIMScheduler``.
        """
        if not isinstance(scheduler, DDIMScheduler):
            raise TypeError("TKG-DM requires a DDIMScheduler")

        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device | str,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: Float[torch.Tensor, "batch channel height width"] | None = None,
    ) -> Float[torch.Tensor, "batch channel height width"]:
        """Prepare DDIM latents and apply the Stable Diffusion 1.5 default shift.

        Args:
            batch_size: Number of generated images.
            num_channels_latents: Number of channels in the initial noise.
            height: Requested output height in pixels.
            width: Requested output width in pixels.
            dtype: Dtype used by the denoising model.
            device: Device where the pipeline generates images.
            generator: Optional random generator or per-image generators.
            latents: Optional caller-supplied initial noise.

        Returns:
            Prepared initial noise after applying the TKG-DM transformation.
        """
        latents = super().prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        return apply_tkg_noise(latents, target_shift=DEFAULT_SD15_TARGET_SHIFT)


class TKGStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    """SDXL with TKG-DM initial-noise processing.

    The default target shift is ``0.11``, matching the author's SDXL implementation
    used for end-to-end parity.

    Examples:
        Load the released SDXL checkpoint and run on CUDA:

        >>> pipe = TKGStableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0"
        ... )
        >>> pipe.to("cuda")
    """

    scheduler: DDIMScheduler

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection | None = None,
        feature_extractor: CLIPImageProcessor | None = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: bool | None = None,
    ) -> None:
        """Initialize the pipeline with a supported scheduler.

        Args:
            vae: Variational autoencoder from the base checkpoint.
            text_encoder: First text encoder from the base checkpoint.
            text_encoder_2: Second text encoder from the base checkpoint.
            tokenizer: Tokenizer matching ``text_encoder``.
            tokenizer_2: Tokenizer matching ``text_encoder_2``.
            unet: Denoising network from the base checkpoint.
            scheduler: DDIM scheduler used for denoising.
            image_encoder: Optional image encoder from the base checkpoint.
            feature_extractor: Optional image processor for image inputs.
            force_zeros_for_empty_prompt: Whether empty negative prompts use zeros.
            add_watermarker: Whether to add a watermark to generated images.

        Raises:
            TypeError: If ``scheduler`` is not a ``DDIMScheduler``.
        """
        if not isinstance(scheduler, DDIMScheduler):
            raise TypeError("TKG-DM requires a DDIMScheduler")

        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker,
        )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device | str,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: Float[torch.Tensor, "batch channel height width"] | None = None,
    ) -> Float[torch.Tensor, "batch channel height width"]:
        """Prepare DDIM latents and apply the SDXL default shift.

        Args:
            batch_size: Number of generated images.
            num_channels_latents: Number of channels in the initial noise.
            height: Requested output height in pixels.
            width: Requested output width in pixels.
            dtype: Dtype used by the denoising model.
            device: Device where the pipeline generates images.
            generator: Optional random generator or per-image generators.
            latents: Optional caller-supplied initial noise.

        Returns:
            Prepared initial noise after applying the TKG-DM transformation.
        """
        latents = super().prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        return apply_tkg_noise(latents, target_shift=DEFAULT_SDXL_TARGET_SHIFT)


class TKGDMHubPipeline(DiffusionPipeline):
    """Select the TKG-DM pipeline that matches the supplied model components."""

    _optional_components: ClassVar[tuple[str, ...]] = (
        "safety_checker",
        "feature_extractor",
        "image_encoder",
        "text_encoder_2",
        "tokenizer_2",
    )
    stable_pipeline_class: ClassVar[type[StableDiffusionPipeline]] = (
        TKGStableDiffusionPipeline
    )
    sdxl_pipeline_class: ClassVar[type[StableDiffusionXLPipeline]] = (
        TKGStableDiffusionXLPipeline
    )

    def __new__(
        cls: type[TKGDMHubPipeline],
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker | None = None,
        feature_extractor: CLIPImageProcessor | None = None,
        image_encoder: CLIPVisionModelWithProjection | None = None,
        text_encoder_2: CLIPTextModelWithProjection | None = None,
        tokenizer_2: CLIPTokenizer | None = None,
        requires_safety_checker: bool = True,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: bool | None = None,
    ) -> DiffusionPipeline:
        """Return a TKG-DM pipeline selected from the checkpoint components.

        Args:
            cls: Dynamic pipeline class created by Diffusers.
            vae: Variational autoencoder from the selected base checkpoint.
            text_encoder: First text encoder from the selected base checkpoint.
            tokenizer: Tokenizer matching ``text_encoder``.
            unet: Denoising network from the selected base checkpoint.
            scheduler: DDIM scheduler used for denoising.
            safety_checker: Optional image safety checker.
            feature_extractor: Image processor used by the safety checker.
            image_encoder: Optional image encoder from the base checkpoint.
            text_encoder_2: Optional second text encoder for SDXL.
            tokenizer_2: Optional tokenizer matching ``text_encoder_2``.
            requires_safety_checker: Whether a safety checker is required.
            force_zeros_for_empty_prompt: Whether empty negative prompts use zeros.
            add_watermarker: Whether generated images receive a watermark.

        Returns:
            A Stable Diffusion 1.5 or SDXL TKG-DM pipeline.

        Raises:
            ValueError: If only one of the two SDXL text-encoder components is set.
            TypeError: If the selected pipeline receives a non-DDIM scheduler.
        """
        # The checkpoint's second text encoder identifies the model family.
        if text_encoder_2 is None:
            if tokenizer_2 is not None:
                raise ValueError(
                    "SDXL loading requires both second text-encoder components"
                )

            return cls.stable_pipeline_class(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
                requires_safety_checker=requires_safety_checker,
            )

        if tokenizer_2 is None:
            raise ValueError(
                "SDXL loading requires both second text-encoder components"
            )

        return cls.sdxl_pipeline_class(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker,
        )

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker | None = None,
        feature_extractor: CLIPImageProcessor | None = None,
        image_encoder: CLIPVisionModelWithProjection | None = None,
        text_encoder_2: CLIPTextModelWithProjection | None = None,
        tokenizer_2: CLIPTokenizer | None = None,
        requires_safety_checker: bool = True,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: bool | None = None,
    ) -> None:
        """Expose component names required by the Diffusers custom pipeline loader.

        Args:
            vae: Variational autoencoder from the selected base checkpoint.
            text_encoder: First text encoder from the selected base checkpoint.
            tokenizer: Tokenizer matching ``text_encoder``.
            unet: Denoising network from the selected base checkpoint.
            scheduler: DDIM scheduler used for denoising.
            safety_checker: Optional image safety checker.
            feature_extractor: Image processor used by the safety checker.
            image_encoder: Optional image encoder from the base checkpoint.
            text_encoder_2: Optional second text encoder for SDXL.
            tokenizer_2: Optional tokenizer matching ``text_encoder_2``.
            requires_safety_checker: Whether a safety checker is required.
            force_zeros_for_empty_prompt: Whether empty negative prompts use zeros.
            add_watermarker: Whether generated images receive a watermark.
        """

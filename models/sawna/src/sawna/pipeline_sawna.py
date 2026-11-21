"""Stable Diffusion pipelines for reserving text-free image regions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import ClassVar, TypedDict, Unpack

import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)
from diffusers.utils.torch_utils import randn_tensor
from jaxtyping import Float
from PIL import Image
from tkg_dm.pipeline_tkg_dm import TKGDMHubPipeline

from sawna.processing_sawna import DEFAULT_TARGET_SHIFT, BoundingBox, apply_sawna_noise


class _StableDiffusionCallOptions(TypedDict, total=False):
    height: int | None
    width: int | None
    num_inference_steps: int
    timesteps: list[int] | None
    sigmas: list[float] | None
    guidance_scale: float
    negative_prompt: str | list[str] | None
    num_images_per_prompt: int | None
    eta: float
    generator: torch.Generator | list[torch.Generator] | None
    latents: Float[torch.Tensor, "batch channel height width"] | None
    prompt_embeds: Float[torch.Tensor, "batch sequence hidden"] | None
    negative_prompt_embeds: Float[torch.Tensor, "batch sequence hidden"] | None
    ip_adapter_image: PipelineImageInput | None
    ip_adapter_image_embeds: list[Float[torch.Tensor, ...]] | None
    output_type: str | None
    return_dict: bool
    cross_attention_kwargs: dict[str, object] | None
    guidance_rescale: float
    clip_skip: int | None
    callback_on_step_end: (
        Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None
    )
    callback_on_step_end_tensor_inputs: list[str]
    callback: Callable[[int, int, Float[torch.Tensor, ...]], None] | None
    callback_steps: int


class _StableDiffusionXLCallOptions(TypedDict, total=False):
    prompt_2: str | list[str] | None
    height: int | None
    width: int | None
    num_inference_steps: int
    timesteps: list[int] | None
    sigmas: list[float] | None
    denoising_end: float | None
    guidance_scale: float
    negative_prompt: str | list[str] | None
    negative_prompt_2: str | list[str] | None
    num_images_per_prompt: int | None
    eta: float
    generator: torch.Generator | list[torch.Generator] | None
    latents: Float[torch.Tensor, "batch channel height width"] | None
    prompt_embeds: Float[torch.Tensor, "batch sequence hidden"] | None
    negative_prompt_embeds: Float[torch.Tensor, "batch sequence hidden"] | None
    pooled_prompt_embeds: Float[torch.Tensor, "batch hidden"] | None
    negative_pooled_prompt_embeds: Float[torch.Tensor, "batch hidden"] | None
    ip_adapter_image: PipelineImageInput | None
    ip_adapter_image_embeds: list[Float[torch.Tensor, ...]] | None
    output_type: str | None
    return_dict: bool
    cross_attention_kwargs: dict[str, object] | None
    guidance_rescale: float
    original_size: tuple[int, int] | None
    crops_coords_top_left: tuple[int, int]
    target_size: tuple[int, int] | None
    negative_original_size: tuple[int, int] | None
    negative_crops_coords_top_left: tuple[int, int]
    negative_target_size: tuple[int, int] | None
    clip_skip: int | None
    callback_on_step_end: (
        Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None
    )
    callback_on_step_end_tensor_inputs: list[str]
    callback: Callable[[int, int, Float[torch.Tensor, ...]], None] | None
    callback_steps: int
    guidance_scale_end: float | None


@dataclass(frozen=True)
class _SAWNACallParameters:
    bounding_boxes: tuple[BoundingBox, ...]
    target_shift: float
    blur_sigma: float | None


_SAWNA_CALL_PARAMETERS: ContextVar[_SAWNACallParameters | None] = ContextVar(
    "sawna_call_parameters",
    default=None,
)


class _SAWNAInitialNoiseMixin:
    """Apply SAWNA to the initial noise while leaving denoising to Diffusers."""

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
        """Prepare initial noise and apply call-scoped SAWNA controls.

        Args:
            batch_size: Number of images to generate.
            num_channels_latents: Number of channels in the initial noise.
            height: Requested image height in pixels.
            width: Requested image width in pixels.
            dtype: Dtype used by the denoising model.
            device: Device used by the denoising pipeline.
            generator: Optional random generator or per-image generators.
            latents: Optional caller-supplied initial noise.

        Returns:
            Prepared initial noise, with the reserved regions blended when set.
        """
        # 2. Prepare raw noise and apply SAWNA before the scheduler scales it.
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        parameters = _SAWNA_CALL_PARAMETERS.get()
        if parameters is not None:
            latents = apply_sawna_noise(
                latents,
                parameters.bounding_boxes,
                target_shift=parameters.target_shift,
                blur_sigma=parameters.blur_sigma,
            )

        return latents * self.scheduler.init_noise_sigma


class SAWNAStableDiffusionPipeline(
    _SAWNAInitialNoiseMixin,
    StableDiffusionPipeline,
):
    """Stable Diffusion 1.5 and 2.1 pipelines with reserved image regions."""

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        *,
        bounding_boxes: Sequence[BoundingBox] | None = None,
        target_shift: float = DEFAULT_TARGET_SHIFT,
        blur_sigma: float | None = None,
        **kwargs: Unpack[_StableDiffusionCallOptions],
    ) -> (
        StableDiffusionPipelineOutput
        | tuple[
            list[Image.Image]
            | Float[np.ndarray, "batch height width channels"]
            | Float[torch.Tensor, "batch channel height width"],
            list[bool] | None,
        ]
    ):
        """Generate images while reserving the requested bounding boxes.

        Args:
            prompt: Text prompt or prompts. Pass prompt embeddings through the
                standard Diffusers keyword arguments when using precomputed text.
            bounding_boxes: Normalized ``(x1, y1, x2, y2)`` regions reserved for
                later design content. Empty or omitted boxes preserve standard
                Diffusers initial-noise behavior.
            target_shift: Positive-value-ratio increase applied inside the boxes.
            blur_sigma: Gaussian standard deviation in latent-grid pixels. ``None``
                uses two percent of the shorter latent-grid side.
            kwargs: Standard Stable Diffusion pipeline arguments.

        Returns:
            The standard Diffusers pipeline output.

        Raises:
            TypeError: If initial noise is not floating-point.
            ValueError: If a SAWNA control or bounding box is invalid.

        Examples:
            >>> boxes = [(0.3, 0.3, 0.7, 0.7)]
            >>> len(boxes)
            1
        """
        parameters = _SAWNACallParameters(
            bounding_boxes=tuple(bounding_boxes) if bounding_boxes is not None else (),
            target_shift=target_shift,
            blur_sigma=blur_sigma,
        )
        token = _SAWNA_CALL_PARAMETERS.set(parameters)
        try:
            # 1. Scope reserved-region controls while Diffusers runs the pipeline.
            return super().__call__(prompt=prompt, **kwargs)
        finally:
            _SAWNA_CALL_PARAMETERS.reset(token)


class SAWNAStableDiffusionXLPipeline(
    _SAWNAInitialNoiseMixin,
    StableDiffusionXLPipeline,
):
    """SDXL pipeline with reserved image regions."""

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        *,
        bounding_boxes: Sequence[BoundingBox] | None = None,
        target_shift: float = DEFAULT_TARGET_SHIFT,
        blur_sigma: float | None = None,
        **kwargs: Unpack[_StableDiffusionXLCallOptions],
    ) -> (
        StableDiffusionXLPipelineOutput
        | tuple[
            list[Image.Image]
            | Float[np.ndarray, "batch height width channels"]
            | Float[torch.Tensor, "batch channel height width"]
        ]
    ):
        """Generate SDXL images while reserving the requested bounding boxes.

        Args:
            prompt: Text prompt or prompts. Pass prompt embeddings through the
                standard Diffusers keyword arguments when using precomputed text.
            bounding_boxes: Normalized ``(x1, y1, x2, y2)`` regions reserved for
                later design content. Empty or omitted boxes preserve standard
                Diffusers initial-noise behavior.
            target_shift: Positive-value-ratio increase applied inside the boxes.
            blur_sigma: Gaussian standard deviation in latent-grid pixels. ``None``
                uses two percent of the shorter latent-grid side.
            kwargs: Standard SDXL pipeline arguments, including size and crop
                conditioning.

        Returns:
            The standard Diffusers pipeline output.

        Raises:
            TypeError: If initial noise is not floating-point.
            ValueError: If a SAWNA control or bounding box is invalid.

        Examples:
            >>> boxes = [(0.3, 0.3, 0.7, 0.7)]
            >>> len(boxes)
            1
        """
        parameters = _SAWNACallParameters(
            bounding_boxes=tuple(bounding_boxes) if bounding_boxes is not None else (),
            target_shift=target_shift,
            blur_sigma=blur_sigma,
        )
        token = _SAWNA_CALL_PARAMETERS.set(parameters)
        try:
            # 1. Scope reserved-region controls while Diffusers runs the pipeline.
            return super().__call__(prompt=prompt, **kwargs)
        finally:
            _SAWNA_CALL_PARAMETERS.reset(token)


class SAWNAHubPipeline(TKGDMHubPipeline):
    """Load the SAWNA pipeline selected for a checkpoint component family."""

    stable_pipeline_class: ClassVar[type[StableDiffusionPipeline]] = (
        SAWNAStableDiffusionPipeline
    )
    sdxl_pipeline_class: ClassVar[type[StableDiffusionXLPipeline]] = (
        SAWNAStableDiffusionXLPipeline
    )

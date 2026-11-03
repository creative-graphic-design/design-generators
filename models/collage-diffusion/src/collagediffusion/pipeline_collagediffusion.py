"""Diffusers pipeline and layer helpers for collage harmonization."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from jaxtyping import Float
from laygen.common.randomness import randn, resolve_torch_generator
from PIL import Image, ImageFilter


class TokenizerProtocol(Protocol):
    """The tokenizer operation needed to locate layer phrases in a prompt."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode text into token ids."""


@dataclass(frozen=True, slots=True)
class CollageLayer:
    """One positioned RGBA layer and the controls associated with its prompt.

    Args:
        image: RGBA image for the layer.
        prompt: Layer description that also appears in the scene prompt.
        position: Pixel ``(x, y)`` position on the output canvas.
        noise_level: Layer-specific amount of source preservation, in ``[0, 1]``.
        positive_attention_weight: Attention-logit boost inside the visible mask.
        negative_attention_weight: Attention-logit penalty outside the visible mask.
        textual_inversion_token: Optional token inserted before ``prompt``.

    Raises:
        TypeError: If the image or position has the wrong type.
        ValueError: If the image mode, prompt, or numeric control is invalid.
    """

    image: Image.Image
    prompt: str
    position: tuple[int, int] = (0, 0)
    noise_level: float = 0.7
    positive_attention_weight: float = 0.0
    negative_attention_weight: float = 0.0
    textual_inversion_token: str | None = None

    def __post_init__(self) -> None:
        """Validate layer input modes and scalar controls."""
        if not isinstance(self.image, Image.Image):
            raise TypeError("image must be a PIL Image")

        if self.image.mode != "RGBA":
            raise ValueError("image must use RGBA mode")

        if not self.prompt.strip():
            raise ValueError("prompt must be a non-empty layer description")

        if len(self.position) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in self.position
        ):
            raise TypeError("position must contain two integer pixel coordinates")

        if not math.isfinite(self.noise_level) or not 0.0 <= self.noise_level <= 1.0:
            raise ValueError("noise_level must be finite and between 0 and 1")

        for name, weight in (
            ("positive_attention_weight", self.positive_attention_weight),
            ("negative_attention_weight", self.negative_attention_weight),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        if (
            self.textual_inversion_token is not None
            and not self.textual_inversion_token.strip()
        ):
            raise ValueError("textual_inversion_token must be non-empty when provided")


def _placed_alpha(
    layer: CollageLayer,
    *,
    width: int,
    height: int,
) -> Float[np.ndarray, "height width"]:
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    # Using the RGBA image as its own paste mask multiplies partial alpha by itself.
    canvas.paste(layer.image, layer.position, layer.image)

    # Keep alpha arithmetic at float64 until conversion to the 8-bit noise map.
    return np.asarray(canvas.getchannel("A"), dtype=np.float64) / 255.0


def compose_collage_layers(
    layers: Sequence[CollageLayer],
    *,
    width: int = 512,
    height: int = 512,
    background_color: tuple[int, int, int] = (150, 150, 150),
) -> tuple[Image.Image, list[Float[np.ndarray, "height width"]]]:
    """Composite layers back-to-front and return their visible alpha masks.

    Args:
        layers: Layers ordered from back to front.
        width: Output canvas width in pixels.
        height: Output canvas height in pixels.
        background_color: RGB color used outside all layers.

    Returns:
        The RGB collage and one visible alpha mask per input layer.

    Raises:
        ValueError: If there are no layers or the canvas is invalid.

    Examples:
        >>> layer = CollageLayer(Image.new("RGBA", (2, 2), (255, 0, 0, 255)), "red square")
        >>> image, masks = compose_collage_layers([layer], width=2, height=2)
        >>> image.getpixel((0, 0)), float(masks[0][0, 0])
        ((255, 0, 0), 1.0)
    """
    if not layers:
        raise ValueError("layers must contain at least one RGBA image")

    if width < 1 or height < 1:
        raise ValueError("canvas width and height must be positive")

    if len(background_color) != 3 or any(
        not 0 <= channel <= 255 for channel in background_color
    ):
        raise ValueError("background_color must contain three byte values")

    composite = Image.new("RGB", (width, height), background_color)
    for layer in layers:
        # Use each RGBA layer's alpha as its paste mask while compositing.
        composite.paste(layer.image, layer.position, layer.image)

    visible_masks: list[Float[np.ndarray, "height width"]] = []
    for index, layer in enumerate(layers):
        layer_alpha = _placed_alpha(layer, width=width, height=height)

        top_canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        for top_layer in layers[index + 1 :]:
            top_canvas.paste(top_layer.image, top_layer.position, top_layer.image)

        top_alpha = np.asarray(top_canvas.getchannel("A"), dtype=np.float64) / 255.0
        visible_masks.append(layer_alpha * (1.0 - top_alpha))

    return composite, visible_masks


def make_layer_noise_map(
    visible_masks: Sequence[Float[np.ndarray, "height width"]],
    layers: Sequence[CollageLayer],
    *,
    strength: float,
    blur_radius: float = 10.0,
) -> Float[np.ndarray, "height width"]:
    """Blend per-layer noise controls over the global image-to-image strength.

    Args:
        visible_masks: Visible alpha masks in back-to-front layer order.
        layers: Layers whose ``noise_level`` values drive the map.
        strength: Global image-to-image strength used outside layer masks.
        blur_radius: Gaussian blur radius for smooth layer boundaries.

    Returns:
        A grayscale float map with values in ``[0, 1]``.

    Raises:
        ValueError: If masks, layers, or controls are inconsistent.
    """
    if len(visible_masks) != len(layers) or not layers:
        raise ValueError("visible_masks and layers must have the same non-zero length")

    if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
        raise ValueError("strength must be finite and between 0 and 1")

    if not math.isfinite(blur_radius) or blur_radius < 0:
        raise ValueError("blur_radius must be finite and non-negative")

    noise_map = np.full_like(visible_masks[0], int(strength * 255), dtype=np.uint8)
    for mask, layer in zip(visible_masks, layers, strict=True):
        if mask.shape != noise_map.shape:
            raise ValueError("all visible masks must have the same shape")

        if not np.isfinite(mask).all() or np.any(mask < 0.0) or np.any(mask > 1.0):
            raise ValueError("visible masks must contain finite values between 0 and 1")

        layer_values = Image.fromarray(
            (mask * layer.noise_level * 255).astype(np.uint8)
        )
        layer_mask = Image.fromarray((mask * 255).astype(np.uint8))
        collage_map = Image.fromarray(noise_map)
        collage_map.paste(layer_values, (0, 0), layer_mask)
        noise_map = np.asarray(collage_map, dtype=np.uint8)

    grayscale = Image.fromarray(noise_map)
    if blur_radius > 0:
        grayscale = grayscale.filter(ImageFilter.GaussianBlur(blur_radius))

    return np.asarray(grayscale, dtype=np.float32) / 255.0


def _find_token_span(
    prompt_token_ids: Sequence[int],
    layer_token_ids: Sequence[int],
) -> tuple[int, ...]:
    if not layer_token_ids:
        raise ValueError("layer descriptions must produce at least one tokenizer token")

    token_count = len(layer_token_ids)
    for start in range(1, len(prompt_token_ids) - token_count + 1):
        if tuple(prompt_token_ids[start : start + token_count]) == tuple(
            layer_token_ids
        ):
            return tuple(range(start, start + token_count))

    raise ValueError("each layer description must appear as a token span in the prompt")


def _prompt_token_spans(
    tokenizer: TokenizerProtocol,
    prompt: str,
    layers: Sequence[CollageLayer],
) -> tuple[tuple[int, ...], ...]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    spans: list[tuple[int, ...]] = []
    textual_inversion_enabled = any(
        layer.textual_inversion_token is not None for layer in layers
    )

    for layer in layers:
        try:
            layer_ids = tokenizer.encode(f" {layer.prompt}", add_special_tokens=False)
            span = _find_token_span(prompt_ids, layer_ids)
        except ValueError:
            layer_ids = tokenizer.encode(layer.prompt, add_special_tokens=False)
            span = _find_token_span(prompt_ids, layer_ids)

        if textual_inversion_enabled:
            # Extend each span's start one token left to include a prepended TI token.
            span = tuple(range(span[0] - 1, span[-1] + 1))

        spans.append(span)

    return tuple(spans)


def _inject_textual_inversion_tokens(
    prompt: str,
    layers: Sequence[CollageLayer],
    embedding_sources: Mapping[str, str | Path] | None,
) -> str:
    requested_tokens = {
        layer.textual_inversion_token
        for layer in layers
        if layer.textual_inversion_token is not None
    }
    supplied_tokens = set() if embedding_sources is None else set(embedding_sources)
    unused_tokens = supplied_tokens - requested_tokens
    if unused_tokens:
        raise ValueError(
            f"textual inversion embeddings have no matching layer: {sorted(unused_tokens)}"
        )

    for layer in layers:
        token = layer.textual_inversion_token
        if token is None:
            continue

        if prompt.count(layer.prompt) != 1:
            raise ValueError(
                "each layer description must occur exactly once in the scene prompt"
            )

        prompt = prompt.replace(layer.prompt, f"{token} {layer.prompt}", 1)

    return prompt


def _resize_mask_to_query(
    mask: Float[np.ndarray, "height width"],
    *,
    size: tuple[int, int],
    device: torch.device,
) -> Float[torch.Tensor, "query"]:
    """Return one flattened mask value for each spatial attention query."""
    mask_tensor = torch.as_tensor(mask, device=device, dtype=torch.float64)
    batched_mask = mask_tensor.unsqueeze(0).unsqueeze(0)
    resized_mask = F.interpolate(
        batched_mask,
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    resized_mask = resized_mask.squeeze(0).squeeze(0)
    query_length = size[0] * size[1]

    return resized_mask.reshape(query_length)


class _CollageCrossAttentionProcessor:
    """Apply visible-layer logit guidance through Diffusers' processor API."""

    def __init__(
        self,
        *,
        visible_masks: Sequence[Float[np.ndarray, "height width"]],
        token_spans: Sequence[Sequence[int]],
        layers: Sequence[CollageLayer],
        output_size: tuple[int, int],
        batch_size: int,
        classifier_free_guidance: bool,
    ) -> None:
        self.visible_masks = tuple(visible_masks)
        self.token_spans = tuple(tuple(span) for span in token_spans)
        self.layers = tuple(layers)
        self.output_size = output_size
        self.batch_size = batch_size
        self.classifier_free_guidance = classifier_free_guidance
        self.sigma = 1.0

    def __call__(
        self,
        attn: Attention,
        hidden_states: Float[torch.Tensor, "batch sequence channels"],
        encoder_hidden_states: Float[torch.Tensor, "batch tokens channels"]
        | None = None,
        attention_mask: Float[torch.Tensor, "..."] | None = None,
        temb: Float[torch.Tensor, "..."] | None = None,
        scale: float | None = None,
    ) -> Float[torch.Tensor, "batch sequence channels"]:
        # Add per-layer score bias before softmax; CFG leaves unconditional rows unchanged.
        del scale
        residual = hidden_states
        spatial_dimensions: tuple[int, int, int] | None = None
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            spatial_dimensions = (channel, height, width)
            hidden_states = hidden_states.view(batch_size, channel, height * width)
            hidden_states = hidden_states.transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = attn.group_norm(hidden_states)
            hidden_states = hidden_states.transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        to_k = attn.to_k
        to_v = attn.to_v
        to_out = attn.to_out
        if to_k is None or to_v is None or to_out is None:
            raise ValueError(
                "cross-attention requires key, value, and output projections"
            )

        key = to_k(encoder_hidden_states)
        value = to_v(encoder_hidden_states)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        query = query.transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        key = key.transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim)
        value = value.transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)

        if attn.norm_k is not None:
            key = attn.norm_k(key)

        score_query = query.float() if attn.upcast_attention else query
        score_key = key.float() if attn.upcast_attention else key
        scores = torch.matmul(score_query, score_key.transpose(-1, -2)) * attn.scale
        if attention_mask is not None:
            scores = scores + attention_mask

        self._apply_layer_guidance(scores)
        softmax_scores = scores.float() if attn.upcast_softmax else scores
        attention_probs = softmax_scores.softmax(dim=-1).to(value.dtype)
        output = torch.matmul(attention_probs, value)
        output = output.transpose(1, 2)
        output = output.reshape(batch_size, -1, attn.heads * head_dim)
        output = output.to(query.dtype)
        output = to_out[0](output)
        output = to_out[1](output)

        if spatial_dimensions is not None:
            channel, height, width = spatial_dimensions
            output = output.transpose(-1, -2)
            output = output.reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            output = output + residual

        return output / attn.rescale_output_factor

    def _apply_layer_guidance(
        self, scores: Float[torch.Tensor, "batch heads query tokens"]
    ) -> None:
        query_length = scores.shape[-2]
        output_height, output_width = self.output_size
        target_height = max(
            1, int(round(math.sqrt(query_length * output_height / output_width)))
        )
        while query_length % target_height:
            target_height -= 1

        target_width = query_length // target_height
        latent_masks = [
            _resize_mask_to_query(
                mask,
                size=(target_height, target_width),
                device=scores.device,
            )
            for mask in self.visible_masks
        ]
        # CFG batches unconditional rows before conditional rows; guide only the latter.
        cfg_offset = self.batch_size if self.classifier_free_guidance else 0
        sigma_scale = math.log(math.log(max(self.sigma, 0.0) + 1.01) + 1.0)

        for batch_index in range(self.batch_size):
            conditional_index = batch_index + cfg_offset
            for mask, span, layer in zip(
                latent_masks, self.token_spans, self.layers, strict=True
            ):
                for token_index in span:
                    if token_index >= scores.shape[-1]:
                        raise ValueError(
                            "layer token span exceeds the text encoder sequence length"
                        )

                    token_scores = scores[conditional_index, :, :, token_index]
                    max_score = token_scores.max()
                    visible_mask = mask.unsqueeze(0)
                    non_visible_mask = 1.0 - visible_mask
                    positive_bias = visible_mask * layer.positive_attention_weight
                    negative_bias = non_visible_mask * layer.negative_attention_weight
                    bias = (positive_bias - negative_bias) * sigma_scale * max_score
                    scores[conditional_index, :, :, token_index] += bias


class CollageDiffusionPipeline(StableDiffusionImg2ImgPipeline):
    """Harmonize positioned image layers with layer-aware diffusion controls.

    The pipeline loads the standard Stable Diffusion image-to-image components and
    replaces only the sampling orchestration needed for visible-layer attention
    and noise guidance.

    Examples:
        >>> from PIL import Image
        >>> from collagediffusion import CollageLayer
        >>> layer = CollageLayer(Image.new("RGBA", (16, 16), (220, 80, 30, 255)), "orange toy")
        >>> layer.prompt
        'orange toy'
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        layers: Sequence[CollageLayer],
        *,
        negative_prompt: str | None = None,
        strength: float = 0.7,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        textual_inversion_embeddings: Mapping[str, str | Path] | None = None,
        height: int = 512,
        width: int = 512,
        noise_blur: float = 10.0,
        eta: float = 0.0,
        output_type: Literal["pil", "np"] = "pil",
        return_dict: bool = True,
        step_callback: Callable[
            [int, float, Float[torch.Tensor, "batch channels height width"]], None
        ]
        | None = None,
    ) -> (
        StableDiffusionPipelineOutput
        | tuple[
            list[Image.Image] | Float[np.ndarray, "batch height width channels"],
            list[bool] | None,
        ]
    ):
        """Generate a harmonized collage from an ordered set of RGBA layers.

        Args:
            prompt: Global scene description containing every layer description.
            layers: Back-to-front RGBA layers with descriptions and controls.
            negative_prompt: Optional text to suppress during classifier-free guidance.
            strength: Global image-to-image strength in ``(0, 1]``.
            num_inference_steps: Number of scheduler steps before strength adjustment.
            guidance_scale: Classifier-free guidance scale.
            num_images_per_prompt: Number of images to generate for the prompt.
            generator: Explicit random generator; takes precedence over ``seed``.
            seed: Optional local seed used when ``generator`` is absent.
            textual_inversion_embeddings: Map from layer token to caller-provided weights.
            height: Output canvas height; must be divisible by the VAE scale factor.
            width: Output canvas width; must be divisible by the VAE scale factor.
            noise_blur: Gaussian blur radius for the blended per-layer noise map.
            eta: Optional scheduler noise parameter used by DDIM-compatible schedulers.
            output_type: ``"pil"`` or ``"np"`` image output format.
            return_dict: Whether to return ``StableDiffusionPipelineOutput``.
            step_callback: Optional callback receiving step index, timestep, and latents.

        Returns:
            A Diffusers image output or its standard tuple representation.

        Raises:
            TypeError: If a layer sequence contains an unsupported value.
            ValueError: If prompts, layer controls, or generation settings are invalid.
        """
        # Add layer-aware attention guidance and restore source regions during denoising.
        # 1. Validate prompts, layers, and generation settings.
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")

        if not layers:
            raise ValueError("layers must contain at least one layer")

        if any(not isinstance(layer, CollageLayer) for layer in layers):
            raise TypeError("layers must contain CollageLayer values")

        if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
            raise ValueError("strength must be finite and between 0 and 1")

        if num_inference_steps < 1 or num_images_per_prompt < 1:
            raise ValueError(
                "num_inference_steps and num_images_per_prompt must be positive"
            )

        if not math.isfinite(guidance_scale) or guidance_scale < 0.0:
            raise ValueError("guidance_scale must be finite and non-negative")

        if height < 1 or width < 1:
            raise ValueError("height and width must be positive")

        if output_type not in {"pil", "np"}:
            raise ValueError("output_type must be 'pil' or 'np'")

        device = self._execution_device
        vae_scale_factor = self.vae_scale_factor
        if height % vae_scale_factor or width % vae_scale_factor:
            raise ValueError(
                "height and width must be divisible by the VAE scale factor"
            )

        # 2. Prepare layer prompts, collage image, visible masks, and noise map.
        generator = resolve_torch_generator(generator=generator, seed=seed)
        prompt = _inject_textual_inversion_tokens(
            prompt, layers, textual_inversion_embeddings
        )
        if textual_inversion_embeddings is not None:
            for token, embedding_source in textual_inversion_embeddings.items():
                self.load_textual_inversion(embedding_source, token=token)

        token_spans = _prompt_token_spans(self.tokenizer, prompt, layers)
        collage, visible_masks = compose_collage_layers(
            layers, width=width, height=height
        )
        noise_map = make_layer_noise_map(
            visible_masks, layers, strength=strength, blur_radius=noise_blur
        )
        image = self.image_processor.preprocess(collage, height=height, width=width)
        image = image.to(device=device, dtype=self.unet.dtype)

        # 3. Encode the prompt and batch both guidance branches when guidance_scale > 1.
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                raise ValueError(
                    "classifier-free guidance requires negative prompt embeddings"
                )

            # Concatenate both branches so the U-Net handles them in one call.
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Encode the collage and prepare its initial latent and noise.
        encoded = self.vae.encode(image)
        init_latents = encoded.latent_dist.sample(generator=generator)
        init_latents = init_latents * self.vae.config.scaling_factor
        init_latents = init_latents.repeat(num_images_per_prompt, 1, 1, 1)
        initial_noise = randn(
            init_latents.shape,
            generator=generator,
            device=device,
            dtype=init_latents.dtype,
        )

        # 5. Set img2img timesteps and build the latent-resolution preservation mask.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        scheduler_offset = int(self.scheduler.config.get("steps_offset", 0))
        init_timestep = min(
            int(num_inference_steps * strength) + scheduler_offset,
            num_inference_steps,
        )
        timestep_start = max(
            num_inference_steps - init_timestep + scheduler_offset,
            0,
        )
        timesteps = self.scheduler.timesteps[timestep_start:]
        effective_steps = num_inference_steps - timestep_start
        if len(timesteps) == 0:
            raise ValueError(
                "strength and num_inference_steps produced no denoising steps"
            )

        latent_timestep = timesteps[:1].repeat(num_images_per_prompt)
        latents = self.scheduler.add_noise(init_latents, initial_noise, latent_timestep)
        noise_mask_pixels = np.rint(noise_map * 255.0).astype(np.uint8)
        # Quantize the source strength map before resizing it to the latent grid.
        noise_mask_image = Image.fromarray(noise_mask_pixels)
        noise_mask_image = noise_mask_image.resize(
            (latents.shape[-1], latents.shape[-2]),
            resample=Image.Resampling.NEAREST,
        )
        noise_mask_values = np.asarray(noise_mask_image, dtype=np.float32)
        noise_mask_values = noise_mask_values.copy()
        normalized_noise_mask = noise_mask_values / 255.0
        preservation_mask = 1.0 - normalized_noise_mask
        preservation_mask_tensor = torch.as_tensor(
            preservation_mask, device=device, dtype=latents.dtype
        )
        batched_preservation_mask = preservation_mask_tensor.unsqueeze(0).unsqueeze(0)
        noise_mask = batched_preservation_mask.expand(
            num_images_per_prompt, 1, *latents.shape[-2:]
        )

        # 6. Install layer-aware attention processors and prepare scheduler arguments.
        previous_processors = dict(self.unet.attn_processors)
        controlled_processors = {
            name: _CollageCrossAttentionProcessor(
                visible_masks=visible_masks,
                token_spans=token_spans,
                layers=layers,
                output_size=(height, width),
                batch_size=num_images_per_prompt,
                classifier_free_guidance=do_classifier_free_guidance,
            )
            if name.endswith(".attn2.processor")
            else processor
            for name, processor in previous_processors.items()
        }
        controlled_attention_processors = tuple(
            processor
            for processor in controlled_processors.values()
            if isinstance(processor, _CollageCrossAttentionProcessor)
        )
        attention_control_enabled = any(
            layer.positive_attention_weight or layer.negative_attention_weight
            for layer in layers
        )
        if attention_control_enabled and not controlled_attention_processors:
            raise ValueError("the U-Net exposes no cross-attention processors")

        if attention_control_enabled:
            self.unet.set_attn_processor(controlled_processors)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        self._num_timesteps = len(timesteps)
        try:
            # 7. Denoise, applying attention guidance and layer-specific preservation.
            with self.progress_bar(total=effective_steps) as progress_bar:
                for step_index, timestep in enumerate(timesteps):
                    if attention_control_enabled:
                        # Scheduler sigma controls attention strength; use normalized t if unavailable.
                        sigma = (
                            float(self.scheduler.sigmas[step_index].item())
                            if hasattr(self.scheduler, "sigmas")
                            else float(timestep.item()) / 1000.0
                        )
                        for processor in controlled_attention_processors:
                            processor.sigma = sigma

                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, timestep
                    )
                    noise_prediction = self.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                    ).sample
                    if do_classifier_free_guidance:
                        # Standard CFG: unconditional + scale * (conditional - unconditional).
                        unconditional, conditional = noise_prediction.chunk(2)
                        noise_prediction = unconditional + guidance_scale * (
                            conditional - unconditional
                        )

                    # Advance x_t to x_(t-1); the mask below restores source regions.
                    latents = self.scheduler.step(
                        noise_prediction,
                        timestep,
                        latents,
                        **extra_step_kwargs,
                    ).prev_sample
                    step_noise = randn(
                        latents.shape,
                        generator=generator,
                        device=device,
                        dtype=latents.dtype,
                    )
                    timestep_batch = timestep.expand(latents.shape[0])
                    preserved_latents = self.scheduler.add_noise(
                        init_latents, step_noise, timestep_batch
                    )
                    timestep_value = float(timestep.item())
                    # Apply the timestep-dependent preservation threshold and spread it to neighbors.
                    mask_t = (noise_mask > 1.0 - timestep_value / 1000.0).to(
                        latents.dtype
                    )
                    dilation = int(4.0 * timestep_value / 1000.0)
                    if dilation > 0:
                        mask_t = F.max_pool2d(
                            mask_t,
                            kernel_size=dilation * 2 + 1,
                            stride=1,
                            padding=dilation,
                        )

                    latents = preserved_latents * mask_t + latents * (1.0 - mask_t)
                    if step_callback is not None:
                        step_callback(step_index, timestep_value, latents)

                    progress_bar.update()
        finally:
            if attention_control_enabled:
                self.unet.set_attn_processor(previous_processors)

        # 8. Decode the final latent and return the standard Diffusers output.
        decoded = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        decoded, has_nsfw_concept = self.run_safety_checker(
            decoded, device, prompt_embeds.dtype
        )
        do_denormalize = (
            [not flag for flag in has_nsfw_concept]
            if has_nsfw_concept is not None
            else None
        )
        images = self.image_processor.postprocess(
            decoded, output_type=output_type, do_denormalize=do_denormalize
        )
        self.maybe_free_model_hooks()

        if not return_dict:
            return images, has_nsfw_concept

        return StableDiffusionPipelineOutput(
            images=images,
            nsfw_content_detected=has_nsfw_concept,
        )

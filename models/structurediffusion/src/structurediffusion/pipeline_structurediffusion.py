"""Diffusers pipeline for training-free structured diffusion guidance."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Final, Literal, cast

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from jaxtyping import Float
from PIL import Image

from .modeling_structurediffusion import (
    StructuredAttentionProcessor,
    StructuredConditioning,
)
from .processing_structurediffusion import (
    NounPhrase,
    align_phrase_context,
    constituency_noun_phrases,
    normalize_prompt,
    scene_graph_noun_phrases,
    token_span_from_char_span,
)


class ParserType(StrEnum):
    """Supported linguistic parsers for extracting prompt noun phrases."""

    constituency = auto()
    scene_graph = auto()


_PARSER_TYPES: Final[frozenset[ParserType]] = frozenset(ParserType)


def normalize_parser_type(parser_type: ParserType | str) -> ParserType:
    """Normalize a public parser name and reject unsupported parser choices."""
    if isinstance(parser_type, ParserType):
        normalized = parser_type
    else:
        try:
            normalized = ParserType(parser_type.lower().replace("-", "_"))
        except ValueError as exc:
            choices = ", ".join(parser.value for parser in ParserType)
            raise ValueError(
                f"Unsupported parser_type {parser_type!r}; choose one of: {choices}."
            ) from exc

    if normalized not in _PARSER_TYPES:
        raise ValueError(f"Unsupported parser_type: {parser_type}")

    return normalized


@dataclass(frozen=True)
class _StructuredCall:
    prompts: tuple[str, ...]
    parser_type: ParserType
    conjunction: bool
    save_attention_maps: bool


OutputType = Literal["pil", "np", "latent"]


class StructureDiffusionPipeline(StableDiffusionPipeline):
    """Generate images with noun-phrase-aware cross-attention.

    The pipeline retains Stable Diffusion's component loading and sampling API.
    Its three method options control noun-phrase extraction, the paper's
    conjunction key/value arrangement, and optional attention-map collection.

    Args:
        vae: Latent image decoder.
        text_encoder: CLIP text encoder.
        tokenizer: CLIP tokenizer.
        unet: Stable Diffusion denoising network.
        scheduler: Diffusion scheduler.
        safety_checker: Optional output safety checker.
        feature_extractor: Optional safety-checker image processor.
        image_encoder: Optional image encoder.
        requires_safety_checker: Whether the safety checker is required.

    Examples:
        >>> from structurediffusion import StructureDiffusionPipeline
        >>> pipe = StructureDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        >>> pipe("A red car and a white sheep", num_inference_steps=10).images[0]
    """

    attention_maps: tuple[tuple[str, Float[torch.Tensor, "..."]], ...] = ()

    def encode_prompt(
        self,
        prompt: str | list[str] | None,
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: Float[torch.Tensor, "batch tokens dim"] | None = None,
        negative_prompt_embeds: Float[torch.Tensor, "batch tokens dim"] | None = None,
        lora_scale: float | None = None,
        clip_skip: int | None = None,
    ) -> tuple[
        Float[torch.Tensor, "batch tokens dim"],
        Float[torch.Tensor, "batch tokens dim"] | None,
    ]:
        """Encode standard prompts and attach parsed key/value contexts per call."""
        prompt_embeds, negative_prompt_embeds = super().encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=clip_skip,
        )

        request = getattr(self, "_call_request", None)
        if request is not None:
            self._install_conditioning(
                request,
                prompt_embeds,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                lora_scale,
                clip_skip,
            )

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        parser_type: ParserType | str = ParserType.constituency,
        conjunction: bool = False,
        save_attention_maps: bool = False,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float = 7.5,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: Float[torch.Tensor, "batch channels latent_height latent_width"]
        | None = None,
        prompt_embeds: Float[torch.Tensor, "batch tokens dim"] | None = None,
        negative_prompt_embeds: Float[torch.Tensor, "batch tokens dim"] | None = None,
        ip_adapter_image: PipelineImageInput | None = None,
        ip_adapter_image_embeds: list[Float[torch.Tensor, "..."]] | None = None,
        output_type: OutputType = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: dict[str, float | bool] | None = None,
        guidance_rescale: float = 0.0,
        clip_skip: int | None = None,
        callback_on_step_end: (
            PipelineCallback
            | MultiPipelineCallbacks
            | Callable[
                [
                    StableDiffusionPipeline,
                    int,
                    int,
                    dict[str, Float[torch.Tensor, "..."]],
                ],
                dict[str, Float[torch.Tensor, "..."]],
            ]
            | None
        ) = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ) -> (
        StableDiffusionPipelineOutput
        | tuple[
            Float[torch.Tensor, "batch channels height width"]
            | Float[np.ndarray, "batch height width channels"]
            | list[Image.Image],
            list[bool] | None,
        ]
    ):
        """Generate one or more images using structured cross-attention.

        Args:
            prompt: Prompt text or prompt batch to parse and render.
            parser_type: Constituency or scene-graph noun-phrase parser.
            conjunction: Use separate aligned noun phrases for both keys and values.
            save_attention_maps: Keep conditional attention maps on ``attention_maps``.
            height: Requested output image height.
            width: Requested output image width.
            num_inference_steps: Number of scheduler denoising steps.
            timesteps: Optional explicit scheduler timesteps.
            sigmas: Optional explicit scheduler noise levels.
            guidance_scale: Classifier-free guidance scale.
            negative_prompt: Optional negative prompt or prompt batch.
            num_images_per_prompt: Number of outputs to generate for each prompt.
            eta: Scheduler stochasticity parameter when supported.
            generator: Optional random generator or per-output generator list.
            latents: Optional initial latent tensor.
            prompt_embeds: Optional full-prompt CLIP embeddings.
            negative_prompt_embeds: Optional negative-prompt embeddings.
            ip_adapter_image: Optional IP-Adapter image input.
            ip_adapter_image_embeds: Optional precomputed IP-Adapter embeddings.
            output_type: Diffusers output format.
            return_dict: Whether to return ``StableDiffusionPipelineOutput``.
            cross_attention_kwargs: Optional Diffusers cross-attention settings.
            guidance_rescale: Optional classifier-free guidance rescaling.
            clip_skip: Number of CLIP layers to skip.
            callback_on_step_end: Optional Diffusers step callback.
            callback_on_step_end_tensor_inputs: Tensors exposed to the callback.

        Returns:
            The standard Stable Diffusion output container or tuple.

        Raises:
            ValueError: If the parser or prompt batch is invalid.
        """
        normalized_parser = normalize_parser_type(parser_type)
        prompt_items = [prompt] if isinstance(prompt, str) else prompt
        normalized_prompts = tuple(normalize_prompt(item) for item in prompt_items)
        if not normalized_prompts or any(not item for item in normalized_prompts):
            raise ValueError("prompt must contain at least one non-empty string.")

        if num_images_per_prompt < 1:
            raise ValueError("num_images_per_prompt must be positive.")

        request = _StructuredCall(
            prompts=normalized_prompts,
            parser_type=normalized_parser,
            conjunction=conjunction,
            save_attention_maps=save_attention_maps,
        )
        self.__dict__["attention_maps"] = ()
        previous_processors = cast(
            dict[str, Callable[..., Float[torch.Tensor, "..."]]],
            dict(self.unet.attn_processors),
        )
        active_processors: list[StructuredAttentionProcessor] = []
        replacement_processors: dict[
            str, Callable[..., Float[torch.Tensor, "..."]]
        ] = {}
        for name, processor in previous_processors.items():
            if name.endswith("attn2.processor"):
                structured_processor = StructuredAttentionProcessor(
                    name,
                    save_attention_maps,
                )
                active_processors.append(structured_processor)
                replacement_processors[name] = structured_processor
            else:
                replacement_processors[name] = processor

        if not active_processors:
            raise RuntimeError(
                "The U-Net has no cross-attention processors to replace."
            )

        self.unet.set_attn_processor(replacement_processors)
        self._call_request = request
        self._processors_for_call = active_processors
        try:
            result = super().__call__(
                prompt=(
                    normalized_prompts[0]
                    if isinstance(prompt, str)
                    else list(normalized_prompts)
                ),
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                output_type=output_type,
                return_dict=return_dict,
                cross_attention_kwargs=cross_attention_kwargs,
                guidance_rescale=guidance_rescale,
                clip_skip=clip_skip,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            )
            self.__dict__["attention_maps"] = tuple(
                (processor.layer_name, attention_map)
                for processor in active_processors
                for attention_map in processor.attention_maps
            )
            return result
        finally:
            del self._call_request
            del self._processors_for_call
            self.unet.set_attn_processor(dict(previous_processors))

    def _install_conditioning(
        self,
        request: _StructuredCall,
        prompt_embeds: Float[torch.Tensor, "batch tokens dim"],
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        lora_scale: float | None,
        clip_skip: int | None,
    ) -> None:
        batch_size = len(request.prompts)
        expected_batch = batch_size * num_images_per_prompt
        if prompt_embeds.shape[0] != expected_batch:
            raise ValueError(
                "The encoded prompt batch does not match prompt and num_images_per_prompt."
            )

        base_contexts = prompt_embeds.reshape(
            batch_size,
            num_images_per_prompt,
            prompt_embeds.shape[1],
            prompt_embeds.shape[2],
        )[:, 0]
        phrase_groups = [
            self._parse_prompt(prompt_text, request.parser_type)
            for prompt_text in request.prompts
        ]
        phrase_texts = [phrase.text for group in phrase_groups for phrase in group]
        phrase_contexts: Float[torch.Tensor, "phrases tokens dim"] | None = None
        if phrase_texts:
            phrase_contexts, _ = StableDiffusionPipeline.encode_prompt(
                self,
                prompt=phrase_texts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                lora_scale=lora_scale,
                clip_skip=clip_skip,
            )

        aligned_groups: list[
            list[
                tuple[
                    Float[torch.Tensor, "batch tokens dim"],
                    tuple[int, int],
                ]
            ]
        ] = []
        phrase_cursor = 0
        for prompt_index, phrase_group in enumerate(phrase_groups):
            aligned: list[
                tuple[
                    Float[torch.Tensor, "batch tokens dim"],
                    tuple[int, int],
                ]
            ] = []
            for phrase in phrase_group:
                phrase_embedding = (
                    phrase_contexts[phrase_cursor : phrase_cursor + 1]
                    if phrase_contexts is not None
                    else base_contexts[prompt_index : prompt_index + 1]
                )
                phrase_cursor += 1
                token_span = token_span_from_char_span(
                    self.tokenizer,
                    request.prompts[prompt_index],
                    phrase.start,
                    phrase.end,
                    prompt_embeds.shape[1],
                )
                if token_span is None:
                    continue

                phrase_token_count = len(self.tokenizer.tokenize(phrase.text))
                if phrase_token_count < token_span[1] - token_span[0]:
                    continue

                aligned.append(
                    (
                        align_phrase_context(
                            base_contexts[prompt_index : prompt_index + 1],
                            phrase_embedding,
                            token_span,
                            phrase_token_count,
                        ),
                        token_span,
                    )
                )

            aligned_groups.append(aligned)

        max_phrase_count = max((len(group) for group in aligned_groups), default=0)
        aligned_contexts = tuple(
            torch.cat(
                [
                    group[index][0]
                    if index < len(group)
                    else base_contexts[prompt_index : prompt_index + 1]
                    for prompt_index, group in enumerate(aligned_groups)
                ],
                dim=0,
            )
            for index in range(max_phrase_count)
        )
        aligned_validity = tuple(
            torch.tensor(
                [index < len(group) for group in aligned_groups],
                dtype=torch.bool,
                device=device,
            )
            for index in range(max_phrase_count)
        )
        base_validity = torch.ones(batch_size, dtype=torch.bool, device=device)
        if request.conjunction:
            combined_contexts: list[Float[torch.Tensor, "batch tokens dim"]] = []
            for prompt_index, base_context in enumerate(base_contexts):
                combined = base_context.clone()
                for aligned_context, (start, end) in aligned_groups[prompt_index]:
                    combined[start:end] = aligned_context[0, start:end]

                combined_contexts.append(combined)

            values = (torch.stack(combined_contexts),)
            keys = (base_contexts, *aligned_contexts)
            key_validity = (base_validity, *aligned_validity)
            value_validity = (base_validity,)
        else:
            keys = (base_contexts,)
            values = (base_contexts, *aligned_contexts)
            key_validity = (base_validity,)
            value_validity = (base_validity, *aligned_validity)

        repeated_keys = tuple(
            context.repeat_interleave(num_images_per_prompt, dim=0) for context in keys
        )
        repeated_values = tuple(
            context.repeat_interleave(num_images_per_prompt, dim=0)
            for context in values
        )
        repeated_key_validity = tuple(
            validity.repeat_interleave(num_images_per_prompt, dim=0)
            for validity in key_validity
        )
        repeated_value_validity = tuple(
            validity.repeat_interleave(num_images_per_prompt, dim=0)
            for validity in value_validity
        )
        conditioning = StructuredConditioning(
            keys=repeated_keys,
            values=repeated_values,
            key_validity=repeated_key_validity,
            value_validity=repeated_value_validity,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        for processor in getattr(self, "_processors_for_call", ()):
            processor.conditioning = conditioning

    @staticmethod
    def _parse_prompt(prompt: str, parser_type: ParserType) -> list[NounPhrase]:
        if parser_type is ParserType.constituency:
            return constituency_noun_phrases(prompt)

        if parser_type is ParserType.scene_graph:
            return scene_graph_noun_phrases(prompt)

        raise ValueError(f"Unsupported parser_type: {parser_type}")

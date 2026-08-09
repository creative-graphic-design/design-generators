"""Pipeline wrapper for PosterLLaVA image-conditioned layout generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Literal, Protocol, TypeGuard, cast

import torch
from jaxtyping import Bool, Float, Int
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer  # ty: ignore[possibly-missing-import]
from transformers import ImageProcessingMixin
from transformers import PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig
from transformers import StoppingCriteriaList

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import LayoutGenerationPipeline, PipelineComponentSpec

from .configuration_posterllava import (
    ConversationMode,
    OutputType,
    PosterLlavaConfig,
)
from .generation_posterllava import build_stopping_criteria
from .image_processing_posterllava import PosterLlavaImageProcessor
from .processing_posterllava import (
    PosterLlavaImageProcessorComponent,
    PosterLlavaOutputDict,
    PosterLlavaProcessor,
)

PosterLlavaContentScalar = str | int | float | bool | None
PosterLlavaContentValue = (
    PosterLlavaContentScalar
    | Image.Image
    | Sequence["PosterLlavaContentValue"]
    | Mapping[str, "PosterLlavaContentValue"]
)
PosterLlavaComponent = (
    PreTrainedModel
    | PreTrainedTokenizerBase
    | PosterLlavaImageProcessorComponent
    | PosterLlavaProcessor
)


class _CausalLMGenerationModel(Protocol):
    def generate(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        *,
        images: Float[torch.Tensor, "batch channels height width"] | None = None,
        attention_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_beams: int | None = None,
        generator: torch.Generator | None = None,
        **generate_kwargs: StoppingCriteriaList | None,
    ) -> Int[torch.Tensor, "batch generated_tokens"]:
        """Generate token ids."""


class _ImagePreprocessor(Protocol):
    def preprocess(
        self,
        images: Sequence[Image.Image],
        *,
        return_tensors: str = "pt",
    ) -> Mapping[str, Float[torch.Tensor, "batch channels height width"]]:
        """Preprocess images."""


def _load_model_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> PreTrainedModel:
    kwargs: dict[str, bool | str] = {"local_files_only": local_files_only}
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    return cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs),
    )


def _load_tokenizer_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> PreTrainedTokenizerBase:
    kwargs: dict[str, bool | str] = {"local_files_only": local_files_only}
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    return cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs),
    )


def _load_image_processor_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> ImageProcessingMixin:
    kwargs: dict[str, bool | str] = {"local_files_only": local_files_only}
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    return cast(
        ImageProcessingMixin,
        AutoImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs),
    )


def _load_processor_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> PosterLlavaProcessor:
    return PosterLlavaProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
        subfolder=subfolder,
    )


def _is_content_mapping(
    content: Mapping[str, PosterLlavaContentValue]
    | Sequence[Mapping[str, PosterLlavaContentValue]]
    | None,
) -> TypeGuard[Mapping[str, PosterLlavaContentValue]]:
    return isinstance(content, Mapping)


def _is_content_sequence(
    content: Mapping[str, PosterLlavaContentValue]
    | Sequence[Mapping[str, PosterLlavaContentValue]]
    | None,
) -> TypeGuard[Sequence[Mapping[str, PosterLlavaContentValue]]]:
    return isinstance(content, Sequence)


class PosterLlavaPipeline(LayoutGenerationPipeline):
    """Generate poster layouts with a LLaVA-style causal LM checkpoint.

    Args:
        config: PosterLLaVA recipe configuration.
        processor: Prompt and JSON layout processor.
        model: Optional upstream causal LM component.
        tokenizer: Optional LLaVA tokenizer component.
        image_processor: Optional CLIP image processor component.

    Examples:
        >>> cfg = PosterLlavaConfig(dataset_name="ad_banner")
        >>> processor = PosterLlavaProcessor.from_config()
        >>> pipe = PosterLlavaPipeline(cfg, processor)
        >>> pipe.config.model_type
        'posterllava'
    """

    config_class: ClassVar[type[PretrainedConfig]] = PosterLlavaConfig
    component_specs: ClassVar[dict[str, PipelineComponentSpec]] = {
        "model": PipelineComponentSpec(
            attribute_name="model",
            loader=_load_model_component,
            config_subfolder_attribute="model_subfolder",
            required=False,
            marker_file="config.json",
        ),
        "tokenizer": PipelineComponentSpec(
            attribute_name="tokenizer",
            loader=_load_tokenizer_component,
            config_subfolder_attribute="tokenizer_subfolder",
            required=False,
            marker_file="tokenizer_config.json",
            save_with_is_main_process=False,
        ),
        "image_processor": PipelineComponentSpec(
            attribute_name="image_processor",
            loader=_load_image_processor_component,
            config_subfolder_attribute="image_processor_subfolder",
            required=False,
            marker_file="preprocessor_config.json",
            save_with_is_main_process=False,
        ),
        "processor": PipelineComponentSpec(
            attribute_name="processor",
            loader=_load_processor_component,
            config_subfolder_attribute="processor_subfolder",
            required=False,
            marker_file="processor_config.json",
            save_with_is_main_process=False,
        ),
    }

    config: PosterLlavaConfig
    processor: PosterLlavaProcessor
    model: PreTrainedModel | None
    tokenizer: PreTrainedTokenizerBase | None
    image_processor: PosterLlavaImageProcessorComponent | None

    def __init__(
        self,
        config: PosterLlavaConfig,
        processor: PosterLlavaProcessor,
        *,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        image_processor: PosterLlavaImageProcessorComponent | None = None,
    ) -> None:
        """Initialize the PosterLLaVA recipe pipeline."""
        super().__init__(config)
        self.config = config
        self.processor = processor
        self.model = model
        self.tokenizer = tokenizer or processor.tokenizer
        self.image_processor = image_processor or processor.image_processor
        if self.tokenizer is not None:
            self.processor.tokenizer = self.tokenizer
        if self.image_processor is not None:
            self.processor.image_processor = self.image_processor

    @classmethod
    def _from_pretrained_components(
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, PosterLlavaComponent | None],
    ) -> "PosterLlavaPipeline":
        """Build a pipeline from loaded root config and components."""
        cfg = cast(PosterLlavaConfig, config)
        processor = cast(PosterLlavaProcessor | None, components.get("processor"))
        if processor is None:
            processor = PosterLlavaProcessor.from_config(
                dataset_name=cfg.dataset_name,
                id2label=cfg.id2label,
                prompt_template=cfg.prompt_template,
            )
        tokenizer = cast(PreTrainedTokenizerBase | None, components.get("tokenizer"))
        image_processor = components.get("image_processor")
        return cls(
            config=cfg,
            processor=processor,
            model=cast(PreTrainedModel | None, components.get("model")),
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

    def __call__(
        self,
        *,
        images: Image.Image | Sequence[Image.Image] | None = None,
        prompt: str | Sequence[str] | None = None,
        content: Mapping[str, PosterLlavaContentValue]
        | Sequence[Mapping[str, PosterLlavaContentValue]]
        | None = None,
        texts: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.content_image,
        labels: Int[torch.Tensor, "batch elements"] | Sequence[str | int] | None = None,
        bbox: Float[torch.Tensor, "batch elements 4"]
        | Sequence[Sequence[float]]
        | None = None,
        mask: Bool[torch.Tensor, "batch elements"] | Sequence[bool] | None = None,
        num_elements: int | Sequence[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: OutputType | Literal["dataclass", "dict"] = OutputType.dataclass,
        return_intermediates: bool = False,
        max_new_tokens: int | None = None,
        do_sample: bool = True,
        temperature: float | None = None,
        top_p: float | None = 1.0,
        top_k: int | None = None,
        num_beams: int | None = 1,
        conv_mode: ConversationMode | str | None = None,
        domain_name: str = "social media promotion poster with qbposter style",
    ) -> LayoutGenerationOutput | PosterLlavaOutputDict:  # ty: ignore[invalid-method-override]
        """Generate a poster layout from an image-conditioned prompt.

        Args:
            images: Poster/background image or image batch.
            prompt: Optional prompt body override.
            content: Optional payload mapping containing ``image``, ``text``,
                ``texts``, ``num_elements``, or ``json_data``.
            texts: Optional aligned text payload.
            batch_size: Expected batch size when scalar inputs are provided.
            seed: Seed used only when ``generator`` is absent.
            generator: Explicit generator passed to model generation.
            condition_type: Canonical condition type. Only ``content_image`` is
                supported in the first package version.
            labels: Optional initial labels.
            bbox: Optional initial boxes aligned with labels.
            mask: Optional initial valid-element mask.
            num_elements: Requested element count.
            box_format: Public input box format.
            normalized: Whether input boxes are normalized.
            canvas_size: Pixel canvas size for non-normalized input boxes.
            num_inference_steps: Accepted for shared-interface compatibility.
            output_type: Output container mode.
            return_intermediates: Whether raw prompts/text are returned.
            max_new_tokens: Token budget for generation.
            do_sample: Whether to sample from the LLM.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            num_beams: Beam count.
            conv_mode: Optional LLaVA conversation template override.
            domain_name: Domain phrase inserted into the default prompt.

        Returns:
            Layout output dataclass or dictionary.

        Raises:
            NotImplementedError: If ``condition_type`` is unsupported.
            ValueError: If required image/model/tokenizer components are absent.
        """
        _ = num_inference_steps
        condition = normalize_condition_type(condition_type)
        if condition is not ConditionType.content_image:
            raise NotImplementedError(
                "PosterLLaVA only supports condition_type='content_image'"
            )
        image_list = self._resolve_images(images=images, content=content)
        prompts = self._build_prompts(
            prompt=prompt,
            content=content,
            texts=texts,
            batch_size=batch_size,
            labels=labels,
            bbox=bbox,
            mask=mask,
            num_elements=num_elements,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
            conv_mode=conv_mode,
            domain_name=domain_name,
        )
        if len(image_list) != len(prompts):
            raise ValueError("images and prompts must resolve to the same batch size")
        if self.model is None:
            raise ValueError("model is required for PosterLLaVA generation")
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for PosterLLaVA generation")
        if self.image_processor is None:
            raise ValueError("image_processor is required for PosterLLaVA generation")
        encoded = self.processor(prompts)
        input_ids = cast(Int[torch.Tensor, "batch tokens"], encoded["input_ids"])
        attention_mask = cast(
            Bool[torch.Tensor, "batch tokens"],
            encoded["attention_mask"],
        )
        pixel_values = self._preprocess_images(image_list)
        generation_generator = self.prepare_generator(generator=generator, seed=seed)
        sequences = cast(_CausalLMGenerationModel, self.model).generate(
            input_ids=input_ids,
            images=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature
            if temperature is not None
            else self.config.default_temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            generator=generation_generator,
            stopping_criteria=build_stopping_criteria(
                self.tokenizer,
                input_ids=input_ids,
            ),
        )
        generated_text = self.tokenizer.batch_decode(
            sequences[:, input_ids.shape[-1] :],
            skip_special_tokens=True,
        )
        return self.processor.decode_layout(
            generated_text,
            output_type=output_type,
            return_intermediates=return_intermediates,
            sequences=sequences,
            prompts=prompts,
        )

    generate = __call__

    def _resolve_images(
        self,
        *,
        images: Image.Image | Sequence[Image.Image] | None,
        content: Mapping[str, PosterLlavaContentValue]
        | Sequence[Mapping[str, PosterLlavaContentValue]]
        | None,
    ) -> list[Image.Image]:
        if images is None:
            if content is None:
                raise ValueError("images or content['image'] is required")
            content_items = [content] if isinstance(content, Mapping) else list(content)
            raw_images = [item.get("image") for item in content_items]
        else:
            raw_images = [images] if isinstance(images, Image.Image) else list(images)
        if not raw_images or any(
            not isinstance(item, Image.Image) for item in raw_images
        ):
            raise ValueError("PosterLLaVA images must be PIL.Image.Image instances")
        return cast(list[Image.Image], raw_images)

    def _build_prompts(
        self,
        *,
        prompt: str | Sequence[str] | None,
        content: Mapping[str, PosterLlavaContentValue]
        | Sequence[Mapping[str, PosterLlavaContentValue]]
        | None,
        texts: str | Sequence[str] | Sequence[Sequence[str]] | None,
        batch_size: int,
        labels: Int[torch.Tensor, "batch elements"] | Sequence[str | int] | None,
        bbox: Float[torch.Tensor, "batch elements 4"]
        | Sequence[Sequence[float]]
        | None,
        mask: Bool[torch.Tensor, "batch elements"] | Sequence[bool] | None,
        num_elements: int | Sequence[int] | Int[torch.Tensor, "batch"] | None,
        box_format: BoxFormat | str,
        normalized: bool,
        canvas_size: tuple[int, int] | None,
        conv_mode: ConversationMode | str | None,
        domain_name: str,
    ) -> list[str]:
        prompt_items = self._broadcast_prompt(prompt, batch_size=batch_size)
        content_items = self._broadcast_content(content, batch_size=len(prompt_items))
        count_items = self._resolve_num_elements(
            num_elements=num_elements,
            content_items=content_items,
            batch_size=len(prompt_items),
        )
        prompts: list[str] = []
        for idx, count in enumerate(count_items):
            content_texts = self._texts_for_index(texts, content_items, idx)
            initial = self.processor.build_initial_json(
                labels=cast(
                    Sequence[str | int] | Int[torch.Tensor, "elements"] | None,
                    self._slice_optional(labels, idx),
                ),
                bbox=cast(
                    Float[torch.Tensor, "elements 4"]
                    | Sequence[Sequence[float]]
                    | None,
                    self._slice_optional(bbox, idx),
                ),
                mask=cast(
                    Bool[torch.Tensor, "elements"] | Sequence[bool] | None,
                    self._slice_optional(mask, idx),
                ),
                box_format=box_format,
                normalized=normalized,
                canvas_size=canvas_size,
            )
            prompts.append(
                self.processor.build_prompt(
                    num_elements=count,
                    canvas_size=canvas_size,
                    elements=initial,
                    domain_name=domain_name,
                    conv_mode=conv_mode or self.config.default_conv_mode,
                    prompt=prompt_items[idx],
                    texts=content_texts,
                )
            )
        return prompts

    def _preprocess_images(
        self,
        images: Sequence[Image.Image],
    ) -> Float[torch.Tensor, "batch channels height width"]:
        image_processor = self.image_processor
        if isinstance(image_processor, PosterLlavaImageProcessor):
            processed = image_processor.preprocess(
                list(images),
                image_aspect_ratio=cast(Literal["pad"], self.config.image_aspect_ratio),
                return_tensors="pt",
            )
        elif hasattr(image_processor, "preprocess"):
            processed = cast(_ImagePreprocessor, image_processor).preprocess(
                list(images),
                return_tensors="pt",
            )
        else:
            raise ValueError("image_processor must provide preprocess")
        return cast(
            Float[torch.Tensor, "batch channels height width"],
            processed["pixel_values"],
        )

    def _broadcast_prompt(
        self,
        prompt: str | Sequence[str] | None,
        *,
        batch_size: int,
    ) -> list[str | None]:
        if prompt is None:
            return [None] * batch_size
        if isinstance(prompt, str):
            return [prompt] * batch_size
        return list(prompt)

    def _broadcast_content(
        self,
        content: Mapping[str, PosterLlavaContentValue]
        | Sequence[Mapping[str, PosterLlavaContentValue]]
        | None,
        *,
        batch_size: int,
    ) -> list[Mapping[str, PosterLlavaContentValue]]:
        if content is None:
            return [{} for _ in range(batch_size)]
        if _is_content_mapping(content):
            return [content for _ in range(batch_size)]
        if _is_content_sequence(content):
            return list(content)
        raise TypeError("content must be a mapping, a sequence of mappings, or None")

    def _resolve_num_elements(
        self,
        *,
        num_elements: int | Sequence[int] | Int[torch.Tensor, "batch"] | None,
        content_items: Sequence[Mapping[str, PosterLlavaContentValue]],
        batch_size: int,
    ) -> list[int]:
        if num_elements is None:
            values = [
                item.get("num_elements") or item.get("elements")
                for item in content_items
            ]
            if any(value is None for value in values):
                raise ValueError("num_elements is required for PosterLLaVA generation")
            return [int(cast(int | str, value)) for value in values]
        if isinstance(num_elements, int):
            return [num_elements] * batch_size
        if isinstance(num_elements, torch.Tensor):
            return [int(value) for value in num_elements.tolist()]
        return [int(value) for value in num_elements]

    def _texts_for_index(
        self,
        texts: str | Sequence[str] | Sequence[Sequence[str]] | None,
        content_items: Sequence[Mapping[str, PosterLlavaContentValue]],
        idx: int,
    ) -> str | Sequence[str] | None:
        content_value = content_items[idx].get("texts", content_items[idx].get("text"))
        if texts is None:
            return cast(str | Sequence[str] | None, content_value)
        if isinstance(texts, str):
            return texts
        item = texts[idx]
        return item

    def _slice_optional(
        self,
        value: Int[torch.Tensor, "batch elements"]
        | Float[torch.Tensor, "batch elements 4"]
        | Bool[torch.Tensor, "batch elements"]
        | Sequence[str | int]
        | Sequence[Sequence[float]]
        | Sequence[bool]
        | None,
        idx: int,
    ) -> (
        Int[torch.Tensor, "elements"]
        | Float[torch.Tensor, "elements 4"]
        | Bool[torch.Tensor, "elements"]
        | Sequence[str | int]
        | Sequence[float]
        | Sequence[Sequence[float]]
        | Sequence[bool]
        | str
        | int
        | bool
        | None
    ):
        if value is None:
            return None
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            return value[idx]
        if isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
            return value[idx]
        return value


__all__ = ["PosterLlavaPipeline"]

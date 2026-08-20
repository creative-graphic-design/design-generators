"""Pipeline orchestration for PosterLlama inference recipes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Literal, cast

import torch
from jaxtyping import Bool, Float, Int
from transformers import PretrainedConfig
from transformers.image_utils import ImageInput

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import LayoutGenerationPipeline, PipelineComponentSpec

from .configuration_posterllama import PosterLlamaConfig
from .modeling_posterllama import PosterLlamaRuntime
from .processing_posterllama import PosterLlamaProcessor


def _load_processor_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> PosterLlamaProcessor:
    return PosterLlamaProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
        subfolder=subfolder,
    )


def _load_runtime_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> PosterLlamaRuntime:
    return PosterLlamaRuntime.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
        subfolder=subfolder,
    )


class PosterLlamaPipeline(LayoutGenerationPipeline):
    """Compose a PosterLlama processor and converted runtime.

    Args:
        config: Explicit pipeline configuration.
        processor: Explicit processor.
        runtime: Optional converted runtime. Parser-only saved artifacts may omit it.

    Examples:
        >>> cfg = PosterLlamaConfig(canvas_size=(100, 100))
        >>> text = '<svg width="100" height="100"><rect data-category="text" x="0" y="0" width="10" height="10"/></svg>'
        >>> pipe = PosterLlamaPipeline(
        ...     config=cfg,
        ...     processor=PosterLlamaProcessor.from_config(cfg),
        ...     runtime=PosterLlamaRuntime(text),
        ... )
        >>> pipe(images=None).bbox.shape
        torch.Size([1, 1, 4])
    """

    config_class: ClassVar[type[PretrainedConfig]] = PosterLlamaConfig
    component_specs: ClassVar[dict[str, PipelineComponentSpec]] = {
        "processor": PipelineComponentSpec(
            attribute_name="processor",
            loader=_load_processor_component,
            config_subfolder_attribute="processor_subfolder",
            marker_file="processor_config.json",
            save_with_is_main_process=False,
        ),
        "runtime": PipelineComponentSpec(
            attribute_name="runtime",
            loader=_load_runtime_component,
            config_subfolder_attribute="runtime_subfolder",
            required=False,
            marker_file="runtime_config.json",
        ),
    }

    config: PosterLlamaConfig
    processor: PosterLlamaProcessor
    runtime: PosterLlamaRuntime | None

    def __init__(
        self,
        *,
        config: PosterLlamaConfig,
        processor: PosterLlamaProcessor,
        runtime: PosterLlamaRuntime | None = None,
    ) -> None:
        """Initialize pipeline components."""
        super().__init__(config)
        self.config = config
        self.processor = processor
        self.runtime = runtime

    @classmethod
    def _from_pretrained_components(  # ty: ignore[invalid-method-override]
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, PosterLlamaProcessor | PosterLlamaRuntime | None],
    ) -> "PosterLlamaPipeline":
        """Build a pipeline from loaded components."""
        return cls(
            config=cast(PosterLlamaConfig, config),
            processor=cast(PosterLlamaProcessor, components["processor"]),
            runtime=cast(PosterLlamaRuntime | None, components["runtime"]),
        )

    @torch.no_grad()
    def __call__(  # ty: ignore[invalid-method-override]
        self,
        *,
        images: ImageInput | Sequence[ImageInput] | None = None,
        prompt: str | Sequence[str] | None = None,
        content: Mapping[str, str | int | float | bool | None]
        | Sequence[Mapping[str, str | int | float | bool | None]]
        | None = None,
        texts: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.content_image,
        labels: Int[torch.Tensor, "..."]
        | Sequence[Sequence[int | str]]
        | Sequence[int | str]
        | None = None,
        bbox: Float[torch.Tensor, "..."]
        | Sequence[Sequence[Sequence[float | int]]]
        | Sequence[Sequence[float | int]]
        | Sequence[float | int]
        | None = None,
        mask: Bool[torch.Tensor, "..."]
        | Sequence[Sequence[bool]]
        | Sequence[bool]
        | None = None,
        num_elements: int | Sequence[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_beams: int | None = None,
    ) -> (
        LayoutGenerationOutput
        | dict[
            str,
            Float[torch.Tensor, "..."]
            | Int[torch.Tensor, "..."]
            | Bool[torch.Tensor, "..."]
            | dict[int, str]
            | Mapping[str, str | bytes | int | float | bool | None]
            | list[str]
            | list[tuple[float, float, float, float]]
            | tuple[int, int]
            | str
            | bytes
            | int
            | float
            | bool
            | None,
        ]
    ):
        """Generate a poster layout.

        Args:
            images: Poster image inputs.
            prompt: Optional prompt prefix override.
            content: Optional content metadata.
            texts: Optional poster text strings.
            batch_size: Batch size when images are omitted.
            seed: Convenience seed used only when ``generator`` is absent.
            generator: Explicit PyTorch generator; takes precedence over ``seed``.
            condition_type: Canonical condition or PosterLlama release alias.
            labels: Optional label constraints.
            bbox: Optional box constraints.
            mask: Optional valid-element mask.
            num_elements: Optional requested element count.
            box_format: Input box format.
            normalized: Whether boxes are normalized.
            canvas_size: Canvas size as ``(width, height)``.
            num_inference_steps: Reserved shared argument.
            output_type: ``dataclass`` or ``dict``.
            return_intermediates: Whether to include prompt and parse diagnostics.
            max_new_tokens: Generation token budget.
            do_sample: Sampling flag.
            temperature: Sampling temperature.
            top_p: Nucleus sampling value.
            top_k: Top-k sampling value.
            num_beams: Beam count.

        Returns:
            LayoutGenerationOutput or dictionary.

        Raises:
            RuntimeError: If converted runtime assets are absent.
        """
        _ = num_inference_steps
        batch = self.processor(
            images=images,
            prompt=prompt,
            content=content,
            texts=texts,
            batch_size=batch_size,
            condition_type=condition_type,
            labels=labels,
            bbox=bbox,
            mask=mask,
            num_elements=num_elements,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )
        if self.runtime is None:
            raise RuntimeError(
                "PosterLlama runtime assets are missing. A processor-only artifact "
                "can build prompts and parse outputs, but generation requires a "
                "converted local runtime."
            )

        active_generator = self.prepare_generator(generator=generator, seed=seed)
        generation_args = {
            "max_new_tokens": max_new_tokens or self.config.default_max_new_tokens,
            "do_sample": self.config.default_do_sample
            if do_sample is None
            else do_sample,
            "temperature": self.config.default_temperature
            if temperature is None
            else temperature,
            "top_p": self.config.default_top_p if top_p is None else top_p,
            "top_k": self.config.default_top_k if top_k is None else top_k,
            "num_beams": self.config.default_num_beams
            if num_beams is None
            else num_beams,
        }
        generated = self.runtime.generate_texts(
            cast(list[str], batch["prompts"]),
            pixel_values=cast(torch.Tensor, batch["pixel_values"]),
            generator=active_generator,
            **generation_args,
        )
        output = self.processor.parse_output(
            generated[0],
            canvas_size=canvas_size,
            output_type="dataclass",
            return_intermediates=return_intermediates,
        )
        result = cast(LayoutGenerationOutput, output)
        if return_intermediates:
            existing = cast(
                dict[
                    str,
                    str
                    | bytes
                    | int
                    | float
                    | bool
                    | None
                    | Mapping[str, str | int | float | bool | None],
                ],
                result.intermediates or {},
            )
            existing["prompt_bytes"] = cast(list[str], batch["prompts"])[0].encode()
            existing["raw_generated_text"] = generated[0]
            existing["condition_type"] = str(batch["condition_type"])
            existing["generation_args"] = generation_args
            result.intermediates = existing
        if output_type == "dict":
            return dict(result)
        if output_type != "dataclass":
            raise ValueError(f"Unsupported output_type: {output_type}")

        return result

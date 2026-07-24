"""Pipeline interface for LayoutVAE layout generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias, TypedDict, assert_never, cast

import torch
from jaxtyping import Bool, Float, Int
from transformers import PretrainedConfig
from transformers.tokenization_utils_base import BatchEncoding

from laygen.common.bbox import BoxFormat, normalize_box_format
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import (
    LayoutGenerationPipeline,
    PipelineComponentSpec,
    model_processor_component_specs,
)

from .configuration_layoutvae import LayoutVAEConfig
from .modeling_layoutvae import LayoutVAEModel, LayoutVAEModelOutput, OutputType
from .modeling_layoutvae import normalize_output_type
from .processing_layoutvae import LayoutVAEProcessor

Id2Label = dict[int, str] | dict[str, str]
InputLabelsTensor: TypeAlias = Int[torch.Tensor, "..."]
LabelSetTensor: TypeAlias = Float[torch.Tensor, "batch internal_labels"]
PublicBboxTensor: TypeAlias = Float[torch.Tensor, "batch elements 4"]
PublicLabelTensor: TypeAlias = Int[torch.Tensor, "batch elements"]
PublicMaskTensor: TypeAlias = Bool[torch.Tensor, "batch elements"]
NumElementsTensor: TypeAlias = Int[torch.Tensor, "batch"]
CountLatentsTensor: TypeAlias = Float[torch.Tensor, "batch internal_labels latent"]
BboxLatentsTensor: TypeAlias = Float[torch.Tensor, "batch elements latent"]
BboxNoiseTensor: TypeAlias = Float[torch.Tensor, "batch elements 4"]
ClassCountsTensor: TypeAlias = Float[torch.Tensor, "batch internal_labels"]


@dataclass(frozen=True)
class GenerationOptions:
    """Common generation options accepted by the pipeline."""

    bbox: PublicBboxTensor | None = None
    labels: PublicLabelTensor | None = None
    mask: PublicMaskTensor | None = None
    num_elements: int | list[int] | NumElementsTensor | None = None
    box_format: BoxFormat | str = BoxFormat.xywh
    normalized: bool = True
    canvas_size: tuple[int, int] | None = None
    seed: int | None = None
    generator: torch.Generator | None = None
    num_inference_steps: int | None = None
    output_type: OutputType | str = OutputType.dataclass
    return_intermediates: bool = False


class GenerationOptionsKwargs(TypedDict, total=False):
    """Keyword dictionary used to avoid repeated call-site scaffolding."""

    bbox: PublicBboxTensor | None
    mask: PublicMaskTensor | None
    num_elements: int | list[int] | NumElementsTensor | None
    box_format: BoxFormat | str
    normalized: bool
    canvas_size: tuple[int, int] | None
    seed: int | None
    generator: torch.Generator | None
    num_inference_steps: int | None
    output_type: OutputType | str
    return_intermediates: bool


def _make_generation_options(data: GenerationOptionsKwargs) -> GenerationOptions:
    return GenerationOptions(**data)


def _resolve_device(device: int | torch.device) -> torch.device:
    if isinstance(device, int):
        return torch.device("cpu") if device < 0 else torch.device(f"cuda:{device}")
    return device


def _pop_generation_options(model_inputs: dict[str, object]) -> GenerationOptions:
    return GenerationOptions(
        bbox=cast(PublicBboxTensor | None, model_inputs.pop("bbox", None)),
        labels=cast(PublicLabelTensor | None, model_inputs.pop("labels", None)),
        mask=cast(PublicMaskTensor | None, model_inputs.pop("mask", None)),
        num_elements=cast(
            int | list[int] | NumElementsTensor | None,
            model_inputs.pop("num_elements", None),
        ),
        box_format=cast(
            BoxFormat | str, model_inputs.pop("box_format", BoxFormat.xywh)
        ),
        normalized=cast(bool, model_inputs.pop("normalized", True)),
        canvas_size=cast(tuple[int, int] | None, model_inputs.pop("canvas_size", None)),
        seed=cast(int | None, model_inputs.pop("seed", None)),
        generator=cast(torch.Generator | None, model_inputs.pop("generator", None)),
        num_inference_steps=cast(
            int | None,
            model_inputs.pop("num_inference_steps", None),
        ),
        output_type=cast(
            OutputType | str,
            model_inputs.pop("output_type", OutputType.dataclass),
        ),
        return_intermediates=cast(
            bool,
            model_inputs.pop("return_intermediates", False),
        ),
    )


def _load_model_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is not None:
        return LayoutVAEModel.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return LayoutVAEModel.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
    )


def _load_processor_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is not None:
        return LayoutVAEProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return LayoutVAEProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
    )


class LayoutVAEPipeline(LayoutGenerationPipeline):
    """Transformers pipeline for LayoutVAE label-conditioned generation.

    Args:
        model: LayoutVAE model instance.
        processor: Optional processor for label-set encoding.
        config: Optional root pipeline config. Defaults to `model.config`.
        device: Optional torch device.
        binary_output: Reserved compatibility flag.

    Examples:
        >>> model = LayoutVAEModel(LayoutVAEConfig())
        >>> pipe = LayoutVAEPipeline(model=model)
        >>> pipe.config.model_type
        'layoutvae'
    """

    config_class: ClassVar[type[PretrainedConfig]] = LayoutVAEConfig
    component_specs: ClassVar[dict[str, PipelineComponentSpec]] = (
        model_processor_component_specs(
            model_loader=_load_model_component,
            processor_loader=_load_processor_component,
        )
    )

    config: LayoutVAEConfig
    model: LayoutVAEModel
    processor: LayoutVAEProcessor

    def __init__(
        self,
        model: LayoutVAEModel,
        processor: LayoutVAEProcessor | None = None,
        config: LayoutVAEConfig | None = None,
        device: int | torch.device | None = None,
        binary_output: bool = False,
    ) -> None:
        """Initialize a LayoutVAE pipeline."""
        _ = binary_output
        super().__init__(config or model.config)
        self.config = config or model.config
        self.model = model
        self.processor = processor or LayoutVAEProcessor(
            dataset_name=model.config.dataset_name,
            id2label=model.config.id2label,
        )
        if device is not None:
            self.to(_resolve_device(device))

    @classmethod
    def _from_pretrained_components(
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, object | None],
    ) -> "LayoutVAEPipeline":
        """Build a pipeline from loaded root components."""
        return cls(
            config=cast(LayoutVAEConfig, config),
            model=cast(LayoutVAEModel, components["model"]),
            processor=cast(LayoutVAEProcessor, components["processor"]),
        )

    def _sanitize_parameters(
        self, **kwargs: object
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        return {}, kwargs, {}

    def preprocess(
        self, input_: object = None, **preprocess_parameters: object
    ) -> BatchEncoding:
        """Encode labels into model inputs."""
        labels = preprocess_parameters.pop("labels", input_)
        if labels is None:
            raise ValueError("labels are required for LayoutVAEPipeline")
        encoded = self.processor(
            cast(list[list[str | int]] | list[str | int] | InputLabelsTensor, labels)
        )
        encoded.update(preprocess_parameters)
        return encoded

    def _forward(
        self, model_inputs: dict[str, object], **forward_params: object
    ) -> LayoutGenerationOutput | dict[str, object]:
        del forward_params
        label_set = torch.as_tensor(model_inputs.pop("label_set"), dtype=torch.float32)
        condition_type = cast(
            ConditionType | str, model_inputs.pop("condition_type", ConditionType.label)
        )
        options = _pop_generation_options(model_inputs)
        count_latents = cast(
            CountLatentsTensor | None, model_inputs.pop("count_latents", None)
        )
        bbox_latents = cast(
            BboxLatentsTensor | None, model_inputs.pop("bbox_latents", None)
        )
        bbox_noise = cast(BboxNoiseTensor | None, model_inputs.pop("bbox_noise", None))
        class_counts = cast(
            ClassCountsTensor | None, model_inputs.pop("class_counts", None)
        )
        if model_inputs:
            unknown = ", ".join(sorted(model_inputs))
            raise ValueError(f"Unsupported generation kwargs: {unknown}")
        return self._generate(
            label_set=label_set,
            condition_type=condition_type,
            options=options,
            count_latents=count_latents,
            bbox_latents=bbox_latents,
            bbox_noise=bbox_noise,
            class_counts=class_counts,
        )

    def postprocess(
        self,
        model_outputs: LayoutGenerationOutput | dict[str, object],
        **kwargs: object,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Return generated layouts unchanged."""
        del kwargs
        return model_outputs

    @torch.no_grad()
    def __call__(
        self,
        labels: list[list[str | int]]
        | list[str | int]
        | InputLabelsTensor
        | None = None,
        *,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.label,
        bbox: PublicBboxTensor | None = None,
        mask: PublicMaskTensor | None = None,
        num_elements: int | list[int] | NumElementsTensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
        count_latents: CountLatentsTensor | None = None,
        bbox_latents: BboxLatentsTensor | None = None,
        bbox_noise: BboxNoiseTensor | None = None,
        class_counts: ClassCountsTensor | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:  # ty: ignore[invalid-method-override]
        """Generate PubLayNet layouts from label conditions.

        Args:
            labels: Public PubLayNet label strings or IDs.
            batch_size: Used when `labels` is omitted.
            seed: Optional random seed used when `generator` is absent.
            generator: Optional PyTorch random generator. Takes precedence.
            condition_type: Condition type or alias. Only `label` is supported.
            bbox: Reserved compatibility argument.
            mask: Reserved compatibility argument.
            num_elements: Reserved compatibility argument.
            box_format: Reserved compatibility argument.
            normalized: Reserved compatibility argument.
            canvas_size: Reserved compatibility argument.
            num_inference_steps: Reserved compatibility argument.
            output_type: Return format.
            return_intermediates: Whether to include raw generation tensors.
            count_latents: Optional fixed count latents.
            bbox_latents: Optional fixed box latents.
            bbox_noise: Optional fixed box output noise.
            class_counts: Optional fixed class counts.

        Returns:
            Layout generation output.

        Raises:
            ValueError: If labels are missing or condition options are unsupported.

        Examples:
            >>> pipe = LayoutVAEPipeline(LayoutVAEModel(LayoutVAEConfig()))
            >>> out = pipe(labels=["text"], class_counts=torch.tensor([[8, 1, 0, 0, 0, 0.]]))
            >>> tuple(out.bbox.shape)
            (1, 9, 4)
        """
        if labels is None:
            if batch_size < 1:
                raise ValueError("batch_size must be positive")
            labels = [["text"] for _ in range(batch_size)]
        encoded = self.processor(labels)
        option_kwargs: GenerationOptionsKwargs = {}
        option_kwargs["bbox"] = bbox
        option_kwargs["mask"] = mask
        option_kwargs["num_elements"] = num_elements
        option_kwargs["box_format"] = box_format
        option_kwargs["normalized"] = normalized
        option_kwargs["canvas_size"] = canvas_size
        option_kwargs["seed"] = seed
        option_kwargs["generator"] = generator
        option_kwargs["num_inference_steps"] = num_inference_steps
        option_kwargs["output_type"] = output_type
        option_kwargs["return_intermediates"] = return_intermediates
        options = _make_generation_options(option_kwargs)
        return self._generate(
            label_set=cast(LabelSetTensor, encoded["label_set"]),
            condition_type=condition_type,
            options=options,
            count_latents=count_latents,
            bbox_latents=bbox_latents,
            bbox_noise=bbox_noise,
            class_counts=class_counts,
        )

    def _generate(
        self,
        *,
        label_set: LabelSetTensor,
        condition_type: ConditionType | str,
        options: GenerationOptions,
        count_latents: CountLatentsTensor | None,
        bbox_latents: BboxLatentsTensor | None,
        bbox_noise: BboxNoiseTensor | None,
        class_counts: ClassCountsTensor | None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        _ = (
            options.bbox,
            options.labels,
            options.mask,
            options.num_elements,
            options.normalized,
            options.canvas_size,
            options.num_inference_steps,
        )
        normalize_box_format(options.box_format)
        canonical = normalize_condition_type(condition_type)
        if canonical is not ConditionType.label:
            raise ValueError(f"Unsupported condition_type for layoutvae: {canonical}")
        device = next(self.model.parameters()).device
        prepared_generator = self.prepare_generator(
            generator=options.generator,
            seed=options.seed,
            device=device,
        )
        out = self.model(
            label_set.to(device=device),
            count_latents=count_latents,
            bbox_latents=bbox_latents,
            bbox_noise=bbox_noise,
            class_counts=class_counts,
            generator=prepared_generator,
            return_dict=True,
        )
        assert isinstance(out, LayoutVAEModelOutput)
        intermediates = None
        if options.return_intermediates:
            intermediates = {
                "condition_type": canonical,
                "raw_ltwh": out.raw_ltwh.detach().cpu(),
                "internal_labels": out.internal_labels.detach().cpu()
                if out.internal_labels is not None
                else None,
                "class_counts": out.class_counts.detach().cpu(),
            }
        layout = LayoutGenerationOutput(
            bbox=out.bbox.detach().cpu(),
            labels=out.labels.detach().cpu(),
            mask=out.mask.detach().cpu(),
            id2label={
                int(k): v for k, v in cast(Id2Label, self.config.id2label).items()
            },
            intermediates=intermediates,
        )
        resolved_output_type = normalize_output_type(options.output_type)
        if resolved_output_type is OutputType.dict:
            return dict(layout)
        if resolved_output_type is OutputType.dataclass:
            return layout
        assert_never(resolved_output_type)

"""Pipeline interface for LayoutGAN++ layout generation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, cast

import torch
from jaxtyping import Bool, Float, Int
from transformers import PretrainedConfig
from transformers.tokenization_utils_base import BatchEncoding

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import (
    LayoutGenerationPipeline,
    PipelineComponentSpec,
    model_processor_component_specs,
)

from .configuration_layoutganpp import LayoutGANPPConfig
from .modeling_layoutganpp import LayoutGANPPModel, OutputType
from .processing_layoutganpp import LayoutGANPPProcessor


def _load_model_component(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> object:
    if subfolder is not None:
        return LayoutGANPPModel.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return LayoutGANPPModel.from_pretrained(
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
        return LayoutGANPPProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
    return LayoutGANPPProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
    )


class LayoutGANPPPipeline(LayoutGenerationPipeline):
    """Transformers pipeline for LayoutGAN++ label-conditioned generation.

    Args:
        model: LayoutGAN++ model instance.
        processor: Optional processor for label encoding and decoding.
        config: Optional root pipeline config. Defaults to `model.config`.
        device: Optional torch device passed to the base pipeline.
        binary_output: Whether the base pipeline should produce binary output.

    Examples:
        >>> model = LayoutGANPPModel(LayoutGANPPConfig(num_labels=2))
        >>> pipe = LayoutGANPPPipeline(model=model)
        >>> pipe.model.config.model_type
        'layoutganpp'
    """

    config_class: ClassVar[type[PretrainedConfig]] = LayoutGANPPConfig
    component_specs: ClassVar[dict[str, PipelineComponentSpec]] = (
        model_processor_component_specs(
            model_loader=_load_model_component,
            processor_loader=_load_processor_component,
        )
    )

    config: LayoutGANPPConfig
    model: LayoutGANPPModel
    processor: LayoutGANPPProcessor

    def __init__(
        self,
        model: LayoutGANPPModel,
        processor: LayoutGANPPProcessor | None = None,
        config: LayoutGANPPConfig | None = None,
        device: int | torch.device | None = None,
        binary_output: bool = False,
    ) -> None:
        """Initialize a LayoutGAN++ pipeline.

        Args:
            model: LayoutGAN++ model instance.
            processor: Optional processor for label encoding and decoding.
            config: Optional root pipeline config.
            device: Optional torch device passed to the base pipeline.
            binary_output: Whether the base pipeline should produce binary output.

        Examples:
            >>> pipe = LayoutGANPPPipeline(LayoutGANPPModel(LayoutGANPPConfig()))
            >>> isinstance(pipe.processor, LayoutGANPPProcessor)
            True
        """
        _ = binary_output
        super().__init__(config or model.config)
        self.config = config or model.config
        self.model = model
        self.processor = processor or LayoutGANPPProcessor(
            dataset_name=model.config.dataset_name,
            id2label=model.config.id2label,
        )
        if device is not None:
            resolved_device = (
                torch.device("cpu")
                if isinstance(device, int) and device < 0
                else torch.device(f"cuda:{device}")
                if isinstance(device, int)
                else device
            )
            self.to(resolved_device)

    @classmethod
    def _from_pretrained_components(
        cls,
        *,
        config: PretrainedConfig,
        components: Mapping[str, object | None],
    ) -> "LayoutGANPPPipeline":
        """Build a pipeline from loaded root components."""
        return cls(
            config=cast(LayoutGANPPConfig, config),
            model=cast(LayoutGANPPModel, components["model"]),
            processor=cast(LayoutGANPPProcessor, components["processor"]),
        )

    def _sanitize_parameters(
        self, **kwargs: object
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        return {}, kwargs, {}

    def preprocess(
        self, input_: object = None, **preprocess_parameters: object
    ) -> BatchEncoding:
        """Encode pipeline inputs into model inputs.

        Args:
            input_: Labels supplied as the positional pipeline input.
            **preprocess_parameters: Keyword labels and generation arguments.

        Returns:
            Batch encoding containing label IDs, attention mask, and generation kwargs.

        Raises:
            ValueError: If labels are not supplied or cannot be encoded.

        Examples:
            >>> pipe = LayoutGANPPPipeline(LayoutGANPPModel(LayoutGANPPConfig()))
            >>> "labels" in pipe.preprocess(["Toolbar"])
            True
        """
        labels = preprocess_parameters.pop("labels", input_)
        if labels is None:
            raise ValueError("labels are required for LayoutGANPPPipeline")
        encoded = self._layoutganpp_processor()(
            cast(list[list[str | int]] | list[str | int] | torch.Tensor, labels)
        )
        encoded.update(preprocess_parameters)
        return encoded

    def _forward(
        self, model_inputs: dict[str, object], **forward_params: object
    ) -> LayoutGenerationOutput | dict[str, object]:
        del forward_params
        labels = torch.as_tensor(model_inputs.pop("labels"), dtype=torch.long)
        attention_mask = torch.as_tensor(
            model_inputs.pop("attention_mask"), dtype=torch.bool
        )
        condition_type = cast(
            ConditionType | str, model_inputs.pop("condition_type", ConditionType.label)
        )
        bbox = cast(torch.Tensor | None, model_inputs.pop("bbox", None))
        mask = cast(torch.Tensor | None, model_inputs.pop("mask", None))
        num_elements = cast(
            int | list[int] | torch.Tensor | None,
            model_inputs.pop("num_elements", None),
        )
        box_format = cast(
            BoxFormat | str, model_inputs.pop("box_format", BoxFormat.xywh)
        )
        normalized = cast(bool, model_inputs.pop("normalized", True))
        canvas_size = cast(
            tuple[int, int] | None, model_inputs.pop("canvas_size", None)
        )
        seed = cast(int | None, model_inputs.pop("seed", None))
        generator = cast(torch.Generator | None, model_inputs.pop("generator", None))
        num_inference_steps = cast(
            int | None, model_inputs.pop("num_inference_steps", None)
        )
        output_type = cast(
            OutputType | str, model_inputs.pop("output_type", OutputType.dataclass)
        )
        return_intermediates = cast(
            bool, model_inputs.pop("return_intermediates", False)
        )
        latents = cast(torch.Tensor | None, model_inputs.pop("latents", None))
        if model_inputs:
            unknown = ", ".join(sorted(model_inputs))
            raise ValueError(f"Unsupported generation kwargs: {unknown}")
        return self._layoutganpp_model().generate(
            condition_type=condition_type,
            bbox=bbox,
            labels=labels,
            mask=mask,
            attention_mask=attention_mask,
            num_elements=num_elements,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
            seed=seed,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            return_intermediates=return_intermediates,
            latents=latents,
        )

    def postprocess(
        self,
        model_outputs: LayoutGenerationOutput | dict[str, object],
        **kwargs: object,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Return generated layouts from the pipeline output.

        Args:
            model_outputs: Output produced by `LayoutGANPPModel.generate`.
            **kwargs: Reserved post-processing keyword arguments.

        Returns:
            The generated layout output unchanged.

        Examples:
            >>> output = LayoutGenerationOutput(bbox=torch.zeros(1, 1, 4))
            >>> pipe = LayoutGANPPPipeline(LayoutGANPPModel(LayoutGANPPConfig()))
            >>> pipe.postprocess(output) is output
            True
        """
        del kwargs
        return model_outputs

    @torch.no_grad()
    def __call__(
        self,
        labels: list[list[str | int]]
        | list[str | int]
        | Int[torch.Tensor, "batch elements"]
        | None = None,
        *,
        batch_size: int = 1,
        condition_type: ConditionType | str = ConditionType.label,
        bbox: Float[torch.Tensor, "batch elements 4"] | None = None,
        mask: Bool[torch.Tensor, "batch elements"] | None = None,
        attention_mask: Bool[torch.Tensor, "batch elements"] | None = None,
        num_elements: int | list[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
        latents: Float[torch.Tensor, "batch elements latent"] | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:  # ty: ignore[invalid-method-override]
        """Generate LayoutGAN++ boxes from labels.

        Args:
            labels: Label strings or label IDs to condition on.
            batch_size: Reserved compatibility argument.
            condition_type: Condition type or alias.
            bbox: Reserved compatibility argument.
            mask: Optional valid-element mask.
            attention_mask: Optional valid-element mask.
            num_elements: Reserved compatibility argument.
            box_format: Reserved compatibility argument.
            normalized: Reserved compatibility argument.
            canvas_size: Reserved compatibility argument.
            seed: Optional random seed for latent sampling.
            generator: Optional PyTorch random generator.
            num_inference_steps: Reserved compatibility argument.
            output_type: Return format, either `dataclass` or `dict`.
            return_intermediates: Whether to include generation intermediates.
            latents: Optional fixed latent vectors.

        Returns:
            A layout generation dataclass or dictionary.

        Raises:
            ValueError: If labels are missing or generation options are invalid.

        Examples:
            >>> pipe = LayoutGANPPPipeline(LayoutGANPPModel(LayoutGANPPConfig()))
            >>> out = pipe(labels=["Toolbar"], seed=0)
            >>> tuple(out.bbox.shape)
            (1, 1, 4)
        """
        del batch_size
        if labels is None:
            raise ValueError("labels are required for layoutganpp v1")
        if isinstance(labels, torch.Tensor):
            encoded_labels = labels
            resolved_mask = attention_mask if attention_mask is not None else mask
        else:
            encoded = self._layoutganpp_processor()(labels)
            encoded_labels = encoded["labels"]
            if attention_mask is not None:
                resolved_mask = attention_mask
            else:
                resolved_mask = encoded["attention_mask"] if mask is None else mask
        return self._layoutganpp_model().generate(
            condition_type=condition_type,
            bbox=bbox,
            labels=cast(torch.Tensor, encoded_labels),
            mask=cast(torch.Tensor | None, resolved_mask),
            num_elements=num_elements,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
            seed=seed,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            return_intermediates=return_intermediates,
            latents=latents,
        )

    def _layoutganpp_model(self) -> LayoutGANPPModel:
        return self.model

    def _layoutganpp_processor(self) -> LayoutGANPPProcessor:
        return self.processor

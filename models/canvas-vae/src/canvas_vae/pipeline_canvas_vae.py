"""Unconditional layout generation pipeline for CanvasVAE."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum, auto
from pathlib import Path
from typing import ClassVar, Literal, Self, cast

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from laygen.common.bbox import BoxFormat, clamp_boxes, ltwh_to_xywh
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.common.randomness import randn
from laygen.modeling_outputs import LayoutGenerationOutput
from laygen.pipelines import LayoutGenerationPipeline, PipelineComponentSpec
from laygen.pipelines.base import PipelineComponent

from .configuration_canvas_vae import CanvasVAEConfig, CanvasVAEField
from .modeling_canvas_vae import CanvasVAEModel

LayoutArgument = torch.Tensor | np.ndarray | list | None


class OutputType(StrEnum):
    """Return container of :meth:`CanvasVAEPipeline.__call__`."""

    dataclass = auto()
    dict = auto()


def _load_model(
    pretrained_model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> CanvasVAEModel:
    return CanvasVAEModel.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
        subfolder=subfolder or "",
    )


def bins_to_ltwh(
    bins: Int[torch.Tensor, "... 4"], num_bins: int
) -> Float[torch.Tensor, "... 4"]:
    """Map left, top, width, and height bin ids to normalized values.

    Args:
        bins: Bin ids in ``[0, num_bins)``.
        num_bins: Number of bins.

    Returns:
        Values ``bin / (num_bins - 1)`` in ``[0, 1]``.

    Examples:
        >>> bins_to_ltwh(torch.tensor([0, 3, 1, 2]), 4).tolist()
        [0.0, 1.0, 0.3333333432674408, 0.6666666865348816]
    """
    return bins.float() / (num_bins - 1)


class CanvasVAEPipeline(LayoutGenerationPipeline):
    """Sample RICO layouts from the CanvasVAE prior.

    Each call draws latent codes from the standard normal prior, decodes the
    element count and every element field with ``argmax``, and returns
    normalized center ``xywh`` boxes clamped to ``[0, 1]`` with ``component``
    labels. ``intermediates`` holds the ``clickable``, ``icon``, and
    ``text_button`` ids and the latent codes.

    Args:
        model: CanvasVAE model.

    Examples:
        >>> from canvas_vae import CanvasVAEConfig
        >>> config = CanvasVAEConfig(
        ...     vocabularies={
        ...         "component": ["[UNK]", "", "Text"],
        ...         "icon": ["[UNK]", ""],
        ...         "text_button": ["[UNK]", ""],
        ...     },
        ...     max_length=4,
        ...     num_bins=8,
        ...     latent_dim=16,
        ...     num_heads=2,
        ... )
        >>> pipe = CanvasVAEPipeline(model=CanvasVAEModel(config))
        >>> out = pipe(batch_size=2, seed=0)
        >>> out.bbox.shape[0], out.bbox.shape[-1]
        (2, 4)
    """

    config_class: ClassVar[type[CanvasVAEConfig]] = CanvasVAEConfig
    component_specs: ClassVar[dict[str, PipelineComponentSpec]] = {
        "model": PipelineComponentSpec(attribute_name="model", loader=_load_model)
    }

    def __init__(self, model: CanvasVAEModel) -> None:
        """Wrap a model; the pipeline config is the model config."""
        super().__init__(model.config)
        self.model = model

    @classmethod
    def _from_pretrained_components(
        cls,
        *,
        config: CanvasVAEConfig,
        components: Mapping[str, PipelineComponent | None],
    ) -> Self:
        """Build the pipeline from a loaded model."""
        del config
        return cls(model=cast(CanvasVAEModel, components["model"]))

    @torch.no_grad()
    def __call__(
        self,
        *,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.unconditional,
        labels: LayoutArgument = None,
        bbox: LayoutArgument = None,
        mask: LayoutArgument = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: Literal["xywh", "ltwh", "ltrb"] = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        """Generate layouts from prior samples.

        Args:
            batch_size: Number of layouts.
            seed: Seed for a local generator when ``generator`` is absent.
            generator: Generator for the prior samples; takes precedence over
                ``seed``.
            condition_type: Only ``unconditional`` is supported.
            labels: Unsupported; must be ``None``.
            bbox: Unsupported; must be ``None``.
            mask: Unsupported; must be ``None``.
            num_elements: Unsupported; the model predicts the element count.
            box_format: Only ``xywh`` output is supported.
            normalized: Only normalized output is supported.
            canvas_size: Unsupported; must be ``None``.
            num_inference_steps: Unsupported; decoding is a single pass.
            output_type: ``dataclass`` or ``dict``.
            return_intermediates: Whether to fill ``intermediates``.

        Returns:
            Generated layouts in the shared layout schema.

        Raises:
            NotImplementedError: If a condition or layout input other than
                unconditional generation is requested.
        """
        condition = normalize_condition_type(condition_type)
        if condition is not ConditionType.unconditional:
            raise NotImplementedError(
                f"CanvasVAE supports only unconditional generation, got {condition}"
            )

        given = {
            "labels": labels,
            "bbox": bbox,
            "mask": mask,
            "num_elements": num_elements,
            "canvas_size": canvas_size,
            "num_inference_steps": num_inference_steps,
        }
        unsupported = [name for name, value in given.items() if value is not None]
        if unsupported or BoxFormat(box_format) is not BoxFormat.xywh or not normalized:
            raise NotImplementedError(
                "CanvasVAE unconditional generation returns normalized xywh boxes "
                f"and takes no layout inputs; got {unsupported or box_format}"
            )

        generator = self.prepare_generator(generator=generator, seed=seed)
        device = self.model.device
        latents = randn(
            (batch_size, self.model.config.latent_dim),
            generator=generator,
            device=device,
            dtype=self.model.dtype,
        )
        was_training = self.model.training
        self.model.eval()
        try:
            output = self.model(latents=latents)
        finally:
            self.model.train(was_training)

        ids = {
            key: logits.argmax(dim=-1) for key, logits in output.element_logits.items()
        }
        geometry = torch.stack(
            [
                ids[CanvasVAEField.left],
                ids[CanvasVAEField.top],
                ids[CanvasVAEField.width],
                ids[CanvasVAEField.height],
            ],
            dim=-1,
        )
        boxes = clamp_boxes(
            ltwh_to_xywh(bins_to_ltwh(geometry, self.model.config.num_bins))
        )
        valid = cast(Bool[torch.Tensor, "batch elements"], output.mask)
        result = LayoutGenerationOutput(
            bbox=boxes * valid.unsqueeze(-1),
            labels=ids[CanvasVAEField.component] * valid,
            mask=valid,
            id2label=dict(self.model.config.id2label),
            intermediates=(
                {
                    "clickable": ids[CanvasVAEField.clickable],
                    "icon": ids[CanvasVAEField.icon],
                    "text_button": ids[CanvasVAEField.text_button],
                    "latents": latents,
                }
                if return_intermediates
                else None
            ),
        )
        if OutputType(output_type) is OutputType.dict:
            return dict(result)

        return result


__all__ = ["CanvasVAEPipeline", "OutputType", "bins_to_ltwh"]

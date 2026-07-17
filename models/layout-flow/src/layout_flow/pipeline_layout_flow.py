"""Diffusers pipeline for LayoutFlow inference."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Self, TypeAlias, assert_never, cast

import numpy as np
import torch
from diffusers import DiffusionPipeline

from laygen.common.bbox import BoxFormat
from laygen.common.outputs_diffusers import LayoutGenerationOutput

from .configuration_layout_flow import LayoutFlowConfig
from .modeling_layout_flow import LayoutFlowTransformerModel
from .processing_layout_flow import (
    ConditionType,
    LayoutFlowProcessor,
    normalize_condition_type,
)
from .sampling import sample_initial_state
from .scheduling_layout_flow import LayoutFlowEulerScheduler

TensorInput: TypeAlias = torch.Tensor | np.ndarray | Sequence[object] | None


class OutputType(StrEnum):
    """Pipeline output containers supported by LayoutFlow."""

    dataclass = "dataclass"
    dict = "dict"


class LayoutFlowPipeline(DiffusionPipeline):
    """Generate layouts with a converted LayoutFlow checkpoint."""

    model_cpu_offload_seq: str = "model"

    def __init__(
        self,
        model: LayoutFlowTransformerModel,
        scheduler: LayoutFlowEulerScheduler,
        config: LayoutFlowConfig | None = None,
        processor: LayoutFlowProcessor | None = None,
    ) -> None:
        """Create a LayoutFlow pipeline.

        Args:
            model: Converted LayoutFlow transformer model.
            scheduler: Increasing-time Euler scheduler.
            config: Optional pipeline configuration.
            processor: Optional input/output processor.
        """
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        dataset_name = "rico25" if model.config.num_labels == 26 else "publaynet"
        self.layout_flow_config = config or LayoutFlowConfig(dataset_name=dataset_name)
        self.processor = processor or LayoutFlowProcessor(self.layout_flow_config)
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        *,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.unconditional,
        labels: TensorInput = None,
        bbox: TensorInput = None,
        mask: TensorInput = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float = 0.0,
        output_type: OutputType | str = "dataclass",
        return_intermediates: bool = False,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        """Generate layout boxes and labels.

        Args:
            batch_size: Number of layouts to generate.
            seed: Optional seed used when ``generator`` is omitted.
            generator: Optional torch random generator.
            condition_type: Public condition name or vendor alias.
            labels: Optional condition labels.
            bbox: Optional condition boxes.
            mask: Optional valid-element mask.
            num_elements: Optional element counts for unconditional masks.
            box_format: Input and output box format.
            normalized: Whether coordinates are normalized.
            canvas_size: Pixel canvas size for denormalized coordinates.
            num_inference_steps: Number of Euler steps.
            guidance_scale: Classifier-free guidance scale.
            output_type: ``"dataclass"`` or ``"dict"``.
            return_intermediates: Whether to include intermediate samples.

        Returns:
            Layout generation output dataclass, or a dictionary when requested.

        Raises:
            ValueError: If ``condition_type``, ``box_format``, or ``output_type``
                is unsupported.

        Examples:
            >>> pipe = LayoutFlowPipeline(
            ...     model=LayoutFlowTransformerModel(
            ...         num_labels=6, latent_dim=8, d_model=16, nhead=4,
            ...         dim_feedforward=32, num_layers=1
            ...     ),
            ...     scheduler=LayoutFlowEulerScheduler(num_inference_steps=2),
            ...     config=LayoutFlowConfig(max_length=2, latent_dim=8, d_model=16),
            ... )
            >>> out = pipe(batch_size=1, num_elements=1, seed=0, num_inference_steps=2)
            >>> out.bbox.shape[-1]
            4
        """
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        canonical = normalize_condition_type(condition_type)
        output_kind = OutputType(output_type)
        processed = self.processor(
            bbox=bbox,
            labels=labels,
            mask=mask,
            num_elements=num_elements,
            batch_size=batch_size,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
            device=self.device,
        )
        batch_size = processed["bbox"].shape[0]
        cond_mask = self.processor.make_condition_mask(
            canonical, mask=processed["mask"], generator=generator
        )
        cond_state = self.processor.preprocess_state(
            self.processor.model_state(processed["bbox"], processed["labels"])
        )
        sample = sample_initial_state(
            batch_size=batch_size,
            max_length=self.layout_flow_config.max_length,
            lengths=processed["length"],
            dim=self.layout_flow_config.sample_dim,
            distribution=self.layout_flow_config.distribution,
            generator=generator,
            device=self.device,
            dtype=cond_state.dtype,
        )
        if canonical is ConditionType.refinement:
            sample = cond_state
            self.scheduler.set_timesteps(
                num_inference_steps, device=self.device, start=0.97, end=1.0
            )
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        trajectory = [] if return_intermediates else None
        for i, timestep in enumerate(self.scheduler.timesteps[:-1]):
            x_in = (1 - cond_mask) * cond_state + cond_mask * sample
            t_batch = timestep.repeat(batch_size)
            vector = self.model(
                sample=x_in, timestep=t_batch, cond_mask=cond_mask
            ).sample
            if guidance_scale:
                uncond_mask = torch.ones_like(cond_mask)
                uncond = self.model(
                    sample=x_in, timestep=t_batch, cond_mask=uncond_mask
                ).sample
                vector = (1 + guidance_scale) * vector - guidance_scale * uncond
            sample = self.scheduler.step(
                vector,
                timestep,
                sample,
                next_timestep=self.scheduler.timesteps[i + 1],
            ).prev_sample
            if trajectory is not None:
                trajectory.append(sample.detach().cpu())
        final_state = (1 - cond_mask) * cond_state + cond_mask * sample
        decoded = self.processor.postprocess(
            final_state,
            mask=processed["mask"],
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )
        output = LayoutGenerationOutput(
            bbox=decoded["bbox"].detach().cpu(),
            labels=decoded["labels"].detach().cpu(),
            mask=decoded["mask"].detach().cpu(),
            id2label=self.layout_flow_config.id2label,
            trajectory=trajectory,
            intermediates={"condition_type": str(canonical)}
            if return_intermediates
            else None,
        )
        if output_kind is OutputType.dict:
            return dict(output)
        if output_kind is OutputType.dataclass:
            return output
        assert_never(output_kind)

    generate = __call__

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save pipeline components and LayoutFlow config.

        Args:
            save_directory: Output directory.
        """
        super().save_pretrained(save_directory)
        self.layout_flow_config.save_config(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path) -> Self:
        """Load a saved LayoutFlow pipeline.

        Args:
            pretrained_model_name_or_path: Local directory or Hub id.

        Returns:
            Loaded LayoutFlow pipeline.
        """
        config_dict, _ = LayoutFlowConfig.load_config(
            pretrained_model_name_or_path,
            return_unused_kwargs=True,
        )
        config = cast(LayoutFlowConfig, LayoutFlowConfig.from_config(config_dict))
        pipe = super().from_pretrained(pretrained_model_name_or_path)
        pipe.layout_flow_config = config
        pipe.processor = LayoutFlowProcessor(config)
        return pipe

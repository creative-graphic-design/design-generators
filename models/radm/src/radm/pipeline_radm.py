"""Diffusers pipeline for RADM poster layout generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import torch
from diffusers import DiffusionPipeline
from jaxtyping import Bool, Float, Int
from transformers.image_utils import ImageInput

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType
from laygen.pipelines.pipeline_output import LayoutGenerationOutput

from .configuration_radm import RADMConfig
from .modeling_radm import RADMDenoiser
from .processing_radm import RADMProcessor
from .scheduling_radm import RADMScheduler

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
NestedFloatSequence = (
    Sequence[float] | Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]]
)
NestedBoolSequence = (
    Sequence[bool] | Sequence[Sequence[bool]] | Sequence[Sequence[Sequence[bool]]]
)


class RADMPipeline(DiffusionPipeline):
    """Generate poster layouts with a RADM proposal diffusion pipeline."""

    model_cpu_offload_seq: str = "denoiser"

    def __init__(
        self,
        denoiser: RADMDenoiser,
        scheduler: RADMScheduler,
        config: RADMConfig,
        processor: RADMProcessor | None = None,
    ) -> None:
        """Create a RADM pipeline.

        Args:
            denoiser: Proposal denoiser component.
            scheduler: Proposal diffusion scheduler.
            config: RADM configuration.
            processor: Optional input/output processor derived from ``config``.
        """
        super().__init__()
        self.register_modules(denoiser=denoiser, scheduler=scheduler)
        self.radm_config = config
        self.processor = processor or RADMProcessor(config=config)
        self.denoiser.eval()

    @torch.no_grad()
    def __call__(
        self,
        images: ImageInput | Sequence[ImageInput] | None = None,
        *,
        content: Mapping[
            str,
            ImageInput
            | Sequence[ImageInput]
            | Float[torch.Tensor, "batch text text_dim"]
            | Float[np.ndarray, "batch text text_dim"]
            | Bool[torch.Tensor, "batch text 1"]
            | Bool[np.ndarray, "batch text 1"]
            | NestedFloatSequence
            | NestedBoolSequence
            | JsonValue,
        ]
        | None = None,
        text_features: Float[torch.Tensor, "batch text text_dim"]
        | Float[np.ndarray, "batch text text_dim"]
        | NestedFloatSequence
        | None = None,
        text_mask: Bool[torch.Tensor, "batch text 1"]
        | Bool[np.ndarray, "batch text 1"]
        | NestedBoolSequence
        | None = None,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.content_image,
        labels: Int[torch.Tensor, "batch elements"]
        | Int[np.ndarray, "batch elements"]
        | Sequence[Sequence[int | str]]
        | None = None,
        bbox: Float[torch.Tensor, "batch elements 4"]
        | Float[np.ndarray, "batch elements 4"]
        | Sequence[Sequence[float]]
        | None = None,
        mask: Bool[torch.Tensor, "batch elements"]
        | Bool[np.ndarray, "batch elements"]
        | Sequence[Sequence[bool]]
        | None = None,
        num_elements: int | Sequence[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        class_threshold: float | None = None,
        nms_threshold: float | None = None,
        return_raw_predictions: bool = False,
    ) -> (
        LayoutGenerationOutput
        | dict[
            str,
            Float[torch.Tensor, "..."]
            | Int[torch.Tensor, "..."]
            | Bool[torch.Tensor, "..."]
            | Mapping[int, str]
            | Mapping[
                str,
                str
                | int
                | float
                | bool
                | None
                | Float[torch.Tensor, "..."]
                | Int[torch.Tensor, "..."]
                | Bool[torch.Tensor, "..."]
                | list[Float[torch.Tensor, "..."]],
            ]
            | list[Float[torch.Tensor, "..."]]
            | None,
        ]
    ):
        """Generate normalized poster layout boxes and labels.

        Args:
            images: Content image or image batch.
            content: Optional content carrier with image and text-feature fields.
            text_features: Optional RADM text feature tensor.
            text_mask: Optional text-feature mask.
            batch_size: Batch size for synthetic smoke inputs.
            seed: Optional seed used when ``generator`` is omitted.
            generator: Optional torch generator. Takes precedence over ``seed``.
            condition_type: Public condition type. RADM supports ``content_image``.
            labels: Accepted common layout argument; unsupported in the first PR.
            bbox: Accepted common layout argument; unsupported in the first PR.
            mask: Accepted common layout argument; unsupported in the first PR.
            num_elements: Accepted common layout argument; unsupported in the first PR.
            box_format: Accepted common layout argument.
            normalized: Accepted common layout argument.
            canvas_size: Accepted common layout argument.
            num_inference_steps: Number of reverse diffusion steps.
            output_type: ``"dataclass"`` or ``"dict"``.
            return_intermediates: Whether to return debug intermediates.
            class_threshold: Optional confidence threshold.
            nms_threshold: Optional NMS threshold.
            return_raw_predictions: Whether raw predictions should be included.

        Returns:
            Layout generation output dataclass or dictionary.

        Raises:
            NotImplementedError: If the selected condition is unsupported.
            ValueError: If public arguments are unsupported.

        Examples:
            >>> from PIL import Image
            >>> config = RADMConfig(num_proposals=2, hidden_dim=8, text_feature_dim=4, backbone_depth=18)
            >>> pipe = RADMPipeline(
            ...     denoiser=RADMDenoiser(config=config),
            ...     scheduler=RADMScheduler(num_train_timesteps=10, num_inference_steps=2),
            ...     config=config,
            ...     processor=RADMProcessor(config=config),
            ... )
            >>> out = pipe(Image.new("RGB", (16, 16)), seed=0, num_inference_steps=2)
            >>> out.bbox.shape[-1]
            4
        """
        del labels, bbox, mask, num_elements, box_format, normalized, canvas_size
        self.processor.validate_condition(condition_type)
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        encoded = self.processor(
            images,
            content=content,
            text_features=text_features,
            text_mask=text_mask,
            batch_size=batch_size,
        )
        batch_size = int(encoded["pixel_values"].shape[0])
        sample = self.scheduler.sample_initial_proposals(
            batch_size=batch_size,
            num_proposals=self.radm_config.num_proposals,
            generator=generator,
            device=self.device,
            dtype=next(self.denoiser.parameters()).dtype,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        text = encoded["text_features"].to(self.device)
        text_valid = encoded["text_mask"].to(self.device)
        trajectory = [] if return_intermediates else None
        logits = sample.new_zeros(
            batch_size, self.radm_config.num_proposals, self.radm_config.num_classes
        )
        for timestep in self.scheduler.timesteps:
            t_batch = torch.full(
                (batch_size,),
                float(timestep.item()),
                device=self.device,
                dtype=sample.dtype,
            )
            denoised = self.denoiser(
                boxes_xyxy=sample,
                timesteps=t_batch,
                text_features=text,
                text_mask=text_valid,
                images=encoded["pixel_values"].to(self.device),
            )
            logits = denoised.logits
            sample = self.scheduler.step(
                denoised.pred_original_sample,
                timestep,
                sample,
                generator=generator,
            ).prev_sample
            if trajectory is not None:
                trajectory.append(sample.detach().cpu())
        intermediates = {
            "condition_type": str(ConditionType.content_image),
            "original_sizes": encoded["original_sizes"].detach().cpu(),
            "resized_sizes": encoded["resized_sizes"].detach().cpu(),
            "text_mask_valid": encoded["text_mask"].sum(dim=(1, 2)).detach().cpu(),
        }
        if trajectory is not None:
            intermediates["trajectory"] = trajectory
        output = self.processor.decode(
            boxes_xyxy=sample,
            logits=logits,
            class_threshold=class_threshold
            if class_threshold is not None
            else self.radm_config.class_threshold,
            nms_threshold=nms_threshold
            if nms_threshold is not None
            else self.radm_config.nms_threshold,
            output_type=output_type,
            return_intermediates=return_intermediates or return_raw_predictions,
            extra_intermediates=intermediates,
        )
        if isinstance(output, LayoutGenerationOutput) and trajectory is not None:
            output.trajectory = trajectory
        return output

    generate = __call__

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save pipeline components and RADM config.

        Args:
            save_directory: Output directory.
        """
        super().save_pretrained(save_directory)
        self.radm_config.save_config(save_directory)
        self.processor.save_pretrained(
            Path(save_directory) / self.radm_config.processor_subfolder
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path) -> Self:
        """Load a saved RADM pipeline.

        Args:
            pretrained_model_name_or_path: Local directory or Hub id.

        Returns:
            Loaded RADM pipeline.
        """
        config_dict, _ = RADMConfig.load_config(
            pretrained_model_name_or_path,
            return_unused_kwargs=True,
        )
        config = cast(RADMConfig, RADMConfig.from_config(config_dict))
        pipe = super().from_pretrained(pretrained_model_name_or_path, config=config)
        pipe.radm_config = config
        pipe.processor = RADMProcessor.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=config.processor_subfolder,
        )
        return pipe

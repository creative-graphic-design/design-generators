"""Diffusers pipeline for converted LayouSyn text-to-layout generation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from diffusers import DiffusionPipeline

from laygen.common import ConditionType, normalize_condition_type
from laygen.common.bbox import BoxFormat
from laygen.pipelines.pipeline_output import LayoutGenerationOutput

from .modeling_layousyn import LayouSynDiTModel
from .processing_layousyn import (
    LAYOUSYN_ASPECT_RATIO_KEY,
    LAYOUSYN_CAPTION_EMBEDS_KEY,
    LAYOUSYN_CAPTION_MASK_KEY,
    LAYOUSYN_CONCEPT_EMBEDS_KEY,
    LAYOUSYN_CONCEPT_MASK_KEY,
    LAYOUSYN_ID2LABEL_KEY,
    LAYOUSYN_LABEL_TEXTS_KEY,
    LAYOUSYN_PER_EXAMPLE_ID2LABEL_KEY,
    LayouSynProcessor,
)
from .scheduling_layousyn import LayouSynScheduler


class LayouSynPipeline(DiffusionPipeline):
    """Generate open-vocabulary scene layouts with LayouSyn.

    Args:
        model: Converted DiT denoiser.
        scheduler: LayouSyn Gaussian/DDIM scheduler.
        processor: Processor for prompt/concept inputs and postprocessing.
    """

    model_cpu_offload_seq = "model"
    _optional_components = ["processor"]

    def __init__(
        self,
        model: LayouSynDiTModel,
        scheduler: LayouSynScheduler,
        processor: LayouSynProcessor,
    ) -> None:
        """Initialize the pipeline."""
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        self.model = model
        self.scheduler = scheduler
        self.processor = processor
        self.model.eval()

    @property
    def components(self) -> dict[str, object]:
        """Return serializable pipeline components."""
        return {
            "model": self.model,
            "scheduler": self.scheduler,
            "processor": self.processor,
        }

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[str],
        **kwargs: object,
    ) -> None:
        """Save pipeline components plus processor metadata."""
        super().save_pretrained(save_directory, **kwargs)
        self.processor.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        **kwargs: object,
    ) -> "LayouSynPipeline":
        """Load pipeline and restore local processor metadata."""
        pipe = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(pipe, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(pipe).__name__}")
        processor_config = (
            Path(pretrained_model_name_or_path) / LayouSynProcessor.config_name
        )
        if pipe.processor is None and processor_config.exists():
            pipe.processor = LayouSynProcessor.from_pretrained(
                pretrained_model_name_or_path
            )
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompt: str | list[str] | None = None,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.text,
        labels: torch.Tensor | np.ndarray | list[object] | None = None,
        id2label: dict[int, str] | None = None,
        bbox: torch.Tensor | np.ndarray | list[object] | None = None,
        mask: torch.Tensor | np.ndarray | list[object] | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        aspect_ratio: float | list[float] | torch.Tensor = 1.0,
        num_inference_steps: int | None = None,
        guidance_scale: float = 2.0,
        sampling_type: Literal["ddim", "ddpm"] = "ddim",
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        caption_embeds: torch.Tensor | None = None,
        caption_padding_mask: torch.Tensor | None = None,
        concept_embeds: torch.Tensor | None = None,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor | object]:
        """Run LayouSyn denoising.

        Args:
            prompt: Caption text.
            batch_size: Number of generated layouts when labels are unbatched.
            seed: Convenience seed used only if ``generator`` is absent.
            generator: Exact reproducibility API.
            condition_type: Canonical condition name. First-class public mode is
                ``text``; unsupported modes fail explicitly.
            labels: String concepts or integer ids.
            id2label: Mapping for integer labels.
            bbox: Reserved for future initialization/refinement support.
            mask: Optional valid concept mask.
            num_elements: Optional expected element count. It is validated
                against labels when supplied.
            box_format: Public input bbox format.
            normalized: Whether input boxes are normalized.
            canvas_size: Required for pixel boxes.
            aspect_ratio: Scalar or per-example aspect ratio.
            num_inference_steps: Number of reverse diffusion steps.
            guidance_scale: Classifier-free guidance scale.
            sampling_type: ``ddim`` or ``ddpm``.
            output_type: ``dataclass`` or ``dict``.
            return_intermediates: Whether to return denoising trajectory.
            caption_embeds: Precomputed caption embeddings.
            caption_padding_mask: Precomputed caption padding mask.
            concept_embeds: Precomputed concept embeddings.

        Returns:
            Public layout output.
        """
        del batch_size
        canonical = normalize_condition_type(condition_type)
        if canonical is not ConditionType.text:
            raise NotImplementedError(
                f"LayouSyn public pipeline supports condition_type='text', got {condition_type}"
            )
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        encoded = self.processor(
            prompt=prompt,
            labels=labels,
            id2label=id2label,
            bbox=bbox,
            mask=mask,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
            aspect_ratio=aspect_ratio,
            caption_embeds=caption_embeds,
            caption_padding_mask=caption_padding_mask,
            concept_embeds=concept_embeds,
        )
        self._validate_num_elements(num_elements, encoded[LAYOUSYN_LABEL_TEXTS_KEY])
        concept_mask = encoded[LAYOUSYN_CONCEPT_MASK_KEY].to(self.device)
        batch = concept_mask.shape[0]
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        sample = self.scheduler.initial_sample(
            batch,
            self.processor.max_in_len,
            self.model.config.in_channels,
            device=self.device,
            generator=generator,
        )
        model_kwargs = {
            "x_padding_mask": concept_mask,
            "aspect_ratio": encoded[LAYOUSYN_ASPECT_RATIO_KEY].to(self.device),
            "concept_embeds": encoded[LAYOUSYN_CONCEPT_EMBEDS_KEY].to(self.device),
            "caption_embeds": encoded[LAYOUSYN_CAPTION_EMBEDS_KEY].to(self.device),
            "caption_padding_mask": encoded[LAYOUSYN_CAPTION_MASK_KEY].to(self.device),
        }
        if guidance_scale != 1.0:
            sample = torch.cat([sample, sample], dim=0)
            model_kwargs = self._cfg_model_kwargs(model_kwargs, batch)
        trajectory = []
        for index, timestep in enumerate(self.scheduler.timesteps):
            model_timestep = self.scheduler.model_timesteps[index]
            t_model = torch.full(
                (sample.shape[0],),
                int(model_timestep),
                device=self.device,
                dtype=torch.long,
            )
            t_step = torch.full(
                (sample.shape[0],), int(timestep), device=self.device, dtype=torch.long
            )
            if guidance_scale != 1.0:
                model_output = self.model.forward_with_cfg(
                    sample, t_model, guidance_scale=guidance_scale, **model_kwargs
                )
            else:
                model_output = self.model(sample, t_model, **model_kwargs)
            step = self.scheduler.step(
                model_output,
                t_step,
                sample,
                generator=generator,
                sampling_type=sampling_type,
            )
            sample = step.prev_sample
            if return_intermediates:
                trajectory.append(step.pred_original_sample[:batch].detach().cpu())
        sample = sample[:batch].clamp(-1.0, 1.0).detach().cpu()
        return self.processor.postprocess(
            sample,
            labels=encoded[LAYOUSYN_LABEL_TEXTS_KEY],
            id2label=encoded[LAYOUSYN_ID2LABEL_KEY],
            id2label_per_example=encoded[LAYOUSYN_PER_EXAMPLE_ID2LABEL_KEY],
            output_type=output_type,
            return_intermediates=return_intermediates,
            intermediates={"trajectory": trajectory} if return_intermediates else None,
        )

    generate = __call__

    def _cfg_model_kwargs(
        self, model_kwargs: dict[str, torch.Tensor], batch_size: int
    ) -> dict[str, torch.Tensor]:
        y_null = (
            self.model.y_embedder.y_embedding.to(self.device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        y_mask_null = (
            self.model.y_embedder.y_padding_mask.to(self.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        return {
            "x_padding_mask": torch.cat(
                [model_kwargs["x_padding_mask"], model_kwargs["x_padding_mask"]], dim=0
            ),
            "aspect_ratio": torch.cat(
                [model_kwargs["aspect_ratio"], model_kwargs["aspect_ratio"]], dim=0
            ),
            "concept_embeds": torch.cat(
                [model_kwargs["concept_embeds"], model_kwargs["concept_embeds"]], dim=0
            ),
            "caption_embeds": torch.cat(
                [model_kwargs["caption_embeds"], y_null], dim=0
            ),
            "caption_padding_mask": torch.cat(
                [model_kwargs["caption_padding_mask"], y_mask_null], dim=0
            ),
        }

    @staticmethod
    def _validate_num_elements(
        num_elements: int | list[int] | torch.Tensor | None, labels: list[list[str]]
    ) -> None:
        if num_elements is None:
            return
        if isinstance(num_elements, int):
            expected = [num_elements] * len(labels)
        elif isinstance(num_elements, torch.Tensor):
            expected = [int(item) for item in num_elements.tolist()]
        else:
            expected = [int(item) for item in num_elements]
        actual = [len(row) for row in labels]
        if expected != actual:
            raise ValueError(f"num_elements {expected} does not match labels {actual}")

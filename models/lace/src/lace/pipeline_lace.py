from __future__ import annotations

from typing import Any, Literal

import torch
from diffusers import DiffusionPipeline

from laygen.common.outputs_diffusers import LayoutGenerationOutput

from .constraints import beautify_layout
from .modeling_lace import LaceTransformerModel
from .processing_lace import LaceProcessor
from .scheduling_lace import LaceScheduler

_CONDITION_ALIASES = {
    None: "unconditional",
    "none": "unconditional",
    "uncond": "unconditional",
    "unconditional": "unconditional",
    "c": "label",
    "label": "label",
    "category": "label",
    "cwh": "label_size",
    "label_size": "label_size",
    "label+size": "label_size",
    "complete": "completion",
    "partial": "completion",
    "completion": "completion",
    "refine": "refinement",
    "refinement": "refinement",
}


def normalize_condition_type(condition_type: str | None) -> str:
    key = None if condition_type is None else condition_type.lower()
    try:
        return _CONDITION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported LACE condition_type: {condition_type}") from exc


class LacePipeline(DiffusionPipeline):
    model_cpu_offload_seq = "model"
    _optional_components = ["processor"]

    def __init__(
        self,
        model: LaceTransformerModel,
        scheduler: LaceScheduler,
        processor: LaceProcessor,
    ) -> None:
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        self.model = model
        self.scheduler = scheduler
        self.processor = processor
        self.model.eval()

    @property
    def components(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "scheduler": self.scheduler,
            "processor": self.processor,
        }

    @torch.no_grad()
    def __call__(
        self,
        *,
        batch_size: int = 1,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        condition_type: str | None = "unconditional",
        bbox: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        box_format: Literal["xywh", "ltwh", "ltrb"] = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        completion_ratio: float = 0.2,
        refinement_noise: float = 0.1,
        beautify: bool = False,
        beautify_overlap_weight: float | None = None,
        beautify_alignment_weight: float = 1.0,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        **model_kwargs: Any,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        del model_kwargs
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        canonical = normalize_condition_type(condition_type)
        encoded = None
        if canonical != "unconditional":
            if bbox is None or labels is None:
                raise ValueError(
                    f"bbox and labels are required for condition_type={condition_type}"
                )
            encoded = self.processor(
                bbox=bbox,
                labels=labels,
                mask=mask,
                box_format=box_format,
                normalized=normalized,
                canvas_size=canvas_size,
            )
            batch_size = encoded["layout"].shape[0]
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        sample = self.scheduler.initial_sample(
            batch_size,
            self.processor.max_seq_length,
            self.processor.seq_dim,
            device=self.device,
            generator=generator,
        )
        if canonical == "refinement":
            assert encoded is not None
            noise = torch.randn(
                encoded["bbox"].shape,
                dtype=encoded["bbox"].dtype,
                device=encoded["bbox"].device,
                generator=generator,
            )
            noisy_bbox = (encoded["bbox"] + refinement_noise * noise).clamp(0, 1)
            sample = self.processor.encode(
                noisy_bbox.to(self.device),
                encoded["labels"].to(self.device),
                encoded["mask"].to(self.device),
            )
        real_layout = None if encoded is None else encoded["layout"].to(self.device)
        fix_mask = self._build_fix_mask(
            canonical, real_layout, completion_ratio, generator
        )
        trajectory = [] if return_intermediates else None
        step_indices = (
            self.scheduler.refinement_indices()
            if canonical == "refinement"
            else list(range(len(self.scheduler.timesteps) - 1, -1, -1))
        )
        for index in step_indices:
            step = self.scheduler.ddim_timesteps[index]
            timestep = torch.full(
                (batch_size,), int(step.item()), device=self.device, dtype=torch.long
            )
            if real_layout is not None and fix_mask is not None:
                sample[fix_mask] = real_layout[fix_mask]
            model_output = self.model(sample=sample, timestep=timestep).sample
            out = self.scheduler.step(
                model_output, timestep, sample, index=index, generator=generator
            )
            sample = out.prev_sample
            if real_layout is not None and fix_mask is not None:
                sample[fix_mask] = real_layout[fix_mask]
            if trajectory is not None:
                trajectory.append(sample.detach().cpu())
        decoded = self.processor.decode(sample.detach().cpu())
        bbox_out = decoded.bbox
        mask_out = decoded.mask
        if beautify:
            overlap = beautify_overlap_weight
            if overlap is None:
                overlap = 1.0 if self.processor.dataset == "publaynet" else 0.0
            bbox_out, mask_out = beautify_layout(
                bbox_out,
                mask_out,
                overlap_weight=overlap,
                alignment_weight=beautify_alignment_weight,
            )
        output = LayoutGenerationOutput(
            bbox=bbox_out,
            labels=decoded.labels,
            mask=mask_out,
            id2label=decoded.id2label,
            trajectory=trajectory,
            intermediates={"condition_type": canonical}
            if return_intermediates
            else None,
        )
        if output_type == "dict":
            return dict(output)
        if output_type != "dataclass":
            raise ValueError(f"Unsupported output_type: {output_type}")
        return output

    generate = __call__

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        self.processor.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = kwargs.pop(
            "processor", LaceProcessor.from_pretrained(pretrained_model_name_or_path)
        )
        pipe = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        pipe.processor = processor
        return pipe

    def _build_fix_mask(
        self,
        condition_type: str,
        real_layout: torch.Tensor | None,
        completion_ratio: float,
        generator: torch.Generator | None,
    ) -> torch.BoolTensor | None:
        if real_layout is None or condition_type == "refinement":
            return None
        batch_size, seq_len, seq_dim = real_layout.shape
        num_class = seq_dim - 4
        if condition_type == "label":
            fix_mask = torch.zeros_like(real_layout, dtype=torch.bool)
            fix_mask[:, :, :num_class] = True
            return fix_mask
        if condition_type == "label_size":
            fix_mask = torch.zeros_like(real_layout, dtype=torch.bool)
            fix_indices = list(range(num_class)) + [num_class + 2, num_class + 3]
            fix_mask[:, :, fix_indices] = True
            return fix_mask
        if condition_type == "completion":
            labels = real_layout[:, :, :num_class].argmax(dim=2)
            real_mask = labels != (num_class - 1)
            cutoff = torch.rand(
                (), device=real_layout.device, generator=generator
            ).item()
            element_mask = (
                torch.rand(
                    batch_size,
                    seq_len,
                    device=real_layout.device,
                    generator=generator,
                )
                <= cutoff * completion_ratio
            ) & real_mask
            return element_mask.unsqueeze(-1).expand(-1, -1, seq_dim)
        raise ValueError(f"Unsupported condition_type: {condition_type}")

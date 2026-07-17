from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from diffusers import DiffusionPipeline

from layout_generation_common.outputs import LayoutGenerationOutput

from .configuration_layout_flow import LayoutFlowConfig
from .modeling_layout_flow import LayoutFlowTransformerModel
from .processing_layout_flow import LayoutFlowProcessor, normalize_condition_type
from .sampling import sample_initial_state
from .scheduling_layout_flow import LayoutFlowEulerScheduler


class LayoutFlowPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "model"

    def __init__(
        self,
        model: LayoutFlowTransformerModel,
        scheduler: LayoutFlowEulerScheduler,
        config: LayoutFlowConfig | None = None,
        processor: LayoutFlowProcessor | None = None,
    ) -> None:
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
        condition_type: str = "unconditional",
        labels: torch.Tensor | np.ndarray | list | None = None,
        bbox: torch.Tensor | np.ndarray | list | None = None,
        mask: torch.Tensor | np.ndarray | list | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: Literal["xywh", "ltwh", "ltrb"] = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float = 0.0,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        **model_kwargs,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        canonical = normalize_condition_type(condition_type)
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
        if canonical == "refinement":
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
        self.layout_flow_config.save_config(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = LayoutFlowConfig.from_config(pretrained_model_name_or_path)
        pipe = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        pipe.layout_flow_config = config
        pipe.processor = LayoutFlowProcessor(config)
        return pipe

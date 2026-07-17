from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from diffusers import DiffusionPipeline

from layout_generation_common.discrete import log_onehot_to_index
from layout_generation_common.outputs import LayoutGenerationOutput

from .conditioning import build_condition, normalize_condition_type
from .denoiser import LayoutDMDenoiser
from .processing_layout_dm import LayoutDMProcessor
from .sampling import LayoutDMSamplingConfig
from .scheduler import LayoutDMScheduler
from .tokenization_layout_dm import LayoutDMTokenizer


class LayoutDMPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "denoiser"

    def __init__(
        self,
        denoiser: LayoutDMDenoiser,
        scheduler: LayoutDMScheduler,
        tokenizer: LayoutDMTokenizer,
        processor: LayoutDMProcessor | None = None,
    ) -> None:
        super().__init__()
        self.register_modules(
            denoiser=denoiser, scheduler=scheduler, tokenizer=tokenizer
        )
        self.tokenizer = tokenizer
        self.processor = processor or LayoutDMProcessor(tokenizer)
        self.denoiser.eval()

    @torch.no_grad()
    def __call__(
        self,
        *,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: str = "unconditional",
        labels: torch.Tensor | np.ndarray | list[Any] | None = None,
        bbox: torch.Tensor | np.ndarray | list[Any] | None = None,
        mask: torch.Tensor | np.ndarray | list[Any] | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: Literal["xywh", "ltwh", "ltrb"] = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        sampling: Literal[
            "deterministic", "random", "gumbel", "top_k", "top_p", "top_k_top_p"
        ] = "random",
        temperature: float = 1.0,
        top_k: int = 5,
        top_p: float = 0.9,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        **model_kwargs: object,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        _ = (num_elements, model_kwargs)
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        canonical = normalize_condition_type(condition_type)
        condition = None
        if canonical != "unconditional":
            if bbox is None or labels is None:
                raise ValueError(
                    f"bbox and labels are required for condition_type={condition_type}"
                )
            processed = self.processor(
                bbox=bbox,
                labels=labels,
                mask=mask,
                box_format=box_format,
                normalized=normalized,
                canvas_size=canvas_size,
            )
            decoded_input = self.tokenizer.decode_layout(processed["input_ids"])
            condition = build_condition(
                self.tokenizer,
                cond_type=canonical,
                bbox=decoded_input["bbox"],
                labels=decoded_input["labels"],
                mask=decoded_input["mask"],
            )
            batch_size = condition.input_ids.shape[0]
        sampling_config = LayoutDMSamplingConfig(
            name=sampling,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_inference_steps=num_inference_steps,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        sample = self.scheduler.initial_sample(
            batch_size,
            self.tokenizer.config.max_token_length,
            device=self.device,
            condition=condition,
        )
        trajectory = [] if return_intermediates else None
        previous_timestep = self.scheduler.config.num_timesteps
        for timestep in self.scheduler.timesteps:
            timestep_batch = torch.full(
                (batch_size,),
                int(timestep.item()),
                device=self.device,
                dtype=torch.long,
            )
            input_ids = log_onehot_to_index(sample)
            logits = self.denoiser(input_ids=input_ids, timesteps=timestep_batch).logits
            out = self.scheduler.step(
                logits,
                timestep_batch,
                sample,
                previous_timestep=previous_timestep,
                sampling=sampling_config,
                condition=condition,
                generator=generator,
            )
            sample = out.prev_sample
            previous_timestep = int(timestep.item())
            if trajectory is not None:
                trajectory.append(log_onehot_to_index(sample).detach().cpu())
        sequences = log_onehot_to_index(sample).detach().cpu()
        decoded = self.tokenizer.decode_layout(sequences)
        output = LayoutGenerationOutput(
            bbox=decoded["bbox"],
            labels=decoded["labels"],
            mask=decoded["mask"],
            id2label=self.tokenizer.config.id2label,
            sequences=sequences,
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

    def save_pretrained(self, save_directory: str | Path, **kwargs: object) -> None:
        super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, **kwargs: object
    ) -> "LayoutDMPipeline":
        tokenizer = LayoutDMTokenizer.from_pretrained(pretrained_model_name_or_path)
        kwargs.setdefault("tokenizer", tokenizer)
        pipe = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        pipe.processor = LayoutDMProcessor(pipe.tokenizer)
        return pipe

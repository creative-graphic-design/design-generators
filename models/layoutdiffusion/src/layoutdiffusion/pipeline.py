"""Diffusers pipeline for converted LayoutDiffusion checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from diffusers import DiffusionPipeline

from laygen.common import ConditionType, normalize_condition_type
from laygen.common.bbox import BoxFormat
from laygen.common.discrete import index_to_log_onehot, log_onehot_to_index
from laygen.pipelines.pipeline_output import LayoutGenerationOutput

from .conditioning import build_condition
from .processing_layoutdiffusion import LayoutDiffusionProcessor
from .sampling import LayoutDiffusionSamplingConfig
from .scheduler import LayoutDiffusionScheduler
from .tokenization_layoutdiffusion import LayoutDiffusionTokenizer
from .transformer import LayoutDiffusionTransformer


class LayoutDiffusionPipeline(DiffusionPipeline):
    """Generate layouts with a converted LayoutDiffusion pipeline.

    Args:
        transformer: LayoutDiffusion transformer denoiser.
        scheduler: Categorical diffusion scheduler.
        tokenizer: LayoutDiffusion layout tokenizer.
        processor: Optional processor.

    Examples:
        >>> from layoutdiffusion import LayoutDiffusionConfig, LayoutDiffusionTokenizer
        >>> from layoutdiffusion import LayoutDiffusionScheduler, LayoutDiffusionTransformer
        >>> cfg = LayoutDiffusionConfig(dataset_name="publaynet", hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64, num_channels=8)
        >>> tok = LayoutDiffusionTokenizer(cfg)
        >>> pipe = LayoutDiffusionPipeline(
        ...     LayoutDiffusionTransformer(vocab_size=cfg.vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64, num_channels=8),
        ...     LayoutDiffusionScheduler(vocab_size=cfg.vocab_size, mask_token_id=cfg.mask_token_id, type_classes=cfg.type_classes, num_train_timesteps=2),
        ...     tok,
        ... )
        >>> pipe(batch_size=1, seed=0, num_inference_steps=1).bbox.shape[-1]
        4
    """

    model_cpu_offload_seq = "transformer"

    def __init__(
        self,
        transformer: LayoutDiffusionTransformer,
        scheduler: LayoutDiffusionScheduler,
        tokenizer: LayoutDiffusionTokenizer,
        processor: LayoutDiffusionProcessor | None = None,
    ) -> None:
        """Initialize and register pipeline modules."""
        super().__init__()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            tokenizer=tokenizer,
        )
        self.tokenizer = tokenizer
        self.processor = processor or LayoutDiffusionProcessor(tokenizer)
        self.transformer.eval()

    @torch.no_grad()
    def __call__(
        self,
        *,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.unconditional,
        labels: torch.Tensor | np.ndarray | list[object] | None = None,
        bbox: torch.Tensor | np.ndarray | list[object] | None = None,
        mask: torch.Tensor | np.ndarray | list[object] | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        sampling: LayoutDiffusionSamplingConfig | None = None,
        **model_kwargs: object,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        """Run LayoutDiffusion generation.

        Args:
            batch_size: Number of layouts for unconditional generation.
            seed: Seed used only when ``generator`` is omitted.
            generator: Optional torch generator. Takes precedence over ``seed``.
            condition_type: Canonical condition type or supported alias.
            labels: Optional conditional labels.
            bbox: Optional conditional boxes.
            mask: Optional conditional valid mask.
            num_elements: Optional element counts.
            box_format: Input box format.
            normalized: Whether conditional boxes are normalized.
            canvas_size: Pixel canvas size for unnormalized inputs.
            num_inference_steps: Optional shortened inference steps.
            output_type: ``"dataclass"`` or ``"dict"``.
            return_intermediates: Whether to include trajectories.
            sampling: Optional sampling config.
            **model_kwargs: Reserved compatibility kwargs.

        Returns:
            Layout output dataclass or dictionary.

        Raises:
            ValueError: If ``output_type`` is unsupported.
        """
        _ = model_kwargs
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        canonical = normalize_condition_type(condition_type)
        processed = self.processor(
            bbox=bbox,
            labels=labels,
            mask=mask,
            num_elements=num_elements,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )
        condition_input = processed.get("input_ids")
        processed_labels = None if labels is None else torch.as_tensor(labels)
        counts = processed.get("num_elements")
        condition = build_condition(
            self.tokenizer,
            condition_type=canonical,
            input_ids=condition_input,
            labels=processed_labels,
            num_elements=counts,
        )
        if condition is not None and condition.input_ids is not None:
            batch_size = condition.input_ids.shape[0]
        if condition is not None and canonical is ConditionType.refinement:
            if condition.input_ids is None:
                raise ValueError("refinement condition is missing input_ids")
            start_ids = condition.input_ids.to(self.device)
        else:
            start_ids = self.tokenizer.build_initial_tokens(
                batch_size=batch_size,
                num_elements=counts,
                labels=processed_labels,
                condition_type=str(canonical),
                generator=generator,
                device=self.device,
            )
        sample = index_to_log_onehot(start_ids, self.scheduler.config.vocab_size)
        start_step = None if condition is None else condition.start_step
        sample_config = sampling or LayoutDiffusionSamplingConfig(
            num_inference_steps=num_inference_steps
        )
        self.scheduler.set_timesteps(
            sample_config.num_inference_steps or num_inference_steps,
            start_step=start_step,
            device=self.device,
        )
        trajectory = [] if return_intermediates else None
        for timestep in self.scheduler.timesteps:
            timestep_batch = torch.full(
                (batch_size,),
                int(timestep.item()),
                device=self.device,
                dtype=torch.long,
            )
            input_ids = log_onehot_to_index(sample)
            logits = self.transformer(
                input_ids=input_ids,
                timesteps=timestep_batch,
                condition_ids=None
                if condition is None or condition.input_ids is None
                else condition.input_ids.to(self.device),
                condition_type=None if condition is None else str(condition.type),
            ).logits
            out = self.scheduler.step(
                logits,
                timestep_batch,
                sample,
                sampling=sample_config,
                condition=condition,
                generator=generator,
            )
            sample = out.prev_sample
            if trajectory is not None:
                trajectory.append(log_onehot_to_index(sample).detach().cpu())
        sequences = log_onehot_to_index(sample).detach().cpu()
        decoded = self.tokenizer.decode_layout(sequences)
        output = LayoutGenerationOutput(
            bbox=decoded["bbox"],
            labels=decoded["labels"],
            mask=decoded["mask"],
            id2label=self.tokenizer.config.id2label,
            sequences=sequences if return_intermediates else None,
            trajectory=trajectory,
            intermediates={"condition_type": str(canonical)}
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
        """Save a Diffusers pipeline directory."""
        super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, **kwargs: object
    ) -> "LayoutDiffusionPipeline":
        """Load a LayoutDiffusion pipeline and rebuild its processor."""
        tokenizer = LayoutDiffusionTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )
        kwargs.setdefault("tokenizer", tokenizer)
        pipe = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        pipe.processor = LayoutDiffusionProcessor(pipe.tokenizer)
        return pipe

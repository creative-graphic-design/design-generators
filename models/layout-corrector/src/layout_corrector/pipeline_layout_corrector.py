"""Diffusers pipeline wrapper for Layout-Corrector guided generation."""

from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import ClassVar, Sequence, assert_never

import numpy as np
import torch
from diffusers import DiffusionPipeline
from jaxtyping import Bool, Float, Int

from layout_dm.conditioning import (
    LayoutDMCondition,
    build_condition,
    normalize_condition_type,
)
from layout_dm.pipeline_layout_dm import LayoutDMPipeline
from layout_dm.processing_layout_dm import LayoutDMProcessor
from layout_dm.sampling import LayoutDMSamplingConfig
from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType
from laygen.common.discrete import index_to_log_onehot, log_onehot_to_index
from laygen.common.discrete import SamplingMode
from laygen.pipelines.pipeline_output import LayoutGenerationOutput

from .configuration_layout_corrector import CorrectorReconType
from .modeling_layout_corrector import LayoutCorrectorModel
from .sampling import (
    LayoutCorrectorSamplingConfig,
    add_confidence_gumbel_noise,
    CorrectorMaskMode,
    select_tokens_to_remask,
    should_apply_corrector,
)


class OutputType(StrEnum):
    """Supported Layout-Corrector pipeline output formats."""

    dataclass = auto()
    dict = auto()


def normalize_output_type(output_type: OutputType | str) -> OutputType:
    """Normalize a public output format string to ``OutputType``."""
    if isinstance(output_type, OutputType):
        return output_type
    try:
        return OutputType(output_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported output_type: {output_type}") from exc


class LayoutCorrectorPipeline(DiffusionPipeline):
    """Diffusers pipeline that applies Layout-Corrector during LayoutDM sampling.

    Args:
        layout_dm: Base LayoutDM pipeline.
        corrector: Corrector model used to score and remask tokens.
        processor: Optional processor for conditional layout inputs.

    Raises:
        ValueError: Pipeline construction does not raise directly.

    Examples:
        >>> LayoutCorrectorPipeline.from_pretrained  # doctest: +ELLIPSIS
        <bound method...
    """

    model_cpu_offload_seq: ClassVar[str] = "layout_dm.denoiser->corrector"

    def __init__(
        self,
        layout_dm: LayoutDMPipeline,
        corrector: LayoutCorrectorModel,
        processor: LayoutDMProcessor | None = None,
    ) -> None:
        """Initialize the composite pipeline.

        Args:
            layout_dm: Base LayoutDM pipeline.
            corrector: Confidence model used to remask low-confidence tokens.
            processor: Optional processor for conditional inputs.
        """
        super().__init__()
        self.register_modules(layout_dm=layout_dm, corrector=corrector)
        self.layout_dm = layout_dm
        self.corrector = corrector
        self.processor = processor or layout_dm.processor
        self.corrector.eval()

    @torch.no_grad()
    def __call__(
        self,
        *,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.unconditional,
        labels: Int[torch.Tensor, "batch elements"]
        | Int[np.ndarray, "batch elements"]
        | list[object]
        | None = None,
        bbox: Float[torch.Tensor, "batch elements 4"]
        | Float[np.ndarray, "batch elements 4"]
        | list[object]
        | None = None,
        mask: Bool[torch.Tensor, "batch elements"]
        | Bool[np.ndarray, "batch elements"]
        | list[object]
        | None = None,
        num_elements: int | list[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        sampling: SamplingMode | str = SamplingMode.random,
        temperature: float = 1.0,
        top_k: int = 5,
        top_p: float = 0.9,
        corrector_steps: int | None = None,
        corrector_t_list: Sequence[int] | None = None,
        corrector_start: int = -1,
        corrector_end: int = -1,
        corrector_mask_mode: CorrectorMaskMode | str | None = None,
        corrector_mask_threshold: float | None = None,
        corrector_temperature: float | None = None,
        use_gumbel_noise: bool | None = None,
        gumbel_temperature: float | None = None,
        time_adaptive_temperature: bool | None = None,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        """Generate layouts with optional Layout-Corrector guidance.

        Args:
            batch_size: Number of layouts to sample for unconditional generation.
            seed: Optional seed used when `generator` is not supplied.
            generator: Optional PyTorch generator.
            condition_type: Condition mode such as `"unconditional"` or `"label"`.
            labels: Optional class ids for conditional generation.
            bbox: Optional boxes for conditional generation.
            mask: Optional element mask for conditional generation.
            num_elements: Reserved for future element-count conditioning.
            box_format: Coordinate format for conditional boxes.
            normalized: Whether conditional boxes are normalized.
            canvas_size: Pixel canvas used when `normalized=False`.
            num_inference_steps: Optional inference timestep count.
            sampling: Base LayoutDM sampling strategy.
            temperature: Base sampling temperature.
            top_k: Top-k cutoff for top-k sampling.
            top_p: Nucleus cutoff for top-p sampling.
            corrector_steps: Optional override for correction passes.
            corrector_t_list: Optional explicit correction timesteps.
            corrector_start: Range start for correction when no list is supplied.
            corrector_end: Range end for correction when no list is supplied.
            corrector_mask_mode: Optional override for remasking mode.
            corrector_mask_threshold: Optional threshold override.
            corrector_temperature: Optional confidence temperature override.
            use_gumbel_noise: Optional confidence-noise override.
            gumbel_temperature: Optional confidence-noise temperature override.
            time_adaptive_temperature: Optional adaptive-noise override.
            output_type: `"dataclass"` or `"dict"`.
            return_intermediates: Whether to include scores and trajectory.

        Returns:
            `LayoutGenerationOutput` by default, or a dictionary when requested.

        Raises:
            ValueError: If conditional generation is missing `bbox` or `labels`, or
                if `output_type` is unsupported.

        Examples:
            >>> LayoutCorrectorPipeline.__call__  # doctest: +ELLIPSIS
            <function...
        """
        _ = num_elements
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        canonical = normalize_condition_type(condition_type)
        condition = None
        if canonical != "unconditional":
            missing_inputs = [
                name
                for name, value in (("bbox", bbox), ("labels", labels))
                if value is None
            ]
            if missing_inputs:
                raise ValueError(
                    "bbox and labels are required "
                    f"for condition_type={condition_type}; "
                    f"missing {', '.join(missing_inputs)}"
                )
            processor_inputs = {
                "bbox": bbox,
                "labels": labels,
                "mask": mask,
                "box_format": box_format,
                "normalized": normalized,
                "canvas_size": canvas_size,
            }
            processed = self.processor(**processor_inputs)
            decoded_input = self.layout_dm.tokenizer.decode_layout(
                processed["input_ids"]
            )
            condition = build_condition(
                self.layout_dm.tokenizer,
                cond_type=canonical,
                bbox=decoded_input["bbox"],
                labels=decoded_input["labels"],
                mask=decoded_input["mask"],
            )
            batch_size = condition.input_ids.shape[0]

        corrector_cfg = LayoutCorrectorSamplingConfig(
            sampling=sampling,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_inference_steps=num_inference_steps,
            corrector_steps=corrector_steps or self.corrector.corrector_steps,
            corrector_t_list=tuple(
                self.corrector.corrector_t_list
                if corrector_t_list is None
                else corrector_t_list
            ),
            corrector_start=corrector_start,
            corrector_end=corrector_end,
            corrector_mask_mode=corrector_mask_mode
            or self.corrector.corrector_mask_mode,
            corrector_mask_threshold=corrector_mask_threshold
            if corrector_mask_threshold is not None
            else self.corrector.corrector_mask_threshold,
            corrector_temperature=corrector_temperature
            if corrector_temperature is not None
            else self.corrector.corrector_temperature,
            use_gumbel_noise=use_gumbel_noise
            if use_gumbel_noise is not None
            else self.corrector.use_gumbel_noise,
            gumbel_temperature=gumbel_temperature
            if gumbel_temperature is not None
            else self.corrector.gumbel_temperature,
            time_adaptive_temperature=time_adaptive_temperature
            if time_adaptive_temperature is not None
            else self.corrector.time_adaptive_temperature,
        )
        sampling_cfg = LayoutDMSamplingConfig(
            name=sampling,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_inference_steps=num_inference_steps,
        )
        self.layout_dm.scheduler.set_timesteps(num_inference_steps, device=self.device)
        sample = self.layout_dm.scheduler.initial_sample(
            batch_size,
            self.layout_dm.tokenizer.config.max_token_length,
            device=self.device,
            condition=condition,
        )
        trajectory = [] if return_intermediates else None
        scores = [] if return_intermediates else None
        previous_timestep = self.layout_dm.scheduler.config.num_timesteps
        for timestep in self.layout_dm.scheduler.timesteps:
            timestep_value = int(timestep.item())
            timestep_batch = torch.full(
                (batch_size,),
                timestep_value,
                device=self.device,
                dtype=torch.long,
            )
            if should_apply_corrector(timestep_value, corrector_cfg):
                sample, confidence = self._step_with_corrector(
                    sample=sample,
                    timestep_batch=timestep_batch,
                    condition=condition,
                    sampling=corrector_cfg,
                    generator=generator,
                )
                if scores is not None and confidence is not None:
                    scores.append(confidence.detach().cpu())
            else:
                input_ids = log_onehot_to_index(sample)
                logits = self.layout_dm.denoiser(
                    input_ids=input_ids, timesteps=timestep_batch
                ).logits
                sample = self.layout_dm.scheduler.step(
                    logits,
                    timestep_batch,
                    sample,
                    previous_timestep=previous_timestep,
                    sampling=sampling_cfg,
                    condition=condition,
                    generator=generator,
                ).prev_sample
            previous_timestep = timestep_value
            if trajectory is not None:
                trajectory.append(log_onehot_to_index(sample).detach().cpu())

        sequences = log_onehot_to_index(sample).detach().cpu()
        decoded = self.layout_dm.tokenizer.decode_layout(sequences)
        output = LayoutGenerationOutput(
            bbox=decoded["bbox"],
            labels=decoded["labels"],
            mask=decoded["mask"],
            id2label=dict(self.corrector.id2label),
            sequences=sequences,
            scores=torch.stack(scores) if scores else None,
            trajectory=trajectory,
            intermediates={"condition_type": canonical}
            if return_intermediates
            else None,
        )
        normalized_output_type = normalize_output_type(output_type)
        if normalized_output_type is OutputType.dict:
            return dict(output)
        if normalized_output_type is OutputType.dataclass:
            return output
        assert_never(normalized_output_type)

    generate = __call__

    def _step_with_corrector(
        self,
        *,
        sample: torch.Tensor,
        timestep_batch: torch.Tensor,
        condition: LayoutDMCondition | None,
        sampling: LayoutCorrectorSamplingConfig,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        confidence = None
        current = sample
        for _ in range(sampling.corrector_steps):
            input_ids = log_onehot_to_index(current)
            denoiser_logits = self.layout_dm.denoiser(
                input_ids=input_ids, timesteps=timestep_batch
            ).logits
            model_log_prob = self.layout_dm.scheduler.predict_start(denoiser_logits)
            if self.corrector.recon_type is CorrectorReconType.x_t_minus_1:
                model_log_prob = self.layout_dm.scheduler.q_posterior(
                    model_log_prob, current, timestep_batch
                )
                model_log_prob[:, self.layout_dm.tokenizer.mask_token_id, :] = -70.0
            if self.layout_dm.scheduler.token_mask is not None:
                valid = self.layout_dm.scheduler.token_mask.to(
                    model_log_prob.device
                ).T.unsqueeze(0)
                model_log_prob = model_log_prob.masked_fill(~valid, -70.0)
            if condition is not None:
                strong_mask = condition.mask.to(model_log_prob.device).unsqueeze(1)
                strong_log_prob = index_to_log_onehot(
                    condition.input_ids.to(model_log_prob.device),
                    self.layout_dm.scheduler.vocab_size,
                )
                model_log_prob = torch.where(
                    strong_mask, strong_log_prob, model_log_prob
                )
            x0_recon_ids = torch.multinomial(
                (model_log_prob.permute(0, 2, 1) / sampling.temperature)
                .softmax(dim=-1)
                .reshape(-1, model_log_prob.size(1)),
                1,
                generator=generator,
            ).reshape(model_log_prob.shape[0], model_log_prob.shape[-1])
            confidence = self.corrector.calc_confidence_score(
                x0_recon_ids,
                timestep_batch,
                padding_mask=x0_recon_ids == self.layout_dm.tokenizer.pad_token_id,
            )
            adjusted = confidence
            mask_ratio = self._mask_ratio(timestep_batch)
            if sampling.use_gumbel_noise:
                adjusted = add_confidence_gumbel_noise(
                    adjusted,
                    timestep=timestep_batch,
                    mask_ratio=mask_ratio,
                    temperature=sampling.gumbel_temperature,
                    time_adaptive_temperature=sampling.time_adaptive_temperature,
                    generator=generator,
                )
            remask = select_tokens_to_remask(
                adjusted,
                mask_ratio=mask_ratio,
                mode=sampling.corrector_mask_mode,
                threshold=sampling.corrector_mask_threshold,
                temperature=sampling.corrector_temperature,
            )
            x0_recon_ids = x0_recon_ids.masked_fill(
                remask, self.layout_dm.tokenizer.mask_token_id
            )
            if condition is not None:
                x0_recon_ids = torch.where(
                    condition.mask.to(x0_recon_ids.device),
                    condition.input_ids.to(x0_recon_ids.device),
                    x0_recon_ids,
                )
            current = index_to_log_onehot(
                x0_recon_ids, self.layout_dm.scheduler.vocab_size
            )
        return current, confidence

    def _mask_ratio(self, timestep_batch: torch.Tensor) -> float:
        timestep = int(timestep_batch[0].item())
        timestep = max(0, min(timestep, self.layout_dm.scheduler.config.num_timesteps))
        if timestep == 0:
            return 0.0
        return float(timestep / self.layout_dm.scheduler.config.num_timesteps)

    def save_pretrained(
        self, save_directory: str | Path, *, safe_serialization: bool = True
    ) -> None:
        """Save the nested LayoutDM pipeline and corrector model.

        Args:
            save_directory: Destination directory.
            safe_serialization: Whether to save weights as safetensors.

        Returns:
            None.

        Raises:
            OSError: If files cannot be written.

        Examples:
            >>> LayoutCorrectorPipeline.save_pretrained  # doctest: +ELLIPSIS
            <function...
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.layout_dm.save_pretrained(
            save_path / "layout_dm", safe_serialization=safe_serialization
        )
        self.corrector.save_pretrained(
            save_path / "corrector", safe_serialization=safe_serialization
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        processor: LayoutDMProcessor | None = None,
    ) -> "LayoutCorrectorPipeline":
        """Load a Layout-Corrector pipeline from a saved directory.

        Args:
            pretrained_model_name_or_path: Directory containing `layout_dm/` and
                `corrector/` subdirectories.
            processor: Optional processor override.

        Returns:
            Loaded `LayoutCorrectorPipeline`.

        Raises:
            OSError: If nested component files are missing.

        Examples:
            >>> LayoutCorrectorPipeline.from_pretrained  # doctest: +ELLIPSIS
            <bound method...
        """
        path = Path(pretrained_model_name_or_path)
        layout_dm = LayoutDMPipeline.from_pretrained(path / "layout_dm")
        corrector = LayoutCorrectorModel.from_pretrained(path / "corrector")
        return cls(layout_dm=layout_dm, corrector=corrector, processor=processor)

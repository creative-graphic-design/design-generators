"""Diffusers pipeline for LACE layout generation."""

from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Final, assert_never

import torch
from diffusers import DiffusionPipeline

from laygen.common import ConditionType
from laygen.common import normalize_condition_type as normalize_shared_condition_type
from laygen.common.bbox import BoxFormat
from laygen.common.labels import DatasetName
from laygen.common.outputs_diffusers import LayoutGenerationOutput

from .configuration_lace import normalize_dataset
from .constraints import beautify_layout
from .modeling_lace import LaceTransformerModel
from .processing_lace import LaceProcessor
from .scheduling_lace import LaceScheduler


class PipelineOutputType(StrEnum):
    """Supported LACE pipeline output containers."""

    dataclass = auto()
    dict = auto()


class LaceConditionAlias(StrEnum):
    """LACE-specific public condition aliases not in the shared registry."""

    none = auto()
    category = auto()
    label_size_plus = "label+size"


_LACE_CONDITION_ALIASES: Final[dict[LaceConditionAlias, ConditionType]] = {
    LaceConditionAlias.none: ConditionType.unconditional,
    LaceConditionAlias.category: ConditionType.label,
    LaceConditionAlias.label_size_plus: ConditionType.label_size,
}

_SUPPORTED_CONDITION_TYPES: Final[frozenset[ConditionType]] = frozenset(
    {
        ConditionType.unconditional,
        ConditionType.label,
        ConditionType.label_size,
        ConditionType.completion,
        ConditionType.refinement,
    }
)


def normalize_condition_type(
    condition_type: ConditionType | str | None,
) -> ConditionType:
    """Normalize public condition aliases.

    Args:
        condition_type: Canonical condition enum, string alias, or ``None``.

    Returns:
        Canonical condition enum.

    Raises:
        ValueError: If the condition type is unsupported.

    Examples:
        >>> normalize_condition_type("cwh") is ConditionType.label_size
        True
    """
    if isinstance(condition_type, ConditionType):
        canonical = condition_type
    elif condition_type is None:
        canonical = ConditionType.unconditional
    else:
        try:
            canonical = normalize_shared_condition_type(condition_type)
        except ValueError:
            key = condition_type.lower().replace("-", "_")
            try:
                canonical = _LACE_CONDITION_ALIASES[LaceConditionAlias(key)]
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported LACE condition_type: {condition_type}"
                ) from exc
    if canonical not in _SUPPORTED_CONDITION_TYPES:
        raise ValueError(f"Unsupported LACE condition_type: {condition_type}")
    return canonical


def normalize_output_type(output_type: PipelineOutputType | str) -> PipelineOutputType:
    """Normalize public output type aliases.

    Args:
        output_type: Output enum or string value.

    Returns:
        Canonical output type.

    Raises:
        ValueError: If the output type is unsupported.
    """
    if isinstance(output_type, PipelineOutputType):
        return output_type
    try:
        return PipelineOutputType(output_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported output_type: {output_type}") from exc


class LacePipeline(DiffusionPipeline):
    """Generate layouts with a converted LACE checkpoint.

    Args:
        model: LACE transformer denoiser.
        scheduler: DDIM-style scheduler.
        processor: Processor that encodes and decodes layout tensors.

    Examples:
        >>> from lace import LaceProcessor, LaceScheduler, LaceTransformerModel
        >>> model = LaceTransformerModel(seq_dim=10, max_seq_length=2, num_layers=1, dim_transformer=8, nhead=2, dim_feedforward=16)
        >>> pipe = LacePipeline(model=model, scheduler=LaceScheduler(ddim_num_steps=1), processor=LaceProcessor.from_dataset("publaynet"))
        >>> pipe.processor.max_seq_length
        25
    """

    model_cpu_offload_seq = "model"
    _optional_components = ["processor"]

    def __init__(
        self,
        model: LaceTransformerModel,
        scheduler: LaceScheduler,
        processor: LaceProcessor,
    ) -> None:
        """Initialize a LACE pipeline."""
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

    @torch.no_grad()
    def __call__(
        self,
        *,
        batch_size: int = 1,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        condition_type: ConditionType | str | None = ConditionType.unconditional,
        bbox: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        completion_ratio: float = 0.2,
        refinement_noise: float = 0.1,
        beautify: bool = False,
        beautify_overlap_weight: float | None = None,
        beautify_alignment_weight: float = 1.0,
        output_type: PipelineOutputType | str = PipelineOutputType.dataclass,
        return_intermediates: bool = False,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        """Run LACE denoising and return generated layouts.

        Args:
            batch_size: Number of layouts to generate for unconditional calls.
            num_inference_steps: Number of DDIM steps. Uses scheduler default if
                omitted.
            generator: Optional torch generator. Takes precedence over ``seed``.
            seed: Convenience seed used only when ``generator`` is absent.
            condition_type: Conditioning mode or alias.
            bbox: Conditioning boxes for non-unconditional modes.
            labels: Conditioning labels for non-unconditional modes.
            mask: Optional conditioning mask.
            box_format: Input box format for conditioning boxes.
            normalized: Whether conditioning boxes are normalized.
            canvas_size: Pixel canvas size required when ``normalized`` is false.
            completion_ratio: Maximum random completion fraction.
            refinement_noise: Noise scale for refinement conditioning.
            beautify: Whether to run the aesthetic post-optimization.
            beautify_overlap_weight: Optional overlap penalty override.
            beautify_alignment_weight: Alignment penalty weight.
            output_type: ``"dataclass"`` or ``"dict"``.
            return_intermediates: Whether to return the denoising trajectory.

        Returns:
            Layout output dataclass or a dictionary.

        Raises:
            ValueError: If a condition/output mode is unsupported or required
                conditioning tensors are missing.
        """
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        canonical = normalize_condition_type(condition_type)
        encoded = None
        if canonical is not ConditionType.unconditional:
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
        if canonical is ConditionType.refinement:
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
            if canonical is ConditionType.refinement
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
                overlap = (
                    1.0
                    if normalize_dataset(self.processor.dataset)
                    is DatasetName.publaynet
                    else 0.0
                )
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
            intermediates={"condition_type": str(canonical)}
            if return_intermediates
            else None,
        )
        out_type = normalize_output_type(output_type)
        if out_type is PipelineOutputType.dict:
            return dict(output)
        if out_type is PipelineOutputType.dataclass:
            return output
        assert_never(out_type)

    generate = __call__

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save pipeline components.

        Args:
            save_directory: Output directory.
        """
        super().save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        processor: LaceProcessor | None = None,
    ) -> "LacePipeline":
        """Load a saved LACE pipeline.

        Args:
            pretrained_model_name_or_path: Local path or Hub id.
            processor: Optional processor override.

        Returns:
            Loaded pipeline with the serialized processor attached.
        """
        loaded_processor = (
            LaceProcessor.from_pretrained(pretrained_model_name_or_path)
            if processor is None
            else processor
        )
        pipe = super().from_pretrained(pretrained_model_name_or_path)
        pipe.processor = loaded_processor
        return pipe

    def _build_fix_mask(
        self,
        condition_type: ConditionType,
        real_layout: torch.Tensor | None,
        completion_ratio: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor | None:
        """Build the fixed-channel mask for conditional generation."""
        if real_layout is None or condition_type is ConditionType.refinement:
            return None
        batch_size, seq_len, seq_dim = real_layout.shape
        num_class = seq_dim - 4
        if condition_type is ConditionType.label:
            fix_mask = torch.zeros_like(real_layout, dtype=torch.bool)
            fix_mask[:, :, :num_class] = True
            return fix_mask
        if condition_type is ConditionType.label_size:
            fix_mask = torch.zeros_like(real_layout, dtype=torch.bool)
            fix_indices = list(range(num_class)) + [num_class + 2, num_class + 3]
            fix_mask[:, :, fix_indices] = True
            return fix_mask
        if condition_type is ConditionType.completion:
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
        if condition_type is ConditionType.unconditional:
            return None
        raise ValueError(f"Unsupported LACE condition_type: {condition_type}")

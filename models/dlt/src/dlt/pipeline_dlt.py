"""Diffusers pipeline for DLT layout generation."""

from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Self, assert_never, cast

import torch
from diffusers import DiffusionPipeline
from jaxtyping import Bool, Float, Int

from laygen.common import ConditionType
from laygen.common import normalize_condition_type as normalize_shared_condition_type
from laygen.common.bbox import BoxFormat
from laygen.pipelines.pipeline_output import LayoutGenerationOutput

from .configuration_dlt import DLTConfig
from .modeling_dlt import DLT
from .processing_dlt import DLTProcessor
from .scheduling_dlt import DLTJointDiffusionScheduler, DLTJointSchedulerOutput


class OutputType(StrEnum):
    """DLT pipeline output containers."""

    dataclass = auto()
    dict = auto()


class DLTConditionAlias(StrEnum):
    """DLT original condition aliases."""

    all = auto()
    whole_box = auto()
    loc = auto()


_DLT_CONDITION_ALIASES: dict[DLTConditionAlias, ConditionType] = {
    DLTConditionAlias.all: ConditionType.unconditional,
    DLTConditionAlias.whole_box: ConditionType.label,
    DLTConditionAlias.loc: ConditionType.label_size,
}

_SUPPORTED_CONDITION_TYPES = frozenset(
    {ConditionType.unconditional, ConditionType.label, ConditionType.label_size}
)


def normalize_condition_type(
    condition_type: ConditionType | str | None,
) -> ConditionType:
    """Normalize public and original DLT condition names."""
    if condition_type is None:
        canonical = ConditionType.unconditional
    elif isinstance(condition_type, ConditionType):
        canonical = condition_type
    else:
        try:
            canonical = normalize_shared_condition_type(condition_type)
        except ValueError:
            key = condition_type.lower().replace("-", "_")
            try:
                canonical = _DLT_CONDITION_ALIASES[DLTConditionAlias(key)]
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported DLT condition_type: {condition_type}"
                ) from exc
    if canonical not in _SUPPORTED_CONDITION_TYPES:
        raise ValueError(f"Unsupported DLT condition_type: {condition_type}")
    return canonical


def _require_condition_inputs(
    *,
    condition_type: ConditionType | str | None,
    bbox: torch.Tensor | None,
    labels: torch.Tensor | None,
) -> None:
    """Validate DLT conditioned generation inputs."""
    if bbox is not None and labels is not None:
        return
    raise ValueError(
        f"bbox and labels are required for condition_type={condition_type}"
    )


def _format_pipeline_output(
    output: LayoutGenerationOutput, output_kind: OutputType
) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
    """Convert a DLT output dataclass to the requested public container."""
    match output_kind:
        case OutputType.dataclass:
            return output
        case OutputType.dict:
            return dict(output)
        case _:
            assert_never(output_kind)


class DLTPipeline(DiffusionPipeline):
    """Generate layouts with a converted DLT checkpoint.

    Args:
        model: DLT denoiser.
        scheduler: Joint box/category scheduler.
        config: Pipeline configuration.
        processor: Layout processor.
    """

    model_cpu_offload_seq = "model"
    _optional_components = ["processor"]

    def __init__(
        self,
        model: DLT,
        scheduler: DLTJointDiffusionScheduler,
        config: DLTConfig,
        processor: DLTProcessor | None = None,
    ) -> None:
        """Initialize a DLT pipeline."""
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        self.dlt_config = config
        self.processor = processor or DLTProcessor(
            dataset=self.dlt_config.dataset_name,
            labels=tuple(self.dlt_config.id2label.values()),
            max_num_comp=self.dlt_config.max_num_comp,
        )
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
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str | None = ConditionType.unconditional,
        labels: Int[torch.Tensor, "batch elements"] | None = None,
        bbox: Float[torch.Tensor, "batch elements 4"] | None = None,
        mask: Bool[torch.Tensor, "batch elements"] | None = None,
        num_elements: int | list[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        temperature: float | None = None,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        """Run DLT joint denoising and return generated layouts.

        Args:
            batch_size: Number of layouts to generate.
            seed: Optional seed used when ``generator`` is absent.
            generator: Optional torch generator. Takes precedence over ``seed``.
            condition_type: Canonical condition or DLT original alias.
            labels: Optional public labels for conditioned modes.
            bbox: Optional public boxes for conditioned modes.
            mask: Optional valid-element mask.
            num_elements: Optional valid element count for unconditional calls.
            box_format: Input box format.
            normalized: Whether input boxes are normalized.
            canvas_size: Pixel canvas size for non-normalized boxes.
            num_inference_steps: Number of reverse diffusion steps.
            temperature: Optional category sampling temperature override.
            output_type: ``"dataclass"`` or ``"dict"``.
            return_intermediates: Whether to include denoising trajectory.

        Returns:
            Layout generation output dataclass or dictionary.

        Raises:
            ValueError: If the condition or output type is unsupported.
        """
        canonical = normalize_condition_type(condition_type)
        output_kind = OutputType(output_type)
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        if canonical is ConditionType.unconditional:
            processed = self.processor.empty_condition(
                batch_size=batch_size, device=self.device
            )
            if num_elements is not None:
                lengths = torch.as_tensor(
                    num_elements, dtype=torch.long, device=self.device
                )
                if lengths.ndim == 0:
                    lengths = lengths.repeat(batch_size)
                processed["mask"] = (
                    torch.arange(self.processor.max_num_comp, device=self.device)[
                        None, :
                    ]
                    < lengths[:, None]
                )
        else:
            _require_condition_inputs(
                condition_type=condition_type, bbox=bbox, labels=labels
            )
            processed = self.processor(
                bbox=bbox,
                labels=labels,
                mask=mask,
                box_format=box_format,
                normalized=normalized,
                canvas_size=canvas_size,
                device=self.device,
            )
            batch_size = processed["box"].shape[0]
        mask_box, mask_cat = self.processor.condition_masks(
            str(canonical), mask=processed["mask"]
        )
        sample = {
            "box": processed["box"],
            "box_cond": processed["box_cond"],
            "cat": processed["cat"],
            "mask_box": mask_box,
            "mask_cat": mask_cat,
        }
        noisy_batch = {
            "box": torch.randn(
                processed["box"].shape,
                dtype=processed["box"].dtype,
                device=self.device,
                generator=generator,
            ),
            "cat": torch.full(
                processed["cat"].shape,
                self.processor.mask_category_id,
                dtype=torch.long,
                device=self.device,
            ),
        }
        original_temperature = self.scheduler.temperature
        if temperature is not None:
            self.scheduler.temperature = temperature
        steps = max(1, num_inference_steps or self.scheduler.num_train_timesteps)
        trajectory: list[torch.Tensor] | None = [] if return_intermediates else None
        bbox_step: DLTJointSchedulerOutput | None = None
        try:
            for i in range(steps - 1, -1, -1):
                t_value = min(i, self.scheduler.num_train_timesteps - 1)
                t = torch.tensor([t_value] * batch_size, device=self.device)
                bbox_pred, cat_pred = self.model(sample, noisy_batch, timesteps=t)
                bbox_step, cat_step = self.scheduler.step_jointly(
                    bbox_pred,
                    {"cat": cat_pred},
                    timestep=t,
                    sample=noisy_batch["box"],
                    generator=generator,
                )
                noisy_batch["box"] = bbox_step.prev_sample
                noisy_batch["cat"] = cat_step["cat"]
                if trajectory is not None:
                    trajectory.append(noisy_batch["box"].detach().cpu())
        finally:
            self.scheduler.temperature = original_temperature
        if bbox_step is None:
            raise RuntimeError("DLT denoising did not run any scheduler steps")
        final_box = (
            sample["mask_box"] * bbox_step.pred_original_sample
            + (1 - sample["mask_box"]) * sample["box_cond"]
        )
        final_cat = (
            sample["mask_cat"] * noisy_batch["cat"]
            + (1 - sample["mask_cat"]) * sample["cat"]
        )
        valid_mask = processed["mask"].detach().cpu()
        output = LayoutGenerationOutput(
            bbox=self.processor.internal_to_public_boxes(final_box).detach().cpu()
            * valid_mask.unsqueeze(-1),
            labels=self.processor.internal_to_public_labels(
                final_cat, processed["mask"]
            )
            .detach()
            .cpu(),
            mask=valid_mask,
            id2label=self.processor.id2label,
            trajectory=trajectory,
            intermediates={"condition_type": str(canonical)}
            if return_intermediates
            else None,
        )
        return _format_pipeline_output(output, output_kind)

    generate = __call__

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Persist DLT model, scheduler, and pipeline metadata."""
        super().save_pretrained(save_directory, safe_serialization=False)
        self.dlt_config.save_config(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path) -> Self:
        """Load a saved DLT pipeline."""
        config_dict, _ = DLTConfig.load_config(
            pretrained_model_name_or_path, return_unused_kwargs=True
        )
        config = cast(DLTConfig, DLTConfig.from_config(config_dict))
        pipe = super().from_pretrained(pretrained_model_name_or_path, config=config)
        pipe.dlt_config = config
        pipe.processor = DLTProcessor(
            dataset=config.dataset_name,
            labels=tuple(config.id2label.values()),
            max_num_comp=config.max_num_comp,
        )
        return pipe

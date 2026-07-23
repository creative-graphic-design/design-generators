"""Diffusers pipeline for CGB-DM content-aware layout generation."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Final

import torch
from diffusers import DiffusionPipeline

from laygen.common import ConditionType
from laygen.common import normalize_condition_type as normalize_shared_condition_type
from laygen.common.bbox import BoxFormat
from laygen.pipelines.pipeline_output import LayoutGenerationOutput

from .modeling_cgb_dm import CGBDMTransformerModel
from .processing_cgb_dm import CGBDMProcessor
from .scheduling_cgb_dm import CGBDMScheduler


class OutputType(StrEnum):
    """Supported CGB-DM pipeline output containers."""

    dataclass = auto()
    dict = auto()


_SUPPORTED_CONDITION_TYPES: Final[frozenset[ConditionType]] = frozenset(
    {
        ConditionType.content_image,
        ConditionType.label,
        ConditionType.label_size,
        ConditionType.completion,
        ConditionType.refinement,
    }
)


def normalize_condition_type(
    condition_type: ConditionType | str | None,
) -> ConditionType:
    """Normalize CGB-DM condition aliases.

    Args:
        condition_type: Canonical condition enum, alias, or ``None``.

    Returns:
        Canonical condition enum.

    Raises:
        ValueError: If the condition is unsupported.

    Examples:
        >>> str(normalize_condition_type("uncond"))
        'content_image'
    """
    if condition_type is None:
        canonical = ConditionType.content_image
    elif isinstance(condition_type, ConditionType):
        canonical = condition_type
    else:
        key = condition_type.lower().replace("-", "_")
        canonical = (
            ConditionType.content_image
            if key == "uncond"
            else normalize_shared_condition_type(condition_type)
        )
    if canonical is ConditionType.unconditional:
        raise ValueError(
            "CGB-DM requires image/content; use condition_type='content_image'"
        )
    if canonical not in _SUPPORTED_CONDITION_TYPES:
        raise ValueError(f"Unsupported CGB-DM condition_type: {condition_type}")
    return canonical


def normalize_output_type(output_type: OutputType | str) -> OutputType:
    """Normalize output container aliases."""
    if isinstance(output_type, OutputType):
        return output_type
    try:
        return OutputType(output_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported output_type: {output_type}") from exc


class CGBDMPipeline(DiffusionPipeline):
    """Generate content-aware poster layouts with CGB-DM.

    Args:
        model: CGB-DM denoiser.
        scheduler: CGB-DM scheduler.
        processor: Processor for images and layouts.

    Examples:
        >>> model = CGBDMTransformerModel(num_labels=4, max_seq_length=2, image_size=(32, 32), dim_model=16, n_head=2, feature_dim=32, num_layers=1)
        >>> pipe = CGBDMPipeline(model=model, scheduler=CGBDMScheduler(num_train_timesteps=10, ddim_num_steps=1), processor=CGBDMProcessor(max_seq_length=2, image_size=(32, 32)))
        >>> pipe.processor.seq_dim
        8
    """

    model_cpu_offload_seq = "model"

    def __init__(
        self,
        model: CGBDMTransformerModel,
        scheduler: CGBDMScheduler,
        processor: CGBDMProcessor,
    ) -> None:
        """Initialize pipeline components."""
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler, processor=processor)
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
        image: object | None = None,
        content: dict[str, object] | None = None,
        saliency: object | None = None,
        saliency_isnet: object | None = None,
        saliency_basnet: object | None = None,
        saliency_box: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        condition_type: ConditionType | str = ConditionType.content_image,
        labels: torch.Tensor | list[object] | None = None,
        bbox: torch.Tensor | list[object] | None = None,
        mask: torch.Tensor | list[object] | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        completion_ratio: float = 0.2,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Run DDIM sampling and return generated layouts.

        Args:
            image: RGB image or batch of images.
            content: Optional content container with ``image`` and saliency keys.
            saliency: Optional merged saliency map.
            saliency_isnet: Optional first saliency map.
            saliency_basnet: Optional second saliency map.
            saliency_box: Optional normalized center ``xywh`` saliency box.
            pixel_values: Preprocessed four-channel image tensor.
            batch_size: Number of layouts when ``pixel_values`` is synthetic.
            seed: Convenience seed used when ``generator`` is absent.
            generator: Optional torch generator. Takes precedence over ``seed``.
            condition_type: Canonical condition mode or alias.
            labels: Conditioning labels for constrained modes.
            bbox: Conditioning boxes for constrained modes.
            mask: Optional valid-element mask.
            num_elements: Accepted for interface compatibility.
            box_format: Input box format.
            normalized: Whether input boxes are normalized.
            canvas_size: Canvas size required for pixel boxes.
            num_inference_steps: DDIM step count.
            completion_ratio: Completion conditioning keep ratio.
            output_type: ``"dataclass"`` or ``"dict"``.
            return_intermediates: Whether to include trajectory/debug tensors.

        Returns:
            Layout output dataclass or dictionary.

        Raises:
            ValueError: If required content or conditioning inputs are absent.
        """
        del num_elements
        canonical = normalize_condition_type(condition_type)
        out_type = normalize_output_type(output_type)
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        if pixel_values is None:
            if content is not None:
                image = content.get("image", image)
                saliency = content.get("saliency", saliency)
            encoded_content = self.processor(
                image,
                saliency=saliency,
                saliency_isnet=saliency_isnet,
                saliency_basnet=saliency_basnet,
                saliency_box=saliency_box,
            )
            pixel_values = encoded_content["pixel_values"]
            resolved_saliency_box = encoded_content["saliency_box"]
        else:
            if saliency_box is None:
                resolved_saliency_box = torch.zeros(pixel_values.shape[0], 1, 4)
            else:
                resolved_saliency_box = 2 * (
                    torch.as_tensor(saliency_box, dtype=torch.float32).clamp(0.0, 1.0)
                    - 0.5
                )
                if resolved_saliency_box.ndim == 2:
                    resolved_saliency_box = resolved_saliency_box.unsqueeze(1)
        batch_size = int(
            pixel_values.shape[0] if pixel_values is not None else batch_size
        )
        encoded_layout = None
        if canonical is not ConditionType.content_image:
            if bbox is None or labels is None:
                raise ValueError(
                    f"bbox and labels are required for condition_type={condition_type}"
                )
            encoded_layout = self.processor.encode_layout(
                bbox=bbox,
                labels=labels,
                mask=mask,
                box_format=box_format,
                normalized=normalized,
                canvas_size=canvas_size,
            )
            batch_size = encoded_layout["layout"].shape[0]
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        sample = self.scheduler.initial_sample(
            batch_size,
            self.processor.max_seq_length,
            self.processor.seq_dim,
            device=self.device,
            generator=generator,
        )
        real_layout = (
            None if encoded_layout is None else encoded_layout["layout"].to(self.device)
        )
        fix_mask = None
        if real_layout is not None:
            fix_mask = self.scheduler.condition_mask(
                real_layout,
                canonical,
                completion_ratio=completion_ratio,
                generator=generator,
            )
            sample = torch.where(fix_mask, real_layout, sample)
        pixel_values = pixel_values.to(self.device)
        resolved_saliency_box = resolved_saliency_box.to(self.device)
        trajectory: list[torch.Tensor] = []
        cgb_weights: list[torch.Tensor] = []
        for index, timestep in enumerate(self.scheduler.timesteps):
            timestep_batch = torch.full(
                (batch_size,),
                int(timestep.item()),
                device=self.device,
                dtype=torch.long,
            )
            model_out = self.model(
                sample, pixel_values, resolved_saliency_box, timestep_batch
            )
            step = self.scheduler.step(
                model_out.sample,
                timestep_batch,
                sample,
                len(self.scheduler.timesteps) - index - 1,
                generator=generator,
            )
            sample = step.prev_sample
            if real_layout is not None and fix_mask is not None:
                sample = torch.where(fix_mask, real_layout, sample)
            if return_intermediates:
                trajectory.append(sample.detach().cpu())
                if model_out.cgb_weight is not None:
                    cgb_weights.append(model_out.cgb_weight.detach().cpu())
        intermediates = None
        if return_intermediates:
            intermediates = {
                "condition_type": str(canonical),
                "saliency_box": resolved_saliency_box.detach().cpu(),
                "cgb_weight": cgb_weights[-1] if cgb_weights else None,
            }
        return self._decode(sample, out_type, trajectory, intermediates)

    def _decode(
        self,
        sample: torch.Tensor,
        output_type: OutputType,
        trajectory: list[torch.Tensor],
        intermediates: object | None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        decoded = self.processor.decode(
            sample.detach().cpu(),
            output_type="dataclass",
            intermediates=intermediates,
        )
        output = LayoutGenerationOutput(
            bbox=decoded.bbox,
            labels=decoded.labels,
            mask=decoded.mask,
            id2label=decoded.id2label,
            sequences=decoded.sequences,
            scores=decoded.scores,
            trajectory=trajectory or None,
            intermediates=decoded.intermediates,
        )
        if output_type is OutputType.dict:
            return dict(output)
        return output

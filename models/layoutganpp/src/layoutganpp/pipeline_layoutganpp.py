from __future__ import annotations

from typing import Literal

import torch
from transformers import Pipeline

from laygen.common.outputs import LayoutGenerationOutput

from .modeling_layoutganpp import LayoutGANPPModel
from .processing_layoutganpp import LayoutGANPPProcessor


class LayoutGANPPPipeline(Pipeline):
    def __init__(
        self,
        model: LayoutGANPPModel,
        processor: LayoutGANPPProcessor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model=model, tokenizer=None, framework="pt", **kwargs)
        self.processor = processor or LayoutGANPPProcessor(
            dataset_name=model.config.dataset_name,
            id2label=model.config.id2label,
        )

    def _sanitize_parameters(self, **kwargs):
        return {}, kwargs, {}

    def preprocess(self, inputs=None, **kwargs):
        labels = kwargs.pop("labels", inputs)
        if labels is None:
            raise ValueError("labels are required for LayoutGANPPPipeline")
        encoded = self.processor(labels)
        encoded.update(kwargs)
        return encoded

    def _forward(self, model_inputs):
        labels = model_inputs.pop("labels")
        attention_mask = model_inputs.pop("attention_mask")
        return self.model.generate(
            labels=labels,
            attention_mask=attention_mask,
            **model_inputs,
        )

    def postprocess(self, model_outputs, **kwargs):
        return model_outputs

    @torch.no_grad()
    def __call__(
        self,
        labels: list[list[str | int]]
        | list[str | int]
        | torch.LongTensor
        | None = None,
        *,
        batch_size: int = 1,
        condition_type: str = "label",
        bbox: torch.FloatTensor | None = None,
        mask: torch.BoolTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        num_elements: int | list[int] | torch.LongTensor | None = None,
        box_format: str = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        latents: torch.FloatTensor | None = None,
        **model_kwargs,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        del batch_size
        if labels is None:
            raise ValueError("labels are required for layoutganpp v1")
        if isinstance(labels, torch.Tensor):
            encoded_labels = labels
            resolved_mask = attention_mask if attention_mask is not None else mask
        else:
            encoded = self.processor(labels)
            encoded_labels = encoded["labels"]
            if attention_mask is not None:
                resolved_mask = attention_mask
            else:
                resolved_mask = encoded["attention_mask"] if mask is None else mask
        return self.model.generate(
            condition_type=condition_type,
            bbox=bbox,
            labels=encoded_labels,
            mask=resolved_mask,
            num_elements=num_elements,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
            seed=seed,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            return_intermediates=return_intermediates,
            latents=latents,
            **model_kwargs,
        )

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        self.model.save_pretrained(save_directory, **kwargs)
        self.processor.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model = LayoutGANPPModel.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        processor = LayoutGANPPProcessor.from_pretrained(pretrained_model_name_or_path)
        return cls(model=model, processor=processor)

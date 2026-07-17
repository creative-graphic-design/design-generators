"""Pipeline interface for LayoutGAN++ layout generation."""

from __future__ import annotations

from typing import Literal

import torch
from transformers import Pipeline
from transformers.tokenization_utils_base import BatchEncoding

from laygen.common.outputs import LayoutGenerationOutput

from .modeling_layoutganpp import LayoutGANPPModel
from .processing_layoutganpp import LayoutGANPPProcessor


class LayoutGANPPPipeline(Pipeline):
    """Transformers pipeline for LayoutGAN++ label-conditioned generation.

    Args:
        model: LayoutGAN++ model instance.
        processor: Optional processor for label encoding and decoding.
        **kwargs: Extra `Pipeline` keyword arguments.

    Examples:
        >>> model = LayoutGANPPModel(LayoutGANPPConfig(num_labels=2))
        >>> pipe = LayoutGANPPPipeline(model=model)
        >>> pipe.model.config.model_type
        'layoutganpp'
    """

    def __init__(
        self,
        model: LayoutGANPPModel,
        processor: LayoutGANPPProcessor | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize a LayoutGAN++ pipeline.

        Args:
            model: LayoutGAN++ model instance.
            processor: Optional processor for label encoding and decoding.
            **kwargs: Extra `Pipeline` keyword arguments.

        Examples:
            >>> pipe = LayoutGANPPPipeline(LayoutGANPPModel(LayoutGANPPConfig()))
            >>> isinstance(pipe.processor, LayoutGANPPProcessor)
            True
        """
        super().__init__(model=model, tokenizer=None, framework="pt", **kwargs)
        self.processor = processor or LayoutGANPPProcessor(
            dataset_name=model.config.dataset_name,
            id2label=model.config.id2label,
        )

    def _sanitize_parameters(
        self, **kwargs: object
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        return {}, kwargs, {}

    def preprocess(self, inputs: object = None, **kwargs: object) -> BatchEncoding:
        """Encode pipeline inputs into model inputs.

        Args:
            inputs: Labels supplied as the positional pipeline input.
            **kwargs: Keyword labels and generation arguments.

        Returns:
            Batch encoding containing label IDs, attention mask, and generation kwargs.

        Raises:
            ValueError: If labels are not supplied or cannot be encoded.

        Examples:
            >>> pipe = LayoutGANPPPipeline(LayoutGANPPModel(LayoutGANPPConfig()))
            >>> "labels" in pipe.preprocess(["Toolbar"])
            True
        """
        labels = kwargs.pop("labels", inputs)
        if labels is None:
            raise ValueError("labels are required for LayoutGANPPPipeline")
        encoded = self.processor(labels)
        encoded.update(kwargs)
        return encoded

    def _forward(
        self, model_inputs: dict[str, object]
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        labels = torch.as_tensor(model_inputs.pop("labels"), dtype=torch.long)
        attention_mask = torch.as_tensor(
            model_inputs.pop("attention_mask"), dtype=torch.bool
        )
        return self.model.generate(
            labels=labels,
            attention_mask=attention_mask,
            **model_inputs,
        )

    def postprocess(
        self,
        model_outputs: LayoutGenerationOutput | dict[str, torch.Tensor],
        **kwargs: object,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        """Return generated layouts from the pipeline output.

        Args:
            model_outputs: Output produced by `LayoutGANPPModel.generate`.
            **kwargs: Reserved post-processing keyword arguments.

        Returns:
            The generated layout output unchanged.

        Examples:
            >>> output = LayoutGenerationOutput(bbox=torch.zeros(1, 1, 4))
            >>> pipe = LayoutGANPPPipeline(LayoutGANPPModel(LayoutGANPPConfig()))
            >>> pipe.postprocess(output) is output
            True
        """
        del kwargs
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
        **model_kwargs: object,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
        """Generate LayoutGAN++ boxes from labels.

        Args:
            labels: Label strings or label IDs to condition on.
            batch_size: Reserved compatibility argument.
            condition_type: Condition type or alias.
            bbox: Reserved compatibility argument.
            mask: Optional valid-element mask.
            attention_mask: Optional valid-element mask.
            num_elements: Reserved compatibility argument.
            box_format: Reserved compatibility argument.
            normalized: Reserved compatibility argument.
            canvas_size: Reserved compatibility argument.
            seed: Optional random seed for latent sampling.
            generator: Optional PyTorch random generator.
            num_inference_steps: Reserved compatibility argument.
            output_type: Return format, either `dataclass` or `dict`.
            return_intermediates: Whether to include generation intermediates.
            latents: Optional fixed latent vectors.
            **model_kwargs: Additional keyword arguments forwarded to generation.

        Returns:
            A layout generation dataclass or dictionary.

        Raises:
            ValueError: If labels are missing or generation options are invalid.

        Examples:
            >>> pipe = LayoutGANPPPipeline(LayoutGANPPModel(LayoutGANPPConfig()))
            >>> out = pipe(labels=["Toolbar"], seed=0)
            >>> tuple(out.bbox.shape)
            (1, 1, 4)
        """
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

    def save_pretrained(self, save_directory: str, **kwargs: object) -> None:
        """Save the pipeline model and processor.

        Args:
            save_directory: Directory where model and processor files are written.
            **kwargs: Extra keyword arguments passed to model saving.

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> pipe = LayoutGANPPPipeline(LayoutGANPPModel(LayoutGANPPConfig()))
            >>> with TemporaryDirectory() as tmp:
            ...     pipe.save_pretrained(tmp)
        """
        self.model.save_pretrained(save_directory, **kwargs)
        self.processor.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: object
    ) -> "LayoutGANPPPipeline":
        """Load a LayoutGAN++ pipeline from a pretrained directory or Hub ID.

        Args:
            pretrained_model_name_or_path: Local path or Hugging Face Hub model ID.
            **kwargs: Extra keyword arguments passed to model loading.

        Returns:
            A loaded `LayoutGANPPPipeline`.

        Raises:
            OSError: If model or processor files cannot be loaded.

        Examples:
            >>> # LayoutGANPPPipeline.from_pretrained("creative-graphic-design/layoutganpp-rico")
        """
        model = LayoutGANPPModel.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        processor = LayoutGANPPProcessor.from_pretrained(pretrained_model_name_or_path)
        return cls(model=model, processor=processor)

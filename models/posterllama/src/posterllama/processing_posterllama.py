"""Processor for PosterLlama prompt construction and output parsing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from os import PathLike
from pathlib import Path
from typing import Final, Literal, cast

import torch
from jaxtyping import Bool, Float, Int
from transformers import BatchEncoding, ProcessorMixin

from laygen.common.bbox import BoxFormat, denormalize_boxes, prepare_layout_tensors
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.modeling_outputs import LayoutGenerationOutput

from .configuration_posterllama import PosterLlamaConfig
from .image_processing_posterllama import PosterLlamaImageProcessor
from .postprocessing import parse_rectangles, rect_ltwh_to_output

VENDOR_CONDITION_ALIASES: Final[dict[str, ConditionType]] = {
    "cond_cate_to_size_pos": ConditionType.label,
    "cond_cate_size_to_pos": ConditionType.label_size,
    "cond_recover_mask": ConditionType.completion,
    "cond_random_mask": ConditionType.completion,
    "cond_cate_pos_to_size": ConditionType.refinement,
}
SUPPORTED_CONDITIONS: Final[frozenset[ConditionType]] = frozenset(
    {
        ConditionType.content_image,
        ConditionType.unconditional,
        ConditionType.label,
        ConditionType.label_size,
        ConditionType.completion,
        ConditionType.refinement,
    }
)
UNSUPPORTED_CONDITIONS: Final[frozenset[ConditionType]] = frozenset(
    {
        ConditionType.text,
        ConditionType.relation,
        ConditionType.hierarchical,
        ConditionType.retrieval,
    }
)
PROMPT_PREFIX: Final[str] = (
    "Generate poster layout markup using <rect> elements inside one <svg>."
)
RECT_TEMPLATE: Final[str] = (
    '<rect data-category="{label}" x="{x}" y="{y}" width="{width}" height="{height}"/>'
)
FILL_TEMPLATE: Final[str] = "<FILL_{}>"


class PosterLlamaProcessor(ProcessorMixin):
    """Build PosterLlama prompts and decode generated HTML/SVG layouts.

    Args:
        image_processor: Image processor metadata wrapper.
        config: Explicit PosterLlama configuration.

    Examples:
        >>> processor = PosterLlamaProcessor.from_config(PosterLlamaConfig())
        >>> "Generate poster layout" in processor.build_prompt(condition_type="unconditional")
        True
    """

    attributes = ["image_processor"]
    image_processor_class = "PosterLlamaImageProcessor"

    def __init__(
        self,
        image_processor: PosterLlamaImageProcessor,
        config: PosterLlamaConfig,
    ) -> None:
        """Initialize processor components."""
        self.image_processor = image_processor
        self.config = config
        super().__init__(image_processor)

    @classmethod
    def from_config(cls, config: PosterLlamaConfig) -> "PosterLlamaProcessor":
        """Create a processor from an explicit config.

        Args:
            config: Processor configuration.

        Returns:
            PosterLlamaProcessor instance.

        Examples:
            >>> PosterLlamaProcessor.from_config(PosterLlamaConfig()).config.model_type
            'posterllama'
        """
        return cls(
            image_processor=PosterLlamaImageProcessor(
                vision_encoder_repo_id=config.vision_encoder_repo_id,
            ),
            config=config,
        )

    def save_pretrained(
        self,
        save_directory: str | PathLike[str],
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> None:
        """Save processor metadata.

        Args:
            save_directory: Directory to write.
            push_to_hub: Accepted for ProcessorMixin compatibility; ignored.
            kwargs: Accepted for ProcessorMixin compatibility; ignored.
        """
        _ = (push_to_hub, kwargs)
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(root)
        self.image_processor.save_pretrained(root)
        (root / "processor_config.json").write_text(
            json.dumps(
                {
                    "processor_class": self.__class__.__name__,
                    "image_processor_class": self.image_processor.__class__.__name__,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        *,
        subfolder: str | None = None,
        **kwargs: object,
    ) -> "PosterLlamaProcessor":
        """Load processor metadata.

        Args:
            pretrained_model_name_or_path: Checkpoint root or processor folder.
            cache_dir: Accepted for ProcessorMixin compatibility.
            force_download: Accepted for ProcessorMixin compatibility.
            local_files_only: Whether to avoid network access.
            token: Accepted for ProcessorMixin compatibility.
            revision: Accepted for ProcessorMixin compatibility.
            subfolder: Optional processor subfolder.
            kwargs: Accepted for ProcessorMixin compatibility.

        Returns:
            Loaded PosterLlamaProcessor.
        """
        _ = (cache_dir, force_download, token, revision, kwargs)
        root = Path(pretrained_model_name_or_path)
        if subfolder is not None:
            root = root / subfolder
        config = PosterLlamaConfig.from_pretrained(
            root, local_files_only=local_files_only
        )
        return cls(
            image_processor=PosterLlamaImageProcessor.from_pretrained(root),
            config=config,
        )

    def normalize_condition_type(
        self,
        condition_type: ConditionType | str,
    ) -> ConditionType:
        """Normalize PosterLlama condition aliases.

        Args:
            condition_type: Canonical condition or PosterLlama release alias.

        Returns:
            Supported canonical condition.

        Raises:
            NotImplementedError: If the condition is known but unsupported.
            ValueError: If the condition is unknown.
        """
        if isinstance(condition_type, str):
            alias = condition_type.lower().replace("-", "_")
            if alias in VENDOR_CONDITION_ALIASES:
                return VENDOR_CONDITION_ALIASES[alias]
        condition = normalize_condition_type(condition_type)
        if condition in UNSUPPORTED_CONDITIONS:
            raise NotImplementedError(f"PosterLlama does not support {condition}")
        if condition not in SUPPORTED_CONDITIONS:
            raise NotImplementedError(f"PosterLlama does not support {condition}")
        return condition

    def __call__(
        self,
        *,
        images: object = None,
        prompt: str | Sequence[str] | None = None,
        content: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        texts: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
        batch_size: int = 1,
        condition_type: ConditionType | str = ConditionType.content_image,
        labels: Int[torch.Tensor, "..."]
        | Sequence[Sequence[int | str]]
        | Sequence[int | str]
        | None = None,
        bbox: Float[torch.Tensor, "..."] | Sequence[object] | None = None,
        mask: Bool[torch.Tensor, "..."] | Sequence[object] | None = None,
        num_elements: int | Sequence[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding:
        """Encode public inputs into recipe prompt and image tensors.

        Args:
            images: Poster image inputs.
            prompt: Optional user-provided prompt text.
            content: Optional content metadata.
            texts: Optional poster text strings.
            batch_size: Batch size when no images are supplied.
            condition_type: Canonical condition or PosterLlama release alias.
            labels: Optional element label constraints.
            bbox: Optional element boxes.
            mask: Optional valid-element mask.
            num_elements: Optional requested element count.
            box_format: Input box format.
            normalized: Whether boxes are normalized.
            canvas_size: Canvas size for pixel boxes and prompt rendering.
            return_tensors: Tensor return format. Only ``pt`` is supported.

        Returns:
            BatchEncoding containing prompt strings and tensors.
        """
        condition = self.normalize_condition_type(condition_type)
        canvas = self._resolve_canvas_size(canvas_size)
        prompt_text = self.build_prompt(
            condition_type=condition,
            prompt=prompt,
            content=content,
            texts=texts,
            labels=labels,
            bbox=bbox,
            mask=mask,
            num_elements=num_elements,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas,
        )
        image_batch = self.image_processor(images, return_tensors=return_tensors)
        return BatchEncoding(
            {
                "pixel_values": image_batch["pixel_values"],
                "prompts": prompt_text
                if isinstance(prompt_text, list)
                else [prompt_text] * batch_size,
                "condition_type": condition,
                "canvas_size": canvas,
            }
        )

    def build_prompt(
        self,
        *,
        condition_type: ConditionType | str = ConditionType.content_image,
        prompt: str | Sequence[str] | None = None,
        content: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        texts: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
        labels: Int[torch.Tensor, "..."]
        | Sequence[Sequence[int | str]]
        | Sequence[int | str]
        | None = None,
        bbox: Float[torch.Tensor, "..."] | Sequence[object] | None = None,
        mask: Bool[torch.Tensor, "..."] | Sequence[object] | None = None,
        num_elements: int | Sequence[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
    ) -> str | list[str]:
        """Build a deterministic PosterLlama HTML prompt.

        Args:
            condition_type: Canonical condition or PosterLlama release alias.
            prompt: Optional prompt prefix override.
            content: Optional content metadata included in diagnostics text.
            texts: Optional poster text strings.
            labels: Optional label constraints.
            bbox: Optional box constraints.
            mask: Optional valid-element mask.
            num_elements: Requested element count for unconstrained slots.
            box_format: Input box format.
            normalized: Whether boxes are normalized.
            canvas_size: Canvas size as ``(width, height)``.

        Returns:
            Prompt string or list of prompt strings.
        """
        _ = content
        condition = self.normalize_condition_type(condition_type)
        canvas = self._resolve_canvas_size(canvas_size)
        base_prompt = self._first_prompt(prompt)
        text_line = self._texts_line(texts)
        known_markup = self._constraint_markup(
            condition=condition,
            labels=labels,
            bbox=bbox,
            mask=mask,
            num_elements=num_elements,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas,
        )
        body = (
            f"{base_prompt}\n"
            f"Condition: {condition.value}\n"
            f"{text_line}"
            f'<body> <svg width="{canvas[0]}" height="{canvas[1]}"> '
            f"{known_markup} </svg> </body>"
        )
        return self.config.prompt_template.format(body)

    def parse_output(
        self,
        text: str,
        *,
        canvas_size: tuple[int, int] | None = None,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        strict: bool = False,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Parse generated HTML/SVG into public layout output.

        Args:
            text: Generated markup.
            canvas_size: Canvas size override.
            output_type: Return dataclass or dictionary.
            return_intermediates: Whether to include parse diagnostics.
            strict: Whether malformed rectangles raise.

        Returns:
            LayoutGenerationOutput or dictionary.
        """
        parsed = parse_rectangles(text, self._label2id(), strict=strict)
        canvas = canvas_size or parsed.canvas_size or self.config.canvas_size
        if canvas is None:
            raise ValueError(
                "canvas_size is required when generated SVG lacks width/height"
            )
        resolved_canvas = (int(canvas[0]), int(canvas[1]))
        output = rect_ltwh_to_output(
            parsed,
            canvas_size=resolved_canvas,
            id2label=cast(dict[int, str], self.config.id2label),
            return_intermediates=return_intermediates,
        )
        if output_type == "dict":
            return dict(output)
        if output_type != "dataclass":
            raise ValueError(f"Unsupported output_type: {output_type}")
        return output

    def _resolve_canvas_size(
        self,
        canvas_size: tuple[int, int] | None,
    ) -> tuple[int, int]:
        canvas = canvas_size or self.config.canvas_size
        if canvas is None:
            return (360, 504)
        return int(canvas[0]), int(canvas[1])

    def _first_prompt(self, prompt: str | Sequence[str] | None) -> str:
        if prompt is None:
            return PROMPT_PREFIX
        if isinstance(prompt, str):
            return prompt
        return next(iter(prompt), PROMPT_PREFIX)

    def _texts_line(
        self,
        texts: str | Sequence[str] | Sequence[Sequence[str]] | None,
    ) -> str:
        if texts is None:
            return ""
        if isinstance(texts, str):
            values = [texts]
        else:
            first = next(iter(texts), "")
            if isinstance(first, str):
                values = cast(list[str], list(texts))
            else:
                values = [str(value) for value in first]
        return "Texts: " + " | ".join(values) + "\n"

    def _constraint_markup(
        self,
        *,
        condition: ConditionType,
        labels: object,
        bbox: object,
        mask: object | None,
        num_elements: object,
        box_format: BoxFormat | str,
        normalized: bool,
        canvas_size: tuple[int, int],
    ) -> str:
        if labels is None:
            count = self._num_elements(num_elements)
            return " ".join(self._fill_rect(index) for index in range(count))
        label_tensor = self._labels_to_tensor(labels)
        bbox_tensor = self._bbox_to_tensor(
            bbox,
            labels=label_tensor,
            mask=mask,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )
        return " ".join(
            self._rect_for_condition(
                condition=condition,
                label=int(label),
                bbox_ltwh=(
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ),
                fill_index=index,
            )
            for index, (label, box) in enumerate(
                zip(label_tensor[0].tolist(), bbox_tensor[0].tolist(), strict=True)
            )
        )

    def _labels_to_tensor(self, labels: object) -> Int[torch.Tensor, "batch elements"]:
        if isinstance(labels, torch.Tensor):
            tensor = labels.long()
            return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor
        rows = cast(Sequence[object], labels)
        if rows and isinstance(rows[0], Sequence) and not isinstance(rows[0], str):
            row = cast(Sequence[object], rows[0])
        else:
            row = rows
        label2id = self._label2id()
        values = [
            label2id[self._normalize_input_label(item)]
            if isinstance(item, str)
            else int(cast(int, item))
            for item in row
        ]
        return torch.tensor([values], dtype=torch.long)

    def _bbox_to_tensor(
        self,
        bbox: object,
        *,
        labels: Int[torch.Tensor, "batch elements"],
        mask: object | None,
        box_format: BoxFormat | str,
        normalized: bool,
        canvas_size: tuple[int, int],
    ) -> Float[torch.Tensor, "batch elements 4"]:
        if bbox is None:
            return torch.zeros((labels.size(0), labels.size(1), 4), dtype=torch.float32)
        bbox_t, _, _ = prepare_layout_tensors(
            bbox=bbox,
            labels=labels,
            mask=mask,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )
        if bbox_t.ndim == 2:
            bbox_t = bbox_t.unsqueeze(0)
        return denormalize_boxes(bbox_t, canvas_size=canvas_size, box_format="ltwh")

    def _rect_for_condition(
        self,
        *,
        condition: ConditionType,
        label: int,
        bbox_ltwh: tuple[float, float, float, float],
        fill_index: int,
    ) -> str:
        label_name = str(cast(dict[int, str], self.config.id2label)[label])
        x, y, width, height = bbox_ltwh
        fill = FILL_TEMPLATE.format(fill_index)
        if condition is ConditionType.label:
            x = y = width = height = fill
        elif condition is ConditionType.label_size:
            x = y = fill
        elif condition is ConditionType.completion:
            return self._fill_rect(fill_index)
        elif condition is ConditionType.refinement:
            width = height = fill
        elif condition in {ConditionType.content_image, ConditionType.unconditional}:
            return self._fill_rect(fill_index)
        return RECT_TEMPLATE.format(
            label=label_name,
            x=x,
            y=y,
            width=width,
            height=height,
        )

    def _fill_rect(self, index: int) -> str:
        fill = FILL_TEMPLATE.format(index)
        return RECT_TEMPLATE.format(label=fill, x=fill, y=fill, width=fill, height=fill)

    def _num_elements(self, num_elements: object) -> int:
        if num_elements is None:
            return 1
        if isinstance(num_elements, torch.Tensor):
            return int(num_elements.flatten()[0].item())
        if isinstance(num_elements, Sequence) and not isinstance(num_elements, str):
            return int(cast(int, num_elements[0]))
        return int(cast(int, num_elements))

    def _label2id(self) -> dict[str, int]:
        return {
            self._normalize_input_label(label): int(index)
            for index, label in cast(dict[int, str], self.config.id2label).items()
        }

    def _normalize_input_label(self, label: object) -> str:
        return str(label).strip().lower().replace("_", " ")

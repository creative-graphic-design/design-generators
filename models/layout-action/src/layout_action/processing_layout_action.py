"""Processor for LayoutAction conditions and output decoding."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from transformers import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding

from laygen.common.bbox import BoxFormat, prepare_layout_tensors
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.modeling_outputs import LayoutGenerationOutput

from .configuration_layout_action import LayoutActionConfig
from .tokenization_layout_action import LayoutActionTokenizer

OutputType = Literal["dataclass", "dict"]
PROCESSOR_CONFIG_FILE = "processor_config.json"
SUPPORTED_CONDITIONS = {
    ConditionType.unconditional,
    ConditionType.label,
    ConditionType.completion,
}


class LayoutActionProcessor(ProcessorMixin):
    """Prepare LayoutAction prompts and decode generated action sequences.

    Args:
        tokenizer: LayoutAction tokenizer.

    Examples:
        >>> processor = LayoutActionProcessor(LayoutActionTokenizer(LayoutActionConfig(max_elements=1)))
        >>> encoded = processor(condition_type="unconditional")
        >>> encoded["input_ids"].shape
        torch.Size([1, 1])
    """

    attributes = ["tokenizer"]
    tokenizer_class = "LayoutActionTokenizer"

    def __init__(self, tokenizer: LayoutActionTokenizer | None = None) -> None:
        """Initialize the processor."""
        self.tokenizer = tokenizer or LayoutActionTokenizer(LayoutActionConfig())
        self.chat_template = None

    @property
    def config(self) -> LayoutActionConfig:
        """Return the paired LayoutAction config."""
        return self.tokenizer.config

    def __call__(
        self,
        *,
        condition_type: ConditionType | str = ConditionType.unconditional,
        bbox: torch.Tensor | np.ndarray | list[object] | None = None,
        labels: torch.Tensor | np.ndarray | list[object] | None = None,
        mask: torch.Tensor | np.ndarray | list[object] | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        batch_size: int = 1,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding:
        """Encode a public generation condition.

        Args:
            condition_type: Canonical condition or supported release alias.
            bbox: Optional public boxes for completion prompts.
            labels: Optional labels for label/completion prompts.
            mask: Optional valid-element mask.
            num_elements: Optional element count or completion prefix length.
            box_format: Input box format.
            normalized: Whether boxes are normalized.
            canvas_size: Canvas size for pixel boxes.
            batch_size: Batch size for unconditional generation.
            return_tensors: Tensor framework. Only ``pt`` is supported.

        Returns:
            Batch encoding with prompt ids and optional forced token ids.

        Raises:
            NotImplementedError: If the condition is unsupported by LayoutAction.
            ValueError: If required condition payloads are missing.
        """
        if return_tensors != "pt":
            raise ValueError("LayoutActionProcessor only supports return_tensors='pt'")
        condition = normalize_condition_type(condition_type)
        if condition not in SUPPORTED_CONDITIONS:
            raise NotImplementedError(f"LayoutAction does not support {condition}.")
        if condition is ConditionType.unconditional:
            input_ids = torch.full(
                (int(batch_size), 1), self.config.bos_token_id, dtype=torch.long
            )
            return BatchEncoding(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "max_new_tokens": self.config.max_token_length,
                }
            )
        if labels is None:
            raise ValueError(f"{condition} generation requires labels")
        if bbox is None:
            label_tensor = torch.as_tensor(labels)
            if label_tensor.ndim == 1:
                label_tensor = label_tensor.unsqueeze(0)
            mask_tensor = (
                torch.ones_like(label_tensor, dtype=torch.bool)
                if mask is None
                else torch.as_tensor(mask, dtype=torch.bool)
            )
            if mask_tensor.ndim == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            bbox_tensor = torch.zeros(
                (*label_tensor.shape, 4),
                dtype=torch.float32,
                device=label_tensor.device,
            )
        else:
            bbox_tensor, label_tensor, mask_tensor = prepare_layout_tensors(
                bbox=bbox,
                labels=self._labels_to_ids(labels),
                mask=mask,
                box_format=box_format,
                normalized=normalized,
                canvas_size=canvas_size,
                clamp_converted_normalized=True,
            )
        full = self.tokenizer.encode_layout(
            bbox=bbox_tensor,
            labels=label_tensor.long(),
            mask=mask_tensor.bool(),
        )
        if condition is ConditionType.label:
            input_ids = full[:, :1]
            forced = torch.full(
                (full.size(0), self.config.max_token_length),
                -100,
                dtype=torch.long,
                device=full.device,
            )
            for step in range(
                0,
                self.config.max_elements * self.config.element_token_width,
                self.config.element_token_width,
            ):
                source_index = step + 1
                if source_index < full.size(1):
                    forced[:, step] = full[:, source_index]
            return BatchEncoding(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "forced_token_ids": forced,
                    "max_new_tokens": self.config.max_token_length,
                }
            )
        prefix_elements = self._prefix_elements(num_elements, mask_tensor)
        prefix_length = 1 + prefix_elements * self.config.element_token_width
        input_ids = full[:, :prefix_length]
        remaining = max(0, self.config.max_token_length + 1 - input_ids.size(1))
        return BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "max_new_tokens": remaining,
            }
        )

    def post_process_layouts(
        self,
        sequences: torch.Tensor,
        *,
        output_type: OutputType = "dataclass",
        return_intermediates: bool = False,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Decode generated sequences to the common output schema."""
        decoded = self.tokenizer.decode_action_tokens(
            sequences.detach().cpu(),
            return_actions=return_intermediates,
        )
        intermediates = None
        if return_intermediates:
            intermediates = {"actions": decoded.get("actions")}
        output = LayoutGenerationOutput(
            bbox=cast(torch.Tensor, decoded["bbox"]),
            labels=cast(torch.Tensor, decoded["labels"]),
            mask=cast(torch.Tensor, decoded["mask"]),
            id2label=dict(cast(dict[int, str], self.config.id2label)),
            sequences=sequences.detach().cpu(),
            intermediates=intermediates,
        )
        if output_type == "dict":
            return dict(output)
        if output_type == "dataclass":
            return output
        raise ValueError(f"Unsupported output_type: {output_type}")

    def save_pretrained(
        self,
        save_directory: str | PathLike[str],
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> None:
        """Save processor and tokenizer metadata."""
        _ = (push_to_hub, kwargs)
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(out_dir)
        with (out_dir / PROCESSOR_CONFIG_FILE).open("w", encoding="utf-8") as f:
            json.dump({"processor_class": self.__class__.__name__}, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: object,
    ) -> "LayoutActionProcessor":
        """Load processor metadata from a checkpoint directory or Hub repo id."""
        tokenizer = LayoutActionTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )
        return cls(tokenizer=tokenizer)

    def _labels_to_ids(self, labels: object) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            return labels.long()
        label_array = np.asarray(labels, dtype=object)
        if all(not isinstance(label, str) for label in label_array.flatten()):
            return torch.as_tensor(labels, dtype=torch.long)
        label2id = cast(dict[str, int], self.config.label2id)
        vectorized = np.vectorize(lambda label: label2id[str(label)])
        return torch.as_tensor(vectorized(label_array), dtype=torch.long)

    def _prefix_elements(
        self, num_elements: int | list[int] | torch.Tensor | None, mask: torch.Tensor
    ) -> int:
        if num_elements is None:
            valid_counts = mask.sum(dim=1)
            return int(valid_counts.min().item())
        if isinstance(num_elements, int):
            return int(num_elements)
        tensor = torch.as_tensor(num_elements)
        return int(tensor.min().item())

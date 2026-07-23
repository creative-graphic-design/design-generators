"""Processor for LayoutDETR content-image conditions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from os import PathLike
from pathlib import Path
from typing import Literal, Self, cast

import torch
from transformers import ProcessorMixin
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import BatchEncoding

from laygen.modeling_outputs import LayoutGenerationOutput

from .configuration_layout_detr import (
    BackgroundPreprocessing,
    LayoutDetrConfig,
)
from .image_processing_layout_detr import LayoutDetrImageProcessor


class LayoutDetrProcessor(ProcessorMixin):
    """Normalize LayoutDETR image, text, label, and mask payloads."""

    attributes = ["image_processor"]
    image_processor_class = "LayoutDetrImageProcessor"
    tokenizer_class = "BertTokenizerFast"
    config_name = "processor_config.json"

    def __init__(
        self,
        *,
        image_processor: LayoutDetrImageProcessor | None = None,
        config: LayoutDetrConfig,
        id2label: Mapping[int | str, str] | None = None,
    ) -> None:
        """Initialize the processor."""
        self.config = config
        self.image_processor = image_processor or LayoutDetrImageProcessor.from_config(
            self.config
        )
        label_source = (
            id2label
            if id2label is not None
            else cast(dict[int, str], self.config.id2label)
        )
        self.id2label = {
            int(k): v for k, v in cast(Mapping[int | str, str], label_source).items()
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.chat_template = None

    def save_pretrained(
        self,
        save_directory: str | Path,
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> None:
        """Save processor metadata and image-processor config."""
        del push_to_hub, kwargs
        root = _processor_root(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        _write_processor_payload(root / self.config_name, self._metadata_payload())
        self.image_processor.save_pretrained(root)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        subfolder: str | None = None,
        **kwargs: object,
    ) -> Self:
        """Load processor metadata from a checkpoint directory."""
        del cache_dir, force_download, local_files_only, token, revision, kwargs
        root = _processor_root(pretrained_model_name_or_path, subfolder=subfolder)
        payload = _read_processor_payload(root / cls.config_name)
        config_payload = payload.get("config", {})
        if not isinstance(config_payload, dict):
            raise TypeError("processor config payload must be a dictionary")
        id2label_payload = payload.get("id2label")
        if id2label_payload is not None and not isinstance(id2label_payload, dict):
            raise TypeError("processor id2label payload must be a dictionary")
        config = LayoutDetrConfig.from_dict(config_payload)  # ty: ignore[invalid-argument-type]
        image_processor = LayoutDetrImageProcessor.from_pretrained(root)
        return cls(
            image_processor=image_processor,
            config=config,
            id2label=cast(Mapping[int | str, str] | None, id2label_payload),
        )

    def _metadata_payload(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "id2label": self.id2label,
            "processor_class": self.__class__.__name__,
        }

    def __call__(
        self,
        *,
        images: ImageInput | Sequence[ImageInput] | torch.Tensor | None = None,
        content: Mapping[str, object] | None = None,
        prompt: str | Sequence[str] | None = None,
        texts: Sequence[Sequence[str]] | Sequence[str] | None = None,
        labels: torch.Tensor
        | Sequence[Sequence[int | str]]
        | Sequence[int | str]
        | None = None,
        mask: torch.Tensor | Sequence[Sequence[bool]] | Sequence[bool] | None = None,
        condition_type: str = "content_image",
        background_preprocessing: BackgroundPreprocessing
        | str = BackgroundPreprocessing.none,
        batch_size: int = 1,
        return_tensors: Literal["pt"] = "pt",
        canvas_size: tuple[int, int] | None = None,
    ) -> BatchEncoding:
        """Encode public inputs for the LayoutDETR model."""
        if return_tensors != "pt":
            raise ValueError("LayoutDetrProcessor only supports return_tensors='pt'")
        if condition_type not in {"content_image", "content", "image", "visual"}:
            raise NotImplementedError(
                "LayoutDETR supports only condition_type='content_image'"
            )
        content = dict(content or {})
        resolved_images = images or content.get("image") or content.get("images")
        if resolved_images is None:
            raise ValueError("LayoutDETR requires images or content['image']")
        resolved_texts = texts if texts is not None else content.get("texts")
        if resolved_texts is None:
            if prompt is not None:
                raise ValueError(
                    "LayoutDETR requires per-element texts; prompt alone is not supported"
                )
            raise ValueError("LayoutDETR requires per-element texts")
        resolved_labels = labels if labels is not None else content.get("labels")
        if resolved_labels is None:
            raise ValueError("LayoutDETR requires per-element labels")

        text_rows = _normalize_text_rows(
            cast(Sequence[Sequence[str]] | Sequence[str], resolved_texts)
        )
        label_rows = self._normalize_label_rows(resolved_labels)
        if len(text_rows) != len(label_rows):
            raise ValueError("texts and labels must have the same batch size")
        if len(text_rows) == 1 and batch_size > 1:
            text_rows = text_rows * batch_size
            label_rows = label_rows * batch_size
        layout_mask = _normalize_mask_rows(mask, label_rows)
        image_features = self.image_processor.preprocess(
            resolved_images,
            background_preprocessing=background_preprocessing,
            canvas_size=canvas_size,
            return_tensors=return_tensors,
        )
        input_ids, text_attention_mask = self._tokenize_rows(text_rows)
        bbox_labels, padded_mask, padded_texts = self._pad_layout_rows(
            text_rows,
            label_rows,
            layout_mask,
        )
        text_lengths = _pad_text_lengths(text_rows, self.config.max_seq_length)
        image_batch = image_features["pixel_values"].shape[0]
        if image_batch == 1 and bbox_labels.shape[0] > 1:
            image_features["pixel_values"] = image_features["pixel_values"].expand(
                bbox_labels.shape[0], -1, -1, -1
            )
            image_features["canvas_size"] = image_features["canvas_size"].expand(
                bbox_labels.shape[0], -1
            )
        elif image_batch != bbox_labels.shape[0]:
            raise ValueError("image batch size must match texts/labels batch size")
        return BatchEncoding(
            {
                "pixel_values": image_features["pixel_values"],
                "canvas_size": image_features["canvas_size"],
                "input_ids": input_ids,
                "text_attention_mask": text_attention_mask,
                "bbox_labels": bbox_labels,
                "layout_mask": padded_mask,
                "texts": padded_texts,
                "text_lengths": text_lengths,
            }
        )

    def post_process_layouts(
        self,
        bbox: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        *,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        intermediates: dict[str, object] | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Return generated boxes in the shared layout output schema."""
        payload = LayoutGenerationOutput(
            bbox=bbox,
            labels=labels,
            mask=mask,
            id2label=self.id2label,
            intermediates=intermediates if return_intermediates else None,
        )
        if output_type == "dict":
            return dict(payload)
        if output_type != "dataclass":
            raise ValueError(f"Unsupported output_type: {output_type}")
        return payload

    def _normalize_label_rows(self, labels: object) -> list[list[int]]:
        if isinstance(labels, torch.Tensor):
            tensor = labels.detach().cpu().long()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            return [[int(value) for value in row] for row in tensor.tolist()]
        rows = _normalize_label_sequence(
            cast(Sequence[Sequence[int | str]] | Sequence[int | str], labels)
        )
        return [[self._label_to_id(label) for label in row] for row in rows]

    def _label_to_id(self, label: int | str) -> int:
        if isinstance(label, int):
            if label < 0 or label >= len(self.id2label):
                raise ValueError(f"Unknown Ad Banner label id: {label}")
            return label
        try:
            return self.label2id[label]
        except KeyError as exc:
            raise ValueError(f"Unknown Ad Banner label: {label}") from exc

    def _tokenize_rows(
        self,
        text_rows: list[list[str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_ids = []
        batch_mask = []
        for row in text_rows:
            ids_row = []
            mask_row = []
            for text in row[: self.config.max_seq_length]:
                token_ids = _hash_token_ids(
                    text, self.config.max_text_length, self.config.text_vocab_size
                )
                ids_row.append(token_ids)
                mask_row.append([token_id != 0 for token_id in token_ids])
            while len(ids_row) < self.config.max_seq_length:
                ids_row.append([0] * self.config.max_text_length)
                mask_row.append([False] * self.config.max_text_length)
            batch_ids.append(ids_row)
            batch_mask.append(mask_row)
        return (
            torch.tensor(batch_ids, dtype=torch.long),
            torch.tensor(batch_mask, dtype=torch.bool),
        )

    def _pad_layout_rows(
        self,
        text_rows: list[list[str]],
        label_rows: list[list[int]],
        mask_rows: list[list[bool]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[str]]]:
        labels = []
        masks = []
        texts = []
        max_len = self.config.max_seq_length
        for text_row, label_row, mask_row in zip(
            text_rows,
            label_rows,
            mask_rows,
            strict=True,
        ):
            if len(label_row) > max_len:
                raise ValueError(f"LayoutDETR supports at most {max_len} elements")
            pad = max_len - len(label_row)
            labels.append(label_row + [self.config.pad_label_id] * pad)
            masks.append(mask_row + [False] * pad)
            texts.append(text_row + [""] * pad)
        return (
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(masks, dtype=torch.bool),
            texts,
        )


def _hash_token_ids(text: str, max_length: int, vocab_size: int) -> list[int]:
    pieces = text.split()
    cls_id = min(101, max(1, vocab_size - 2))
    sep_id = min(102, max(1, vocab_size - 1))
    base = min(999, max(1, vocab_size // 2))
    ids = [cls_id]
    for piece in pieces[: max(0, max_length - 2)]:
        digest = hashlib.sha1(piece.lower().encode("utf-8")).hexdigest()
        ids.append(base + (int(digest[:8], 16) % max(1, vocab_size - base)))
    ids.append(sep_id if text else 0)
    ids = ids[:max_length]
    return ids + [0] * (max_length - len(ids))


def _pad_text_lengths(
    text_rows: list[list[str]],
    max_seq_length: int,
) -> torch.Tensor:
    rows = []
    for row in text_rows:
        lengths = [len(text) for text in row[:max_seq_length]]
        lengths.extend([0] * (max_seq_length - len(lengths)))
        rows.append(lengths)
    return torch.tensor(rows, dtype=torch.long)


def _processor_root(
    pretrained_model_name_or_path: str | PathLike[str],
    *,
    subfolder: str | None = None,
) -> Path:
    root = Path(pretrained_model_name_or_path)
    return root / subfolder if subfolder is not None else root


def _write_processor_payload(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_processor_payload(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _normalize_text_rows(
    texts: Sequence[Sequence[str]] | Sequence[str],
) -> list[list[str]]:
    if not texts:
        raise ValueError("texts must not be empty")
    first = texts[0]
    if isinstance(first, str):
        return [[str(text) for text in cast(Sequence[str], texts)]]
    return [[str(text) for text in row] for row in cast(Sequence[Sequence[str]], texts)]


def _normalize_label_sequence(
    labels: Sequence[Sequence[int | str]] | Sequence[int | str],
) -> list[list[int | str]]:
    if not labels:
        raise ValueError("labels must not be empty")
    first = labels[0]
    if isinstance(first, (int, str)):
        return [[label for label in cast(Sequence[int | str], labels)]]
    return [list(row) for row in cast(Sequence[Sequence[int | str]], labels)]


def _normalize_mask_rows(
    mask: torch.Tensor | Sequence[Sequence[bool]] | Sequence[bool] | None,
    label_rows: list[list[int]],
) -> list[list[bool]]:
    if mask is None:
        return [[True] * len(row) for row in label_rows]
    if isinstance(mask, torch.Tensor):
        tensor = mask.detach().cpu().bool()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return [[bool(value) for value in row] for row in tensor.tolist()]
    if not mask:
        raise ValueError("mask must not be empty")
    first = mask[0]
    if isinstance(first, bool):
        return [[bool(value) for value in cast(Sequence[bool], mask)]]
    return [
        [bool(value) for value in row] for row in cast(Sequence[Sequence[bool]], mask)
    ]

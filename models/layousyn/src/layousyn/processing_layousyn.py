"""Processor for LayouSyn text/concept-conditioned layout tensors."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from transformers import ProcessorMixin

from laygen.common.bbox import (
    BoxFormat,
    clamp_boxes,
    ltrb_to_xywh,
    normalize_boxes,
    normalize_box_format,
    xywh_to_ltrb,
)
from laygen.pipelines.pipeline_output import LayoutGenerationOutput

LAYOUSYN_CONCEPT_EMBEDS_KEY: Final[str] = "concept_embeds"
LAYOUSYN_CONCEPT_MASK_KEY: Final[str] = "concept_padding_mask"
LAYOUSYN_CAPTION_EMBEDS_KEY: Final[str] = "caption_embeds"
LAYOUSYN_CAPTION_MASK_KEY: Final[str] = "caption_padding_mask"
LAYOUSYN_ASPECT_RATIO_KEY: Final[str] = "aspect_ratio"
LAYOUSYN_LABEL_TEXTS_KEY: Final[str] = "label_texts"
LAYOUSYN_ID2LABEL_KEY: Final[str] = "id2label"
LAYOUSYN_PER_EXAMPLE_ID2LABEL_KEY: Final[str] = "id2label_per_example"


class LayouSynBatch(TypedDict):
    """Encoded LayouSyn processor batch."""

    concept_embeds: torch.Tensor
    concept_padding_mask: torch.Tensor
    caption_embeds: torch.Tensor
    caption_padding_mask: torch.Tensor
    aspect_ratio: torch.Tensor
    label_texts: list[list[str]]
    id2label: dict[int, str]
    id2label_per_example: list[dict[int, str]]


class LayouSynProcessor(ProcessorMixin):
    """Encode prompts and open-vocabulary concepts for LayouSyn.

    Args:
        layout_type: Reference layout type used by generated coordinates.
        max_in_len: Maximum number of concept slots.
        caption_model_name: Text encoder identifier used for captions.
        concept_model_name: Sentence-transformers model id for concept labels.
        id2label: Optional fixed vocabulary for integer labels.
        open_vocabulary: Whether string labels are accepted per request.
    """

    config_name = "processor_config.json"

    def __init__(
        self,
        *,
        layout_type: Literal["xyxy", "cxcywh"] = "xyxy",
        max_in_len: int = 60,
        max_y_len: int = 120,
        concept_in_channels: int = 768,
        y_in_channels: int = 768,
        caption_model_name: str = "t5-v1_1-base",
        concept_model_name: str = "sentence-transformers/sentence-t5-base",
        id2label: dict[int, str] | None = None,
        open_vocabulary: bool = True,
    ) -> None:
        """Initialize processor metadata."""
        super().__init__()
        self.layout_type = layout_type
        self.max_in_len = max_in_len
        self.max_y_len = max_y_len
        self.concept_in_channels = concept_in_channels
        self.y_in_channels = y_in_channels
        self.caption_model_name = caption_model_name
        self.concept_model_name = concept_model_name
        self.id2label = id2label
        self.open_vocabulary = open_vocabulary

    def to_dict(self) -> dict[str, object]:
        """Serialize processor metadata."""
        return {
            "layout_type": self.layout_type,
            "max_in_len": self.max_in_len,
            "max_y_len": self.max_y_len,
            "concept_in_channels": self.concept_in_channels,
            "y_in_channels": self.y_in_channels,
            "caption_model_name": self.caption_model_name,
            "concept_model_name": self.concept_model_name,
            "id2label": self.id2label,
            "open_vocabulary": self.open_vocabulary,
            "license": "cc-by-nc-4.0",
        }

    def save_pretrained(
        self,
        save_directory: str | Path,
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> tuple[str]:
        """Save processor metadata."""
        del kwargs
        if push_to_hub:
            raise ValueError("LayouSynProcessor does not push to Hub directly")
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        out = path / self.config_name
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return (str(out),)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        cache_dir: str | os.PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: object,
    ) -> "LayouSynProcessor":
        """Load processor metadata from a local directory."""
        del cache_dir, force_download, local_files_only, token, revision
        path = Path(pretrained_model_name_or_path) / cls.config_name
        data = json.loads(path.read_text())
        data.update(kwargs)
        data.pop("license", None)
        if data.get("id2label") is not None:
            data["id2label"] = {int(k): str(v) for k, v in data["id2label"].items()}
        return cls(**data)

    def __call__(
        self,
        *,
        prompt: str | Sequence[str] | None = None,
        labels: Sequence[str]
        | Sequence[Sequence[str]]
        | Int[torch.Tensor, "batch elements"]
        | Int[np.ndarray, "batch elements"]
        | None = None,
        id2label: dict[int, str] | None = None,
        bbox: Float[torch.Tensor, "batch elements 4"]
        | Float[np.ndarray, "batch elements 4"]
        | list[object]
        | None = None,
        mask: Bool[torch.Tensor, "batch elements"]
        | Bool[np.ndarray, "batch elements"]
        | list[object]
        | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        aspect_ratio: float | Sequence[float] | Float[torch.Tensor, "batch"] = 1.0,
        caption_embeds: Float[torch.Tensor, "batch tokens embedding_dim"] | None = None,
        caption_padding_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        concept_embeds: Float[torch.Tensor, "batch elements embedding_dim"]
        | None = None,
    ) -> LayouSynBatch:
        """Encode public text and concept inputs.

        Args:
            prompt: Caption text or batch of captions.
            labels: String concepts or integer labels.
            id2label: Mapping required for integer labels when no fixed
                processor mapping exists.
            bbox: Optional conditioning boxes for future init/refinement paths.
            mask: Optional valid-element mask.
            box_format: Public bbox format.
            normalized: Whether bbox coordinates are normalized.
            canvas_size: Required when ``normalized=False``.
            aspect_ratio: Scalar or per-example aspect ratio.
            caption_embeds: Precomputed caption embeddings.
            caption_padding_mask: Precomputed caption padding mask.
            concept_embeds: Precomputed concept embeddings.

        Returns:
            Encoded processor batch.

        Raises:
            ValueError: If required labels or embeddings are missing.
        """
        prompts = self._normalize_prompts(prompt)
        label_texts, union_id2label, per_example = self._normalize_labels(
            labels, id2label=id2label, batch_size=len(prompts)
        )
        batch_size = len(label_texts)
        if len(prompts) == 1 and batch_size > 1:
            prompts = prompts * batch_size
        if len(prompts) != batch_size:
            raise ValueError("prompt and labels batch sizes must match")
        concept_padding_mask = self._concept_padding_mask(label_texts, mask=mask)
        if concept_embeds is None:
            concept_embeds = self._encode_concepts(label_texts)
        concept_embeds = self._pad_concept_embeds(concept_embeds, batch_size)
        if caption_embeds is None or caption_padding_mask is None:
            caption_embeds, caption_padding_mask = self._encode_captions(prompts)
        caption_embeds = caption_embeds.float()
        caption_padding_mask = caption_padding_mask.bool()
        if bbox is not None:
            self._normalize_optional_bbox(
                bbox,
                box_format=box_format,
                normalized=normalized,
                canvas_size=canvas_size,
            )
        return LayouSynBatch(
            concept_embeds=concept_embeds.float(),
            concept_padding_mask=concept_padding_mask,
            caption_embeds=caption_embeds,
            caption_padding_mask=caption_padding_mask,
            aspect_ratio=self._aspect_ratio_tensor(aspect_ratio, batch_size),
            label_texts=label_texts,
            id2label=union_id2label,
            id2label_per_example=per_example,
        )

    def postprocess(
        self,
        sample: torch.Tensor,
        *,
        labels: list[list[str]],
        id2label: dict[int, str],
        id2label_per_example: list[dict[int, str]] | None = None,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        intermediates: object | None = None,
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor | object]:
        """Convert generated reference coordinates into the public schema."""
        sample = ((sample.clamp(-1, 1) + 1.0) / 2.0).float()
        if self.layout_type == "xyxy":
            left, top, right, bottom = sample.unbind(dim=-1)
            fixed = torch.stack(
                (
                    torch.minimum(left, right),
                    torch.minimum(top, bottom),
                    torch.maximum(left, right),
                    torch.maximum(top, bottom),
                ),
                dim=-1,
            )
            bbox = clamp_boxes(ltrb_to_xywh(fixed))
        else:
            bbox = clamp_boxes(sample)
        batch_size = sample.shape[0]
        label_ids = torch.zeros(batch_size, self.max_in_len, dtype=torch.long)
        mask = torch.zeros(batch_size, self.max_in_len, dtype=torch.bool)
        label2id = {text: idx for idx, text in id2label.items()}
        for batch_idx, batch_labels in enumerate(labels):
            for pos, text in enumerate(batch_labels[: self.max_in_len]):
                label_ids[batch_idx, pos] = label2id[text]
                mask[batch_idx, pos] = True
        payload = intermediates if return_intermediates else None
        if return_intermediates:
            payload = {
                "label_texts": labels,
                "id2label_per_example": id2label_per_example,
                "reference_layout_type": self.layout_type,
                "intermediates": intermediates,
            }
        output = LayoutGenerationOutput(
            bbox=bbox,
            labels=label_ids,
            mask=mask,
            id2label=id2label,
            intermediates=payload,
        )
        if output_type == "dict":
            return dict(output)
        if output_type != "dataclass":
            raise ValueError(f"Unsupported output_type: {output_type}")
        return output

    def _normalize_prompts(self, prompt: str | Sequence[str] | None) -> list[str]:
        if prompt is None:
            return [""]
        if isinstance(prompt, str):
            return [prompt]
        return [str(item) for item in prompt]

    def _normalize_labels(
        self,
        labels: Sequence[str]
        | Sequence[Sequence[str]]
        | torch.Tensor
        | np.ndarray
        | None,
        *,
        id2label: dict[int, str] | None,
        batch_size: int,
    ) -> tuple[list[list[str]], dict[int, str], list[dict[int, str]]]:
        if labels is None:
            raise ValueError("LayouSyn requires labels/concepts for object slots")
        mapping = id2label or self.id2label
        if isinstance(labels, torch.Tensor | np.ndarray):
            if mapping is None:
                raise ValueError("id2label is required when labels are integer ids")
            labels_t = torch.as_tensor(labels, dtype=torch.long)
            if labels_t.ndim == 1:
                labels_t = labels_t.unsqueeze(0)
            label_texts = [
                [mapping[int(idx)] for idx in row.tolist() if int(idx) in mapping]
                for row in labels_t
            ]
        else:
            label_texts = self._string_label_batches(labels, batch_size=batch_size)
        union: dict[str, int] = {}
        per_example: list[dict[int, str]] = []
        for batch_labels in label_texts:
            local: dict[int, str] = {}
            for text in batch_labels:
                if text not in union:
                    union[text] = len(union)
                if text not in local.values():
                    local[len(local)] = text
            per_example.append(local)
        return label_texts, {idx: text for text, idx in union.items()}, per_example

    def _string_label_batches(
        self,
        labels: Sequence[str] | Sequence[Sequence[str]],
        *,
        batch_size: int,
    ) -> list[list[str]]:
        if len(labels) == 0:
            return [[]]
        first = labels[0]
        if isinstance(first, str):
            return [[str(item) for item in labels]]
        return [[str(item) for item in row] for row in labels]

    def _concept_padding_mask(
        self,
        labels: list[list[str]],
        *,
        mask: torch.Tensor | np.ndarray | list[object] | None,
    ) -> torch.Tensor:
        if mask is not None:
            mask_t = torch.as_tensor(mask, dtype=torch.bool)
            if mask_t.ndim == 1:
                mask_t = mask_t.unsqueeze(0)
            valid = torch.zeros(mask_t.shape[0], self.max_in_len, dtype=torch.bool)
            valid[:, : min(mask_t.shape[1], self.max_in_len)] = mask_t[
                :, : self.max_in_len
            ]
            return ~valid
        padding = torch.ones(len(labels), self.max_in_len, dtype=torch.bool)
        for batch_idx, batch_labels in enumerate(labels):
            padding[batch_idx, : min(len(batch_labels), self.max_in_len)] = False
        return padding

    def _pad_concept_embeds(
        self, embeds: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        if embeds.ndim != 3:
            raise ValueError("concept_embeds must have shape (batch, seq, dim)")
        if embeds.shape[0] != batch_size:
            raise ValueError("concept_embeds batch size must match labels")
        if embeds.shape[1] > self.max_in_len:
            return embeds[:, : self.max_in_len]
        if embeds.shape[1] == self.max_in_len:
            return embeds
        pad = torch.zeros(
            batch_size,
            self.max_in_len - embeds.shape[1],
            embeds.shape[2],
            dtype=embeds.dtype,
            device=embeds.device,
        )
        return torch.cat((embeds, pad), dim=1)

    def _encode_concepts(self, labels: list[list[str]]) -> torch.Tensor:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError(
                "concept_embeds are required without sentence-transformers"
            ) from exc
        flat = [text for row in labels for text in row]
        if not flat:
            return torch.zeros(len(labels), 0, self.concept_in_channels)
        encoder = SentenceTransformer(self.concept_model_name)
        encoded = torch.as_tensor(encoder.encode(flat), dtype=torch.float32)
        rows = []
        offset = 0
        for row in labels:
            rows.append(encoded[offset : offset + len(row)])
            offset += len(row)
        return torch.nested.as_nested_tensor(rows).to_padded_tensor(0.0)

    def _encode_captions(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if any(prompts):
            raise ValueError(
                "caption_embeds are required for prompt-conditioned tests/offline use"
            )
        return (
            torch.zeros(len(prompts), self.max_y_len, self.y_in_channels),
            torch.ones(len(prompts), self.max_y_len, dtype=torch.bool),
        )

    def _aspect_ratio_tensor(
        self, aspect_ratio: float | Sequence[float] | torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        if isinstance(aspect_ratio, torch.Tensor):
            out = aspect_ratio.float()
        elif isinstance(aspect_ratio, float | int):
            out = torch.full((batch_size,), float(aspect_ratio))
        else:
            out = torch.tensor([float(item) for item in aspect_ratio])
        if out.numel() == 1 and batch_size > 1:
            out = out.repeat(batch_size)
        if out.shape != (batch_size,):
            raise ValueError("aspect_ratio must be scalar or match batch size")
        return out

    def _normalize_optional_bbox(
        self,
        bbox: torch.Tensor | np.ndarray | list[object],
        *,
        box_format: BoxFormat | str,
        normalized: bool,
        canvas_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        bbox_t = torch.as_tensor(bbox, dtype=torch.float32)
        if not normalized:
            if canvas_size is None:
                raise ValueError("canvas_size is required when normalized=False")
            bbox_t = normalize_boxes(
                bbox_t, canvas_size=canvas_size, box_format=box_format
            )
        elif normalize_box_format(box_format) is BoxFormat.ltrb:
            bbox_t = ltrb_to_xywh(bbox_t)
        if self.layout_type == "xyxy":
            return xywh_to_ltrb(bbox_t) * 2 - 1
        return bbox_t * 2 - 1

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from layout_generation_common.bbox import normalize_boxes

from .tokenization_layout_dm import LayoutDMTokenizer


class LayoutDMProcessor:
    config_name = "processor_config.json"

    def __init__(self, tokenizer: LayoutDMTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(
        self,
        *,
        bbox: torch.Tensor | np.ndarray | list[Any],
        labels: torch.Tensor | np.ndarray | list[Any],
        mask: torch.Tensor | np.ndarray | list[Any] | None = None,
        box_format: Literal["xywh", "ltwh", "ltrb"] = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        return_tensors: Literal["pt"] = "pt",
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("LayoutDMProcessor only supports return_tensors='pt'")
        bbox_t = torch.as_tensor(bbox, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.long)
        if labels_t.ndim == 1:
            labels_t = labels_t.unsqueeze(0)
            bbox_t = bbox_t.unsqueeze(0)
        if mask is None:
            mask_t = torch.ones(labels_t.shape, dtype=torch.bool)
        else:
            mask_t = torch.as_tensor(mask, dtype=torch.bool)
            if mask_t.ndim == 1:
                mask_t = mask_t.unsqueeze(0)
        if not normalized:
            if canvas_size is None:
                raise ValueError("canvas_size is required when normalized=False")
            bbox_t = normalize_boxes(
                bbox_t, canvas_size=canvas_size, box_format=box_format
            )
        elif box_format == "ltwh":
            from layout_generation_common.bbox import ltwh_to_xywh

            bbox_t = ltwh_to_xywh(bbox_t)
        elif box_format == "ltrb":
            from layout_generation_common.bbox import ltrb_to_xywh

            bbox_t = ltrb_to_xywh(bbox_t)
        return self.tokenizer.encode_layout(bbox=bbox_t, labels=labels_t, mask=mask_t)

    def save_pretrained(self, save_directory: str) -> None:
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, path: str) -> "LayoutDMProcessor":
        return cls(LayoutDMTokenizer.from_pretrained(path))

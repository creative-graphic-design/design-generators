"""Input processor for LayoutDM structured layout tensors."""

from __future__ import annotations

from os import PathLike
from typing import Literal

import numpy as np
import torch
from transformers import ProcessorMixin

from laygen.common.bbox import (
    BoxFormat,
    prepare_layout_tensors,
)

from .tokenization_layout_dm import LayoutDMTokenizer


class LayoutDMProcessor(ProcessorMixin):
    """Normalize layout arrays and encode them with ``LayoutDMTokenizer``.

    Args:
        tokenizer: Tokenizer used to encode processed layouts.

    Examples:
        >>> from layout_dm.configuration_layout_dm import LayoutDMConfig
        >>> from layout_dm.tokenization_layout_dm import LayoutDMTokenizer
        >>> processor = LayoutDMProcessor(LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet")))
        >>> sorted(processor(bbox=[[[0.5, 0.5, 0.2, 0.2]]], labels=[[0]]))
        ['attention_mask', 'input_ids', 'mask']
    """

    config_name = "processor_config.json"
    tokenizer_class = "LayoutDMTokenizer"

    def __init__(self, tokenizer: LayoutDMTokenizer) -> None:
        """Initialize the processor with a tokenizer."""
        super().__init__(tokenizer=tokenizer)

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
    ) -> "LayoutDMProcessor":
        """Load a processor with the LayoutDM tokenizer implementation."""
        tokenizer = LayoutDMTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )
        return cls(tokenizer=tokenizer)

    def __call__(
        self,
        *,
        bbox: torch.Tensor | np.ndarray | list[object],
        labels: torch.Tensor | np.ndarray | list[object],
        mask: torch.Tensor | np.ndarray | list[object] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        return_tensors: Literal["pt"] = "pt",
    ) -> dict[str, torch.Tensor]:
        """Process a layout batch into model input tensors.

        Args:
            bbox: Layout boxes in ``box_format``.
            labels: Integer labels matching the layout boxes.
            mask: Optional valid-element mask. All elements are valid when omitted.
            box_format: Input box format.
            normalized: Whether boxes are already normalized to ``[0, 1]``.
            canvas_size: Pixel canvas size required when ``normalized=False``.
            return_tensors: Tensor backend. Only ``"pt"`` is supported.

        Returns:
            Tokenizer output containing ``input_ids``, ``attention_mask``, and
            ``mask`` tensors.

        Raises:
            ValueError: If ``return_tensors`` is not ``"pt"`` or if
                ``canvas_size`` is missing for pixel-space boxes.
        """
        if return_tensors != "pt":
            raise ValueError("LayoutDMProcessor only supports return_tensors='pt'")
        bbox_t, labels_t, mask_t = prepare_layout_tensors(
            bbox=bbox,
            labels=labels,
            mask=mask,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )
        return self.tokenizer.encode_layout(bbox=bbox_t, labels=labels_t, mask=mask_t)

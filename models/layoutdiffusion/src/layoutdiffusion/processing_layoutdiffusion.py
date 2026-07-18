"""Input processor for LayoutDiffusion pipelines."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from transformers import ProcessorMixin

from laygen.common.bbox import BoxFormat

from .tokenization_layoutdiffusion import LayoutDiffusionTokenizer


class LayoutDiffusionProcessor(ProcessorMixin):
    """Normalize public layout inputs and delegate tokenization.

    Args:
        tokenizer: LayoutDiffusion tokenizer.

    Examples:
        >>> from layoutdiffusion import LayoutDiffusionConfig, LayoutDiffusionTokenizer
        >>> cfg = LayoutDiffusionConfig(dataset_name="publaynet")
        >>> proc = LayoutDiffusionProcessor(LayoutDiffusionTokenizer(cfg))
        >>> proc.num_elements_to_tensor(2, batch_size=1).tolist()
        [2]
    """

    attributes = ["tokenizer"]
    tokenizer_class = "LayoutDiffusionTokenizer"

    def __init__(self, tokenizer: LayoutDiffusionTokenizer) -> None:
        """Initialize the processor."""
        self.tokenizer = tokenizer

    def __call__(
        self,
        *,
        bbox: torch.Tensor | np.ndarray | list[object] | None = None,
        labels: torch.Tensor | np.ndarray | list[object] | None = None,
        mask: torch.Tensor | np.ndarray | list[object] | None = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        return_tensors: Literal["pt"] = "pt",
    ) -> dict[str, torch.Tensor]:
        """Process layout tensors for conditional generation.

        Args:
            bbox: Optional layout boxes.
            labels: Optional labels.
            mask: Optional valid-element mask.
            num_elements: Optional element counts.
            box_format: Format of ``bbox``.
            normalized: Whether ``bbox`` is normalized.
            canvas_size: Pixel canvas size.
            return_tensors: Only ``"pt"`` is supported.

        Returns:
            Tokenizer output or element-count tensor.

        Raises:
            ValueError: If required conditional tensors are missing.
        """
        if return_tensors != "pt":
            raise ValueError(
                "LayoutDiffusionProcessor only supports return_tensors='pt'"
            )
        if bbox is None or labels is None:
            batch_size = 1
            if isinstance(num_elements, list):
                batch_size = len(num_elements)
            counts = self.num_elements_to_tensor(num_elements, batch_size=batch_size)
            return {} if counts is None else {"num_elements": counts}
        return self.tokenizer(
            bbox=torch.as_tensor(bbox),
            labels=torch.as_tensor(labels),
            mask=None if mask is None else torch.as_tensor(mask),
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )

    def num_elements_to_tensor(
        self,
        num_elements: int | list[int] | torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor | None:
        """Convert public element counts to a tensor."""
        if num_elements is None:
            return None
        counts = torch.as_tensor(num_elements, dtype=torch.long)
        if counts.ndim == 0:
            counts = counts.expand(batch_size)
        return counts

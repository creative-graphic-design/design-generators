from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from laygen.common.bbox import normalize_boxes

from .tokenization_layout_dm import LayoutDMTokenizer


class LayoutDMProcessor:
    """Prepare structured layout arrays for `LayoutDMTokenizer`.

    Args:
        tokenizer: Tokenizer used to encode normalized layout tensors.

    Returns:
        A callable processor compatible with saved Diffusers pipeline directories.

    Raises:
        ValueError: If called with unsupported tensor format options.

    Examples:
        >>> import torch
        >>> from layout_dm.configuration_layout_dm import LayoutDMConfig
        >>> from layout_dm.tokenization_layout_dm import LayoutDMTokenizer
        >>> processor = LayoutDMProcessor(LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet")))
        >>> processor(bbox=torch.zeros(1, 1, 4), labels=torch.zeros(1, 1, dtype=torch.long))["input_ids"].shape
        torch.Size([1, 125])
    """

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
        """Encode labels and boxes as LayoutDM token tensors.

        Args:
            bbox: Layout boxes as a tensor, NumPy array, or nested list.
            labels: Class ids as a tensor, NumPy array, or nested list.
            mask: Optional element mask.
            box_format: Coordinate format for `bbox`.
            normalized: Whether input boxes are already normalized.
            canvas_size: Pixel canvas used when `normalized=False`.
            return_tensors: Output tensor framework. Only `"pt"` is supported.

        Returns:
            Dictionary with `input_ids`, `attention_mask`, and `mask`.

        Raises:
            ValueError: If `return_tensors` is not `"pt"` or pixel boxes lack
                `canvas_size`.

        Examples:
            >>> import torch
            >>> from layout_dm.configuration_layout_dm import LayoutDMConfig
            >>> from layout_dm.tokenization_layout_dm import LayoutDMTokenizer
            >>> processor = LayoutDMProcessor(LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet")))
            >>> encoded = processor(bbox=torch.zeros(1, 1, 4), labels=torch.zeros(1, 1, dtype=torch.long))
            >>> sorted(encoded)
            ['attention_mask', 'input_ids', 'mask']
        """

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
            from laygen.common.bbox import ltwh_to_xywh

            bbox_t = ltwh_to_xywh(bbox_t)
        elif box_format == "ltrb":
            from laygen.common.bbox import ltrb_to_xywh

            bbox_t = ltrb_to_xywh(bbox_t)
        return self.tokenizer.encode_layout(bbox=bbox_t, labels=labels_t, mask=mask_t)

    def save_pretrained(self, save_directory: str) -> None:
        """Save processor files beside the tokenizer files.

        Args:
            save_directory: Directory where tokenizer files are written.

        Returns:
            None.

        Raises:
            OSError: If files cannot be written.

        Examples:
            >>> import tempfile
            >>> from layout_dm.configuration_layout_dm import LayoutDMConfig
            >>> from layout_dm.tokenization_layout_dm import LayoutDMTokenizer
            >>> with tempfile.TemporaryDirectory() as path:
            ...     LayoutDMProcessor(LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet"))).save_pretrained(path)
        """

        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, path: str) -> "LayoutDMProcessor":
        """Load a processor from a saved tokenizer directory.

        Args:
            path: Directory containing LayoutDM tokenizer files.

        Returns:
            Loaded processor.

        Raises:
            OSError: If required tokenizer files are missing.

        Examples:
            >>> LayoutDMProcessor.from_pretrained  # doctest: +ELLIPSIS
            <bound method...
        """

        return cls(LayoutDMTokenizer.from_pretrained(path))

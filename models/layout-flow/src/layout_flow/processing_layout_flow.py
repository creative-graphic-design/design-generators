"""Input and output processing for LayoutFlow pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, assert_never

import numpy as np
import torch

from laygen.common.bbox import (
    BoxFormat,
    clamp_boxes,
    denormalize_boxes,
    ltrb_to_xywh,
    ltwh_to_xywh,
    normalize_boxes,
)
from laygen.common.conditions import ConditionType, normalize_condition_type

from .configuration_layout_flow import LayoutFlowConfig

TensorInput: TypeAlias = torch.Tensor | np.ndarray | Sequence[object] | None


class LayoutFlowProcessor:
    """Prepare public layout tensors for the LayoutFlow model."""

    def __init__(self, config: LayoutFlowConfig) -> None:
        """Create a processor for a LayoutFlow configuration.

        Args:
            config: LayoutFlow pipeline configuration.
        """
        self.config = config
        bit_mask = [1 << k for k in range(config.attr_dim)]
        self.bit_mask = torch.tensor(bit_mask, dtype=torch.long)

    def __call__(
        self,
        *,
        bbox: TensorInput = None,
        labels: TensorInput = None,
        mask: TensorInput = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        batch_size: int = 1,
        box_format: BoxFormat | str = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convert public inputs into padded model-ready tensors.

        Args:
            bbox: Optional boxes in the requested ``box_format``.
            labels: Optional dataset-local label ids.
            mask: Optional valid-element mask.
            num_elements: Optional element counts used when ``mask`` is omitted.
            batch_size: Batch size used when tensors are omitted.
            box_format: Format of ``bbox``.
            normalized: Whether ``bbox`` is already normalized to ``[0, 1]``.
            canvas_size: Pixel canvas size required for denormalized boxes.
            device: Target torch device.

        Returns:
            Dictionary with ``bbox``, ``labels``, ``mask``, and ``length`` tensors.

        Raises:
            ValueError: If denormalized boxes are missing ``canvas_size`` or the
                box format is unsupported.
        """
        device = torch.device(device) if device is not None else torch.device("cpu")
        max_length = self.config.max_length
        if bbox is None:
            bbox_t = torch.zeros(
                batch_size, max_length, 4, dtype=torch.float32, device=device
            )
        else:
            bbox_t = torch.as_tensor(bbox, dtype=torch.float32, device=device)
            if bbox_t.ndim == 2:
                bbox_t = bbox_t.unsqueeze(0)
            batch_size = bbox_t.shape[0]
            if not normalized:
                if canvas_size is None:
                    raise ValueError("canvas_size is required when normalized=False")
                bbox_t = normalize_boxes(
                    bbox_t, canvas_size=canvas_size, box_format=box_format
                )
            else:
                fmt = BoxFormat(box_format)
                if fmt is BoxFormat.ltwh:
                    bbox_t = ltwh_to_xywh(bbox_t)
                elif fmt is BoxFormat.ltrb:
                    bbox_t = ltrb_to_xywh(bbox_t)
                elif fmt is not BoxFormat.xywh:
                    assert_never(fmt)
            bbox_t = self._pad_tensor(bbox_t, max_length, 0.0)
        if labels is None:
            labels_t = torch.zeros(
                batch_size, max_length, dtype=torch.long, device=device
            )
        else:
            labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
            if labels_t.ndim == 1:
                labels_t = labels_t.unsqueeze(0)
            labels_t = self._pad_tensor(labels_t, max_length, 0)
        if mask is None:
            lengths = self._num_elements_to_lengths(
                num_elements, batch_size, max_length, device
            )
            mask_t = torch.arange(max_length, device=device)[None, :] < lengths[:, None]
        else:
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
            if mask_t.ndim == 1:
                mask_t = mask_t.unsqueeze(0)
            mask_t = self._pad_tensor(mask_t, max_length, False)
            lengths = mask_t.sum(dim=1).long()
        bbox_t = bbox_t * mask_t.unsqueeze(-1)
        labels_t = labels_t * mask_t.long()
        return {"bbox": bbox_t, "labels": labels_t, "mask": mask_t, "length": lengths}

    def encode_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Encode integer labels as vendor analog-bit vectors."""
        bit_mask = self.bit_mask.to(labels.device)
        return (
            torch.bitwise_and(labels.unsqueeze(-1), bit_mask).float() / bit_mask.float()
        )

    def decode_labels(self, bits: torch.Tensor) -> torch.Tensor:
        """Decode vendor analog-bit vectors into integer labels."""
        bit_mask = self.bit_mask.to(bits.device)
        active = (bits - 0.5 >= 0).long()
        return (
            (active * bit_mask).sum(dim=-1).clamp(0, self.config.num_labels - 1).long()
        )

    def model_state(self, bbox: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Concatenate normalized boxes and analog-bit labels."""
        return torch.cat([bbox, self.encode_labels(labels)], dim=-1)

    def preprocess_state(
        self, state: torch.Tensor, *, reverse: bool = False
    ) -> torch.Tensor:
        """Map between public ``[0, 1]`` state and model distribution range."""
        if self.config.distribution in {"gaussian", "gmm", "uniform", "gauss_uniform"}:
            return (state + 1) / 2 if reverse else 2 * state - 1
        return state

    def make_condition_mask(
        self,
        condition_type: ConditionType | str,
        *,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Create the vendor condition mask for a conditioning mode.

        Args:
            condition_type: Canonical condition or alias.
            mask: Valid-element mask.
            generator: Optional generator used by completion masking.

        Returns:
            Long tensor where ``1`` means generated and ``0`` means conditioned.

        Raises:
            ValueError: If the condition type is unsupported.
        """
        canonical = normalize_condition_type(condition_type)
        batch, seq = mask.shape
        cond_mask = torch.ones(
            batch,
            seq,
            self.config.sample_dim,
            dtype=torch.long,
            device=mask.device,
        )
        if canonical in {ConditionType.label, ConditionType.refinement}:
            cond_mask[:, :, 4:] = 0
        elif canonical is ConditionType.label_size:
            cond_mask[:, :, 2:] = 0
        elif canonical is ConditionType.completion:
            cond_mask = self._completion_mask(cond_mask, mask, generator)
        elif canonical is ConditionType.unconditional:
            pass
        else:
            raise ValueError(f"Unsupported LayoutFlow condition_type: {canonical}")
        return cond_mask

    def postprocess(
        self,
        state: torch.Tensor,
        *,
        mask: torch.Tensor,
        box_format: BoxFormat | str = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convert model state back to public layout tensors.

        Args:
            state: Model state tensor.
            mask: Valid-element mask.
            box_format: Requested output box format.
            normalized: Whether to return normalized coordinates.
            canvas_size: Pixel canvas size for denormalized coordinates.

        Returns:
            Dictionary with ``bbox``, ``labels``, and ``mask``.

        Raises:
            ValueError: If denormalized output is requested without
                ``canvas_size``.
        """
        restored = self.preprocess_state(state, reverse=True)
        bbox = clamp_boxes(restored[:, :, :4]) * mask.unsqueeze(-1)
        labels = self.decode_labels(restored[:, :, 4:]) * mask.long()
        fmt = BoxFormat(box_format)
        if fmt is not BoxFormat.xywh:
            if not normalized and canvas_size is None:
                raise ValueError("canvas_size is required for denormalized output")
            if canvas_size is None:
                canvas_size = (1, 1)
            bbox = denormalize_boxes(bbox, canvas_size=canvas_size, box_format=fmt)
            if normalized:
                scale = torch.tensor(
                    (*canvas_size, *canvas_size), dtype=bbox.dtype, device=bbox.device
                )
                bbox = bbox / scale
        elif not normalized:
            if canvas_size is None:
                raise ValueError("canvas_size is required for denormalized output")
            bbox = denormalize_boxes(
                bbox, canvas_size=canvas_size, box_format=BoxFormat.xywh
            )
        return {"bbox": bbox, "labels": labels, "mask": mask}

    def _completion_mask(
        self,
        cond_mask: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        for i, length in enumerate(mask.sum(dim=1).tolist()):
            if length <= 1:
                continue
            keep = max(1, int(length * 0.2))
            scores = torch.rand(length, device=mask.device, generator=generator)
            idx = scores.topk(keep).indices
            cond_mask[i, idx] = 0
        return cond_mask

    @staticmethod
    def _pad_tensor(
        tensor: torch.Tensor, max_length: int, value: float | int | bool
    ) -> torch.Tensor:
        if tensor.shape[1] > max_length:
            return tensor[:, :max_length]
        if tensor.shape[1] == max_length:
            return tensor
        pad_shape = (tensor.shape[0], max_length - tensor.shape[1], *tensor.shape[2:])
        pad = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=1)

    @staticmethod
    def _num_elements_to_lengths(
        num_elements: int | list[int] | torch.Tensor | None,
        batch_size: int,
        max_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_elements is None:
            return torch.full(
                (batch_size,), max_length, dtype=torch.long, device=device
            )
        lengths = torch.as_tensor(num_elements, dtype=torch.long, device=device)
        if lengths.ndim == 0:
            lengths = lengths.repeat(batch_size)
        return lengths.clamp(0, max_length)

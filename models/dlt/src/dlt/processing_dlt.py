"""Processor for DLT public layouts and internal tensors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from transformers import ProcessorMixin

from laygen.common.bbox import BoxFormat, prepare_layout_tensors
from laygen.common.labels import DatasetName

from .configuration_dlt import default_id2label, normalize_dataset

TensorInput: TypeAlias = torch.Tensor | np.ndarray | Sequence[object] | None


class DLTProcessor(ProcessorMixin):
    """Encode DLT public inputs into the original tensor format.

    Args:
        dataset: Canonical dataset name.
        labels: Ordered public labels without internal pad/drop ids.
        max_num_comp: Maximum number of layout elements.
    """

    config_name = "processor_config.json"

    def __init__(
        self,
        dataset: DatasetName | str,
        labels: Sequence[str],
        max_num_comp: int,
    ) -> None:
        """Initialize processor metadata."""
        super().__init__()
        self.dataset = str(normalize_dataset(dataset))
        self.labels = tuple(str(label) for label in labels)
        self.max_num_comp = max_num_comp

    @classmethod
    def from_dataset(cls, dataset: DatasetName | str) -> "DLTProcessor":
        """Create a processor from shared dataset metadata."""
        canonical = normalize_dataset(dataset)
        return cls(
            dataset=canonical,
            labels=tuple(default_id2label(canonical).values()),
            max_num_comp=9
            if canonical in {DatasetName.publaynet, DatasetName.rico13}
            else 33,
        )

    @property
    def id2label(self) -> dict[int, str]:
        """Return public label names keyed by dataset-local ids."""
        return dict(enumerate(self.labels))

    @property
    def categories_num(self) -> int:
        """Return internal category count including pad and mask/drop ids."""
        return len(self.labels) + 2

    @property
    def pad_category_id(self) -> int:
        """Return the internal padding category id."""
        return 0

    @property
    def mask_category_id(self) -> int:
        """Return the internal mask/drop category id."""
        return self.categories_num - 1

    def __call__(
        self,
        *,
        bbox: Float[torch.Tensor, "batch elements 4"]
        | Float[np.ndarray, "batch elements 4"]
        | Sequence[object],
        labels: Int[torch.Tensor, "batch elements"]
        | Int[np.ndarray, "batch elements"]
        | Sequence[object],
        mask: Bool[torch.Tensor, "batch elements"]
        | Bool[np.ndarray, "batch elements"]
        | Sequence[object]
        | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convert public layout tensors into padded internal tensors."""
        bbox_t, labels_t, mask_t = prepare_layout_tensors(
            bbox=bbox,
            labels=labels,
            mask=mask,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )
        if device is not None:
            bbox_t = bbox_t.to(device)
            labels_t = labels_t.to(device)
            mask_t = mask_t.to(device)
        bbox_t, labels_t, mask_t = self.pad(bbox_t, labels_t, mask_t)
        return {
            "box": self.public_to_internal_boxes(bbox_t) * mask_t.unsqueeze(-1),
            "box_cond": self.public_to_internal_boxes(bbox_t) * mask_t.unsqueeze(-1),
            "cat": self.public_to_internal_labels(labels_t, mask_t),
            "mask": mask_t,
        }

    def empty_condition(
        self,
        *,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Return an empty unconditional internal batch."""
        device = torch.device(device)
        bbox = torch.zeros(batch_size, self.max_num_comp, 4, dtype=dtype, device=device)
        labels = torch.zeros(
            batch_size, self.max_num_comp, dtype=torch.long, device=device
        )
        mask = torch.ones(
            batch_size, self.max_num_comp, dtype=torch.bool, device=device
        )
        return {
            "box": self.public_to_internal_boxes(bbox),
            "box_cond": self.public_to_internal_boxes(bbox),
            "cat": self.public_to_internal_labels(labels, mask),
            "mask": mask,
        }

    def pad(
        self,
        bbox: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad a layout batch to ``max_num_comp``."""
        if bbox.shape[1] > self.max_num_comp:
            raise ValueError(f"DLT supports at most {self.max_num_comp} elements")
        if mask is None:
            mask = torch.ones(labels.shape, dtype=torch.bool, device=labels.device)
        pad_count = self.max_num_comp - bbox.shape[1]
        if pad_count:
            bbox = torch.cat(
                [
                    bbox,
                    torch.zeros(
                        bbox.shape[0],
                        pad_count,
                        4,
                        dtype=bbox.dtype,
                        device=bbox.device,
                    ),
                ],
                dim=1,
            )
            labels = torch.cat(
                [
                    labels,
                    torch.zeros(
                        labels.shape[0],
                        pad_count,
                        dtype=labels.dtype,
                        device=labels.device,
                    ),
                ],
                dim=1,
            )
            mask = torch.cat(
                [
                    mask,
                    torch.zeros(
                        mask.shape[0], pad_count, dtype=torch.bool, device=mask.device
                    ),
                ],
                dim=1,
            )
        return bbox, labels, mask

    def public_to_internal_boxes(self, bbox: torch.Tensor) -> torch.Tensor:
        """Map public normalized ``xywh`` boxes to DLT's internal range."""
        return bbox.clamp(0.0, 1.0) * 4.0 - 2.0

    def internal_to_public_boxes(self, bbox: torch.Tensor) -> torch.Tensor:
        """Map DLT internal-range boxes to public normalized ``xywh``."""
        return (bbox / 2.0 + 1.0).div(2.0).clamp(0.0, 1.0)

    def public_to_internal_labels(
        self, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Shift public labels into DLT's internal category ids."""
        shifted = labels.long() + 1
        return torch.where(mask.bool(), shifted, torch.zeros_like(shifted))

    def internal_to_public_labels(
        self, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Shift DLT internal category ids back to public dataset-local ids."""
        public = (labels.long() - 1).clamp(0, len(self.labels) - 1)
        return public * mask.long()

    def condition_masks(
        self, condition_type: str, *, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return DLT ``mask_box`` and ``mask_cat`` tensors.

        ``1`` means generated/noised and ``0`` means conditioned, matching the
        original implementation.
        """
        mask_box = torch.ones(
            mask.shape[0], mask.shape[1], 4, dtype=torch.long, device=mask.device
        )
        mask_cat = torch.ones(mask.shape, dtype=torch.long, device=mask.device)
        if condition_type == "label":
            mask_cat.zero_()
        elif condition_type == "label_size":
            mask_box[:, :, 2:] = 0
            mask_cat.zero_()
        elif condition_type == "unconditional":
            pass
        else:
            raise ValueError(f"Unsupported DLT condition_type: {condition_type}")
        mask_box = mask_box * mask.unsqueeze(-1).long()
        mask_cat = mask_cat * mask.long()
        return mask_box, mask_cat

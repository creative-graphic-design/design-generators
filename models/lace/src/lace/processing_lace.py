"""Processor for encoding and decoding LACE layout tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import torch
from transformers import ProcessorMixin

from laygen.common.bbox import (
    BoxFormat,
    prepare_layout_tensors,
)
from laygen.common.labels import DatasetName
from laygen.pipelines.pipeline_output import LayoutGenerationOutput

from .configuration_lace import get_dataset_spec, normalize_dataset

LACE_LAYOUT_KEY: Final[str] = "layout"
LACE_BBOX_KEY: Final[str] = "bbox"
LACE_LABELS_KEY: Final[str] = "labels"
LACE_MASK_KEY: Final[str] = "mask"


class LaceProcessor(ProcessorMixin):
    """Encode public layout tensors into the continuous LACE sequence format.

    Args:
        dataset: Canonical dataset name serialized with the processor config.
        labels: Ordered category labels without the padding label.
        max_seq_length: Maximum number of layout elements.

    Examples:
        >>> processor = LaceProcessor.from_dataset("publaynet")
        >>> processor.seq_dim
        10
    """

    config_name = "processor_config.json"

    def __init__(
        self,
        dataset: DatasetName | str,
        labels: list[str],
        max_seq_length: int = 25,
    ) -> None:
        """Initialize processor metadata.

        Args:
            dataset: Canonical dataset name.
            labels: Ordered category labels without padding.
            max_seq_length: Maximum number of layout elements.
        """
        super().__init__()
        self.dataset = str(normalize_dataset(dataset))
        self.labels = tuple(labels)
        self.max_seq_length = max_seq_length

    @classmethod
    def from_dataset(cls, dataset: DatasetName | str) -> "LaceProcessor":
        """Create a processor from built-in dataset metadata.

        Args:
            dataset: LACE dataset name or alias.

        Returns:
            Processor configured for the dataset.

        Raises:
            ValueError: If the dataset is unsupported.

        Examples:
            >>> LaceProcessor.from_dataset("rico13").pad_label_id
            13
        """
        spec = get_dataset_spec(dataset)
        return cls(
            dataset=str(spec.dataset),
            labels=[str(label) for label in spec.labels],
            max_seq_length=spec.max_seq_length,
        )

    @property
    def id2label(self) -> dict[int, str]:
        """Return the category id to label mapping."""
        return dict(enumerate(self.labels))

    @property
    def pad_label_id(self) -> int:
        """Return the padding label id."""
        return len(self.labels)

    @property
    def num_classes_with_pad(self) -> int:
        """Return the label-channel count including padding."""
        return len(self.labels) + 1

    @property
    def seq_dim(self) -> int:
        """Return the latent per-element feature size."""
        return self.num_classes_with_pad + 4

    def __call__(
        self,
        *,
        bbox: torch.Tensor | np.ndarray | list[object],
        labels: torch.Tensor | np.ndarray | list[object],
        mask: torch.Tensor | np.ndarray | list[object] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a public layout batch.

        Args:
            bbox: Boxes in ``box_format``.
            labels: Integer category labels.
            mask: Optional valid-element mask.
            box_format: Input box format.
            normalized: Whether input coordinates are already normalized.
            canvas_size: Pixel canvas size required when ``normalized`` is false.

        Returns:
            Dictionary containing encoded layout, normalized boxes, labels, and mask.

        Raises:
            ValueError: If pixel boxes are passed without ``canvas_size`` or if
                ``box_format`` is unsupported.

        Examples:
            >>> processor = LaceProcessor.from_dataset("publaynet")
            >>> out = processor(bbox=[[[0.5, 0.5, 0.2, 0.2]]], labels=[[0]])
            >>> tuple(out["layout"].shape)
            (1, 25, 10)
        """
        bbox_t, labels_t, mask_t = prepare_layout_tensors(
            bbox=bbox,
            labels=labels,
            mask=mask,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )
        bbox_t, labels_t, mask_t = self.pad(bbox_t, labels_t, mask_t)
        return {
            LACE_LAYOUT_KEY: self.encode(bbox_t, labels_t, mask_t),
            LACE_BBOX_KEY: bbox_t,
            LACE_LABELS_KEY: labels_t,
            LACE_MASK_KEY: mask_t,
        }

    def pad(
        self,
        bbox: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
        max_seq_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad a batch to ``max_seq_length``.

        Args:
            bbox: Normalized boxes with shape ``(batch, seq, 4)``.
            labels: Integer labels with shape ``(batch, seq)``.
            mask: Optional valid-element mask.
            max_seq_length: Optional override for output length.

        Returns:
            Padded boxes, labels, and mask.

        Raises:
            ValueError: If the input has too many elements.
        """
        max_len = max_seq_length or self.max_seq_length
        if bbox.shape[1] > max_len:
            raise ValueError(f"LACE supports at most {max_len} elements")
        if mask is None:
            mask = torch.ones(labels.shape, dtype=torch.bool, device=labels.device)
        pad_count = max_len - bbox.shape[1]
        if pad_count:
            bbox_pad = torch.zeros(
                bbox.shape[0], pad_count, 4, dtype=bbox.dtype, device=bbox.device
            )
            label_pad = torch.full(
                (labels.shape[0], pad_count),
                self.pad_label_id,
                dtype=labels.dtype,
                device=labels.device,
            )
            mask_pad = torch.zeros(
                mask.shape[0], pad_count, dtype=torch.bool, device=mask.device
            )
            bbox = torch.cat((bbox, bbox_pad), dim=1)
            labels = torch.cat((labels, label_pad), dim=1)
            mask = torch.cat((mask, mask_pad), dim=1)
        labels = labels.clone()
        labels[~mask] = self.pad_label_id
        return bbox, labels, mask

    def encode(
        self, bbox: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode normalized boxes and labels into the vendor latent range.

        Args:
            bbox: Normalized center ``xywh`` boxes.
            labels: Integer labels.
            mask: Optional valid-element mask.

        Returns:
            Tensor with one-hot labels followed by box channels in ``[-1, 1]``.
        """
        bbox, labels, mask = self.pad(bbox, labels, mask)
        bbox_in = 2 * (bbox.clamp(0.0, 1.0) - 0.5)
        labels = labels.clamp(0, self.pad_label_id)
        labels[~mask] = self.pad_label_id
        one_hot = torch.nn.functional.one_hot(
            labels, num_classes=self.num_classes_with_pad
        ).to(dtype=bbox.dtype, device=bbox.device)
        return torch.cat((one_hot, bbox_in), dim=-1)

    def decode(
        self, layout: torch.Tensor, clamp: bool = True
    ) -> LayoutGenerationOutput:
        """Decode a LACE layout tensor into public output fields.

        Args:
            layout: Tensor with one-hot label channels and box channels.
            clamp: Whether to clamp latent box channels before conversion.

        Returns:
            Layout generation output with boxes in normalized center ``xywh``.
        """
        decoded = layout.clone()
        bbox_latent = decoded[:, :, self.num_classes_with_pad :]
        if clamp:
            bbox_latent = bbox_latent.clamp(-1.0, 1.0)
        bbox = bbox_latent / 2 + 0.5
        labels = decoded[:, :, : self.num_classes_with_pad].argmax(dim=2).long()
        mask = labels != self.pad_label_id
        return LayoutGenerationOutput(
            bbox=bbox,
            labels=labels,
            mask=mask,
            id2label=self.id2label,
            intermediates={"dataset": self.dataset},
        )

    def save_pretrained(  # ty: ignore[invalid-method-override]
        self, save_directory: str | Path
    ) -> None:
        """Save processor config to a Diffusers directory.

        Args:
            save_directory: Directory where ``processor_config.json`` is written.
        """
        super().save_pretrained(save_directory)

    @classmethod
    def from_pretrained(  # ty: ignore[invalid-method-override]
        cls,
        pretrained_model_name_or_path: str | Path,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
    ) -> "LaceProcessor":
        """Load processor config from a Diffusers directory.

        Args:
            pretrained_model_name_or_path: Directory or Hub id containing
                ``processor_config.json``.
            cache_dir: Optional Hugging Face cache directory.
            force_download: Whether to force a fresh download.
            local_files_only: Whether to avoid network access.
            token: Optional Hugging Face token.
            revision: Hub revision to load.

        Returns:
            Loaded processor.
        """
        return super().from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )

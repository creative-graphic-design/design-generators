"""Input processing for layout FID evaluators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from transformers import ProcessorMixin

from laygen.common.bbox import (
    BoxFormat,
    prepare_layout_tensors,
    xywh_to_ltrb,
)

from .configuration_layout_fid import LayoutFIDConfig


@dataclass(frozen=True)
class LayoutFIDBatch:
    """Model-ready layout FID batch."""

    bbox: torch.Tensor
    labels: torch.Tensor
    padding_mask: torch.Tensor
    mask: torch.Tensor
    id2label: dict[int, str] | None


class LayoutFIDProcessor(ProcessorMixin):
    """Convert public layout tensors into layout FID model inputs."""

    config_name = "processor_config.json"

    def __init__(self, config: LayoutFIDConfig) -> None:
        """Create a processor.

        Args:
            config: Explicit layout FID configuration.
        """
        super().__init__()
        self.config = config

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
        id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        box_format: BoxFormat | str = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        label_id_offset: int | None = None,
        max_length: int | None = None,
        pad_label_id: int | None = None,
        device: torch.device | str | None = None,
    ) -> LayoutFIDBatch:
        """Prepare model inputs from the repository public layout schema.

        Args:
            bbox: Public layout boxes.
            labels: Public dataset-local label ids.
            mask: Optional public valid-element mask.
            id2label: Optional public id-to-label metadata.
            box_format: Public input box format.
            normalized: Whether boxes are normalized to ``[0, 1]``.
            canvas_size: Pixel canvas size required when ``normalized=False``.
            label_id_offset: Optional parity/debug label-offset override.
            max_length: Optional maximum sequence length override.
            pad_label_id: Optional padded-position model label id.
            device: Target torch device.

        Returns:
            A ``LayoutFIDBatch`` with ``padding_mask=True`` for padded elements.

        Raises:
            ValueError: If metadata or tensor shapes are inconsistent.

        Examples:
            >>> from layout_fid import LayoutFIDConfig, LayoutFIDProcessor
            >>> cfg = LayoutFIDConfig(
            ...     dataset_name="publaynet", architecture="layoutnet",
            ...     source="layoutflow", num_public_labels=5,
            ...     num_label_embeddings=6, max_length=2,
            ... )
            >>> batch = LayoutFIDProcessor(cfg)(
            ...     bbox=[[[0.5, 0.5, 0.2, 0.2]]], labels=[[0]]
            ... )
            >>> batch.padding_mask.tolist()
            [[False, True]]
        """
        target_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        bbox_t, labels_t, mask_t = prepare_layout_tensors(
            bbox=bbox,
            labels=labels,
            mask=mask,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )
        bbox_t = bbox_t.to(device=target_device)
        labels_t = labels_t.to(device=target_device)
        mask_t = mask_t.to(device=target_device)
        max_len = max_length if max_length is not None else self.config.max_length
        pad_id = pad_label_id if pad_label_id is not None else self.config.pad_label_id
        bbox_t = self._pad_tensor(bbox_t, max_len, 0.0)
        labels_t = self._pad_tensor(labels_t, max_len, pad_id)
        mask_t = self._pad_tensor(mask_t, max_len, False)
        model_bbox = bbox_t
        if self.config.bbox_format_for_model == "ltrb":
            model_bbox = xywh_to_ltrb(model_bbox)
        offset = (
            label_id_offset
            if label_id_offset is not None
            else self.config.label_id_offset
        )
        model_labels = labels_t + offset
        model_labels = torch.where(
            mask_t, model_labels, torch.full_like(model_labels, pad_id)
        )
        if model_labels[mask_t].numel() and (
            int(model_labels[mask_t].min()) < 0
            or int(model_labels[mask_t].max()) >= self.config.num_label_embeddings
        ):
            raise ValueError("labels after label_id_offset exceed embedding table")
        normalized_id2label = self._normalize_id2label(id2label)
        if (
            normalized_id2label is not None
            and normalized_id2label != self.config.id2label
        ):
            raise ValueError("id2label does not match the evaluator config")
        return LayoutFIDBatch(
            bbox=model_bbox * mask_t.unsqueeze(-1),
            labels=model_labels,
            padding_mask=~mask_t,
            mask=mask_t,
            id2label=normalized_id2label,
        )

    def save_pretrained(  # ty: ignore[invalid-method-override]
        self, save_directory: str | os.PathLike[str]
    ) -> tuple[str]:
        """Save processor metadata.

        Args:
            save_directory: Directory receiving ``processor_config.json``.

        Returns:
            Tuple containing the saved config path.
        """
        import json
        import os

        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, self.config_name)
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump({"config_class": self.config.__class__.__name__}, file_obj)
            file_obj.write("\n")
        return (path,)

    @classmethod
    def from_pretrained(  # ty: ignore[invalid-method-override]
        cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs: object
    ) -> "LayoutFIDProcessor":
        """Load a processor from a saved model directory.

        Args:
            pretrained_model_name_or_path: Local path or Hub id.
            kwargs: Extra config-loading keyword arguments.

        Returns:
            Loaded processor bound to the model config.
        """
        config = LayoutFIDConfig.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
        return cls(config=config)

    @staticmethod
    def _normalize_id2label(
        id2label: Mapping[int, str] | Mapping[str, str] | None,
    ) -> dict[int, str] | None:
        if id2label is None:
            return None
        return {int(key): value for key, value in id2label.items()}

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

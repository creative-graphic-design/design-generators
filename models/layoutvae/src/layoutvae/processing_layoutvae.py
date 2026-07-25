"""Processor for LayoutVAE label-set encoding and layout decoding."""

from __future__ import annotations

from typing import Literal, TypeAlias, cast

import torch
from jaxtyping import Bool, Float, Int
from transformers import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding

from laygen.common.labels import DatasetName, id2label_for_dataset, label2id_for_dataset
from laygen.common.labels import normalize_dataset_name

from .configuration_layoutvae import Id2LabelMapping

INTERNAL_EMPTY_LABEL_ID = 0
InputLabelsTensor: TypeAlias = Int[torch.Tensor, "..."]
MaybeUnbatchedLabelTensor: TypeAlias = Int[torch.Tensor, "..."]
MaybeUnbatchedBboxTensor: TypeAlias = Float[torch.Tensor, "... 4"]
PublicBboxTensor: TypeAlias = Float[torch.Tensor, "batch elements 4"]
PublicLabelTensor: TypeAlias = Int[torch.Tensor, "batch elements"]
PublicMaskTensor: TypeAlias = Bool[torch.Tensor, "batch elements"]


class LayoutVAEProcessor(ProcessorMixin):
    """Encode PubLayNet labels into LayoutVAE label sets.

    Args:
        dataset_name: Dataset key. The first release supports PubLayNet.
        id2label: Optional public ID-to-label mapping.

    Raises:
        ValueError: If the dataset is unsupported.

    Examples:
        >>> processor = LayoutVAEProcessor()
        >>> processor.label2id["text"]
        0
    """

    config_name = "preprocessor_config.json"

    def __init__(
        self,
        dataset_name: DatasetName | str = DatasetName.publaynet,
        id2label: Id2LabelMapping | None = None,
    ) -> None:
        """Initialize the processor.

        Args:
            dataset_name: Dataset key. The first release supports PubLayNet.
            id2label: Optional public ID-to-label mapping.

        Raises:
            ValueError: If the dataset is unsupported.

        Examples:
            >>> LayoutVAEProcessor("publaynet").id2label[4]
            'figure'
        """
        self.chat_template = None
        canonical_dataset = normalize_dataset_name(dataset_name)
        if canonical_dataset is not DatasetName.publaynet:
            raise ValueError("LayoutVAEProcessor supports only PubLayNet")
        self.dataset_name = str(canonical_dataset)
        raw_id2label = id2label or id2label_for_dataset(canonical_dataset)
        self.id2label = {int(k): v for k, v in raw_id2label.items()}
        self.label2id = label2id_for_dataset(canonical_dataset)
        self.internal_id2label = {
            INTERNAL_EMPTY_LABEL_ID: "None",
            **{index + 1: label for index, label in self.id2label.items()},
        }

    def __call__(
        self,
        labels: list[list[str | int]] | list[str | int] | InputLabelsTensor,
        *,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding:
        """Encode public labels as a six-way label-set tensor.

        Args:
            labels: Public label names or IDs. A flat list is treated as one row.
            return_tensors: Tensor framework. Only `pt` is supported.

        Returns:
            Batch encoding with `label_set`.

        Raises:
            ValueError: If labels are empty, unknown, or tensors are unsupported.

        Examples:
            >>> encoded = LayoutVAEProcessor()(["text", "figure"])
            >>> encoded["label_set"].tolist()
            [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
        """
        if return_tensors != "pt":
            raise ValueError("LayoutVAEProcessor only supports return_tensors='pt'")
        rows = self._normalize_rows(labels)
        label_set = torch.zeros(
            len(rows), len(self.internal_id2label), dtype=torch.float32
        )
        for row_index, row in enumerate(rows):
            for label in row:
                public_id = self._label_to_id(label)
                label_set[row_index, public_id + 1] = 1.0
        return BatchEncoding({"label_set": label_set})

    def public_from_internal(
        self,
        internal_labels: Int[torch.Tensor, "batch elements"],
    ) -> tuple[
        Int[torch.Tensor, "batch elements"], Bool[torch.Tensor, "batch elements"]
    ]:
        """Map six-way labels to public labels and validity masks.

        Args:
            internal_labels: Internal label IDs where zero marks empty slots.

        Returns:
            Public label IDs and mask tensors.

        Examples:
            >>> processor = LayoutVAEProcessor()
            >>> labels, mask = processor.public_from_internal(torch.tensor([[0, 1, 5]]))
            >>> labels.tolist(), mask.tolist()
            ([[0, 0, 4]], [[False, True, True]])
        """
        labels = torch.clamp(internal_labels.to(dtype=torch.long) - 1, min=0)
        mask = internal_labels.to(dtype=torch.long) != INTERNAL_EMPTY_LABEL_ID
        return labels, mask

    def batch_decode(
        self,
        bbox: PublicBboxTensor,
        labels: PublicLabelTensor,
        mask: PublicMaskTensor | None = None,
    ) -> list[list[dict[str, object]]]:
        """Decode layout tensors into records.

        Args:
            bbox: Public normalized center `xywh` boxes.
            labels: Public label IDs.
            mask: Optional valid-element mask.

        Returns:
            Nested records with label text, label ID, and box coordinates.

        Raises:
            KeyError: If a public label ID is unknown.

        Examples:
            >>> records = LayoutVAEProcessor().batch_decode(
            ...     torch.zeros(1, 1, 4), torch.tensor([[0]])
            ... )
            >>> records[0][0]["label"]
            'text'
        """
        bbox_t = torch.as_tensor(bbox, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.long)
        labels_t, bbox_t = self._ensure_batched(labels_t, bbox_t)
        mask_t = self._prepare_mask(mask, labels_t.shape)
        records = []
        for boxes, ids, valid in zip(bbox_t, labels_t, mask_t, strict=True):
            row = []
            for box, label_id in zip(boxes[valid], ids[valid], strict=True):
                idx = int(label_id.item())
                row.append(
                    {
                        "label": self.id2label[idx],
                        "label_id": idx,
                        "bbox": box.tolist(),
                    }
                )
            records.append(row)
        return records

    def _ensure_batched(
        self, labels: MaybeUnbatchedLabelTensor, bbox: MaybeUnbatchedBboxTensor
    ) -> tuple[PublicLabelTensor, PublicBboxTensor]:
        if labels.ndim != 1:
            return labels, bbox
        return labels.unsqueeze(0), bbox.unsqueeze(0)

    def _prepare_mask(
        self,
        mask: PublicMaskTensor | object | None,
        shape: torch.Size,
    ) -> PublicMaskTensor:
        if mask is None:
            return torch.ones(shape, dtype=torch.bool)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        return mask_t.unsqueeze(0) if mask_t.ndim == 1 else mask_t

    def _normalize_rows(
        self,
        labels: list[list[str | int]] | list[str | int] | InputLabelsTensor,
    ) -> list[list[str | int]]:
        if isinstance(labels, torch.Tensor):
            if labels.ndim == 0 or labels.ndim > 2:
                raise ValueError("labels tensor must have one or two dimensions")
            if labels.ndim == 1:
                return [[int(value) for value in labels.tolist()]]
            return [[int(value) for value in row] for row in labels.tolist()]
        if not labels:
            raise ValueError("labels must not be empty")
        contains_rows = [isinstance(item, list) for item in labels]
        if any(contains_rows) and not all(contains_rows):
            raise ValueError("labels must be a flat list or list of rows")
        if all(contains_rows):
            return [list(row) for row in cast(list[list[str | int]], labels)]
        return [list(cast(list[str | int], labels))]

    def _label_to_id(self, label: str | int) -> int:
        if not isinstance(label, int):
            if label in self.label2id:
                return self.label2id[label]
            raise ValueError(f"Unknown label: {label}")
        if 0 <= label < len(self.id2label):
            return label
        raise ValueError(f"Unknown label id: {label}")

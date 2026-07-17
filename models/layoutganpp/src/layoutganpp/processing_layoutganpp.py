"""Processor for LayoutGAN++ label encoding and output decoding."""

from __future__ import annotations

from typing import Literal

import torch
from transformers.tokenization_utils_base import BatchEncoding

from .configuration_layoutganpp import Id2LabelMapping
from .datasets import DatasetName, id2label_for_dataset, normalize_dataset_name


class LayoutGANPPProcessor:
    """Encode LayoutGAN++ labels and decode generated layouts.

    Args:
        dataset_name: Dataset key or alias.
        id2label: Optional label ID to text mapping.

    Examples:
        >>> processor = LayoutGANPPProcessor(dataset_name="rico")
        >>> processor.label2id["Toolbar"]
        0
    """

    config_name = "preprocessor_config.json"

    def __init__(
        self,
        dataset_name: DatasetName | str = DatasetName.rico13,
        id2label: Id2LabelMapping | None = None,
    ) -> None:
        """Initialize a LayoutGAN++ processor.

        Args:
            dataset_name: Dataset key or alias.
            id2label: Optional label ID to text mapping.

        Raises:
            ValueError: If the dataset name is unsupported.

        Examples:
            >>> LayoutGANPPProcessor("publaynet").id2label[0]
            'text'
        """
        self.dataset_name = str(normalize_dataset_name(dataset_name))
        raw_id2label = id2label or id2label_for_dataset(self.dataset_name)
        self.id2label = {int(k): v for k, v in raw_id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def __call__(
        self,
        labels: list[list[str | int]] | list[str | int] | torch.Tensor,
        *,
        padding: bool = True,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding:
        """Encode label strings or IDs into tensors.

        Args:
            labels: Label strings, label IDs, or a tensor of label IDs.
            padding: Whether to pad ragged batches.
            return_tensors: Tensor framework. Only `pt` is supported.

        Returns:
            Batch encoding with `labels` and `attention_mask` tensors.

        Raises:
            ValueError: If labels are empty, ragged without padding, unknown,
                or `return_tensors` is not `pt`.

        Examples:
            >>> processor = LayoutGANPPProcessor()
            >>> encoded = processor(["Toolbar", "Image"])
            >>> tuple(encoded["labels"].shape)
            (1, 2)
        """
        if return_tensors != "pt":
            raise ValueError("LayoutGANPPProcessor only supports return_tensors='pt'")
        rows = self._normalize_rows(labels)
        max_len = max(len(row) for row in rows)
        if not padding and len({len(row) for row in rows}) != 1:
            raise ValueError("Ragged labels require padding=True")
        encoded = []
        attention = []
        for row in rows:
            ids = [self._label_to_id(label) for label in row]
            pad = max_len - len(ids)
            encoded.append(ids + [0] * pad)
            attention.append([True] * len(ids) + [False] * pad)
        return BatchEncoding(
            {
                "labels": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.bool),
            }
        )

    def batch_decode(
        self,
        bbox: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[list[dict[str, object]]]:
        """Decode generated boxes and label IDs into records.

        Args:
            bbox: Generated boxes shaped `(batch, sequence, 4)` or `(sequence, 4)`.
            labels: Label IDs shaped `(batch, sequence)` or `(sequence,)`.
            attention_mask: Optional valid-element mask.

        Returns:
            Nested records containing label text, label ID, and bounding box.

        Raises:
            KeyError: If a label ID is not known to this processor.

        Examples:
            >>> processor = LayoutGANPPProcessor()
            >>> records = processor.batch_decode(
            ...     torch.zeros(1, 1, 4), torch.tensor([[0]])
            ... )
            >>> records[0][0]["label"]
            'Toolbar'
        """
        bbox_t = torch.as_tensor(bbox, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.long)
        if labels_t.ndim == 1:
            labels_t = labels_t.unsqueeze(0)
            bbox_t = bbox_t.unsqueeze(0)
        if attention_mask is None:
            mask_t = torch.ones(labels_t.shape, dtype=torch.bool)
        else:
            mask_t = torch.as_tensor(attention_mask, dtype=torch.bool)
            if mask_t.ndim == 1:
                mask_t = mask_t.unsqueeze(0)
        records = []
        for boxes, ids, mask in zip(bbox_t, labels_t, mask_t, strict=True):
            row = []
            for box, label_id in zip(boxes[mask], ids[mask], strict=True):
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

    def save_pretrained(self, save_directory: str) -> None:
        """Save processor metadata to a directory.

        Args:
            save_directory: Directory that will receive the processor config file.

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> processor = LayoutGANPPProcessor()
            >>> with TemporaryDirectory() as tmp:
            ...     processor.save_pretrained(tmp)
        """
        from pathlib import Path
        import json

        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {"dataset_name": self.dataset_name, "id2label": self.id2label}
        (path / self.config_name).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, path: str) -> "LayoutGANPPProcessor":
        """Load processor metadata from a directory.

        Args:
            path: Directory containing `preprocessor_config.json`.

        Returns:
            A loaded `LayoutGANPPProcessor`.

        Raises:
            FileNotFoundError: If the processor config is missing.

        Examples:
            >>> # LayoutGANPPProcessor.from_pretrained("./layoutganpp-rico")
        """
        from pathlib import Path
        import json

        payload = json.loads((Path(path) / cls.config_name).read_text())
        return cls(
            dataset_name=payload["dataset_name"],
            id2label={int(k): v for k, v in payload["id2label"].items()},
        )

    def _normalize_rows(
        self, labels: list[list[str | int]] | list[str | int] | torch.Tensor
    ) -> list[list[str | int]]:
        if isinstance(labels, torch.Tensor):
            if labels.ndim == 1:
                return [[int(v) for v in labels.tolist()]]
            return [[int(v) for v in row] for row in labels.tolist()]
        if not labels:
            raise ValueError("labels must not be empty")
        first = labels[0]
        if isinstance(first, list):
            rows: list[list[str | int]] = []
            for row in labels:
                if not isinstance(row, list):
                    raise ValueError("labels must be a flat list or list of rows")
                rows.append(row)
            return rows
        row = []
        for label in labels:
            if isinstance(label, list):
                raise ValueError("labels must be a flat list or list of rows")
            row.append(label)
        return [row]

    def _label_to_id(self, label: str | int) -> int:
        if isinstance(label, int):
            if label < 0 or label >= len(self.id2label):
                raise ValueError(f"Unknown label id: {label}")
            return label
        try:
            return self.label2id[label]
        except KeyError as exc:
            raise ValueError(f"Unknown label: {label}") from exc


def processor_for_dataset(dataset_name: DatasetName | str) -> LayoutGANPPProcessor:
    """Create a processor with the default labels for a dataset.

    Args:
        dataset_name: Dataset key or alias.

    Returns:
        Processor initialized with the dataset's default label mapping.

    Raises:
        ValueError: If the dataset name is unknown.

    Examples:
        >>> processor_for_dataset("magazine").dataset_name
        'magazine'
    """
    return LayoutGANPPProcessor(
        dataset_name=dataset_name,
        id2label=id2label_for_dataset(dataset_name),
    )

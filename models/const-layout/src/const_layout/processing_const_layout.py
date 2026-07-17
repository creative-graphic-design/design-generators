from __future__ import annotations

from typing import Literal

import torch
from transformers.tokenization_utils_base import BatchEncoding

from .datasets import id2label_for_dataset, normalize_dataset_name


class ConstLayoutProcessor:
    config_name = "preprocessor_config.json"

    def __init__(
        self,
        dataset_name: str = "rico",
        id2label: dict[int | str, str] | None = None,
    ) -> None:
        self.dataset_name = normalize_dataset_name(dataset_name)
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
        if return_tensors != "pt":
            raise ValueError("ConstLayoutProcessor only supports return_tensors='pt'")
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
        from pathlib import Path
        import json

        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {"dataset_name": self.dataset_name, "id2label": self.id2label}
        (path / self.config_name).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, path: str) -> "ConstLayoutProcessor":
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
            return labels  # type: ignore[return-value]
        return [labels]  # type: ignore[list-item]

    def _label_to_id(self, label: str | int) -> int:
        if isinstance(label, int):
            if label < 0 or label >= len(self.id2label):
                raise ValueError(f"Unknown label id: {label}")
            return label
        try:
            return self.label2id[label]
        except KeyError as exc:
            raise ValueError(f"Unknown label: {label}") from exc


def processor_for_dataset(dataset_name: str) -> ConstLayoutProcessor:
    return ConstLayoutProcessor(
        dataset_name=dataset_name,
        id2label=id2label_for_dataset(dataset_name),
    )

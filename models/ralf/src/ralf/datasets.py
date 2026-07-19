"""Dataset adapters for RALF-compatible poster layouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal
from typing import Protocol, cast

import torch

from laygen.common.bbox import ltwh_to_xywh, ltrb_to_xywh
from posgen.common.labels import DatasetName, normalize_dataset_name

from .retrieval import RalfRetrievedBatch


class _IndexableDataset(Protocol):
    """Minimal protocol for an indexable dataset."""

    def __getitem__(self, index: int) -> Mapping[str, object]:
        """Return one dataset row."""


def normalize_org_sample(
    sample: Mapping[str, object], dataset_name: str
) -> dict[str, object]:
    """Normalize one org-dataset sample to RALF-style fields.

    Args:
        sample: Dataset row.
        dataset_name: Poster dataset key.

    Returns:
        Dictionary with normalized public layout fields.

    Raises:
        ValueError: If the dataset is unsupported.
    """
    dataset = normalize_dataset_name(dataset_name)
    if dataset in {DatasetName.cgl, DatasetName.cgl_v2}:
        annotations_obj = sample.get("annotations", {})
        annotations = annotations_obj if isinstance(annotations_obj, Mapping) else {}
        bbox = torch.as_tensor(annotations.get("bbox", []), dtype=torch.float32)
        labels = torch.as_tensor(annotations.get("category", []), dtype=torch.long)
        width_obj = sample.get("width", 1)
        height_obj = sample.get("height", 1)
        width = int(width_obj) if isinstance(width_obj, int | float | str) else 1
        height = int(height_obj) if isinstance(height_obj, int | float | str) else 1
        scale = torch.tensor((width, height, width, height), dtype=torch.float32)
        bbox = ltwh_to_xywh(bbox / scale)
        return {
            "bbox": bbox,
            "labels": labels,
            "mask": torch.ones(labels.shape, dtype=torch.bool),
        }
    if dataset is DatasetName.pku_posterlayout:
        annotations_obj = sample.get("annotations", {})
        annotations = annotations_obj if isinstance(annotations_obj, Mapping) else {}
        bbox = torch.as_tensor(annotations.get("box_elem", []), dtype=torch.float32)
        labels = torch.as_tensor(annotations.get("cls_elem", []), dtype=torch.long)
        valid = labels.ne(3)
        size = sample.get("poster") or sample.get("canvas") or sample.get("image")
        width, height = getattr(size, "size", (1, 1))
        scale = torch.tensor((width, height, width, height), dtype=torch.float32)
        bbox = ltrb_to_xywh(bbox / scale)
        return {
            "bbox": bbox[valid],
            "labels": labels[valid],
            "mask": torch.ones(int(valid.sum()), dtype=torch.bool),
        }
    raise ValueError(f"Unsupported RALF dataset: {dataset_name}")


def load_ralf_dataset(
    dataset_name: Literal["cgl", "cgl_v2", "pku_posterlayout"],
    split: str,
    *,
    source: str = "hf_org",
) -> object:
    """Load a RALF-compatible dataset lazily.

    Args:
        dataset_name: Canonical poster dataset name.
        split: Dataset split.
        source: Data source. Only `hf_org` is supported.

    Returns:
        Hugging Face dataset object.

    Raises:
        ValueError: If the source or dataset is unsupported.
    """
    if source != "hf_org":
        raise ValueError("Only source='hf_org' is supported")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install ralf[vendor] to load org datasets") from exc
    dataset = normalize_dataset_name(dataset_name)
    if dataset is DatasetName.cgl:
        return load_dataset(
            "creative-graphic-design/CGL-Dataset", name="ralf-style", split=split
        )
    if dataset is DatasetName.cgl_v2:
        return load_dataset(
            "creative-graphic-design/CGL-Dataset-v2", name="ralf-style", split=split
        )
    if dataset is DatasetName.pku_posterlayout:
        return load_dataset(
            "creative-graphic-design/PKU-PosterLayout", name="ralf-style", split=split
        )
    raise ValueError(f"Unsupported RALF dataset: {dataset_name}")


def build_retrieved_batch(
    dataset: _IndexableDataset,
    indexes: torch.Tensor,
    *,
    max_seq_length: int,
) -> RalfRetrievedBatch:
    """Build explicit retrieved layout tensors from dataset indexes.

    Args:
        dataset: Indexable dataset whose rows match `normalize_org_sample`.
        indexes: Tensor of retrieved row indexes with shape `(batch, candidates)`.
        max_seq_length: Maximum elements retained per layout.

    Returns:
        Retrieved batch with layout fields filled and image tensors as zeros.
    """
    bbox_rows = []
    label_rows = []
    mask_rows = []
    for row in indexes.tolist():
        bbox_candidates = []
        label_candidates = []
        mask_candidates = []
        for idx in row:
            sample = normalize_org_sample(dataset[int(idx)], "cgl")
            bbox = torch.zeros(max_seq_length, 4)
            labels = torch.zeros(max_seq_length, dtype=torch.long)
            mask = torch.zeros(max_seq_length, dtype=torch.bool)
            sample_labels = cast(torch.Tensor, sample["labels"])
            sample_bbox = cast(torch.Tensor, sample["bbox"])
            length = min(max_seq_length, sample_labels.numel())
            bbox[:length] = sample_bbox[:length]
            labels[:length] = sample_labels[:length]
            mask[:length] = True
            bbox_candidates.append(bbox)
            label_candidates.append(labels)
            mask_candidates.append(mask)
        bbox_rows.append(torch.stack(bbox_candidates))
        label_rows.append(torch.stack(label_candidates))
        mask_rows.append(torch.stack(mask_candidates))
    bbox_tensor = torch.stack(bbox_rows)
    labels_tensor = torch.stack(label_rows)
    mask_tensor = torch.stack(mask_rows)
    batch, candidates = indexes.shape
    return RalfRetrievedBatch(
        image=torch.zeros(batch, candidates, 3, 1, 1),
        saliency=torch.zeros(batch, candidates, 1, 1, 1),
        bbox=bbox_tensor,
        labels=labels_tensor,
        mask=mask_tensor,
        indexes=indexes,
    )

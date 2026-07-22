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


PKU_ORG_TO_CHECKPOINT_LABEL_ID = torch.tensor([1, 0, 2], dtype=torch.long)
RALF_STYLE_LABEL2ID = {
    DatasetName.cgl: {"embellishment": 0, "logo": 1, "text": 2, "underlay": 3},
    DatasetName.cgl_v2: {"embellishment": 0, "logo": 1, "text": 2, "underlay": 3},
    DatasetName.pku_posterlayout: {"logo": 0, "text": 1, "underlay": 2},
}


def _remap_retrieval_labels(labels: torch.Tensor, dataset: DatasetName) -> torch.Tensor:
    if dataset is not DatasetName.pku_posterlayout:
        return labels
    if labels.numel() == 0:
        return labels
    return PKU_ORG_TO_CHECKPOINT_LABEL_ID[labels.clamp(0, 2)]


def _labels_to_tensor(labels: object, dataset: DatasetName) -> torch.Tensor:
    values = list(cast(list[object], labels))
    if values and isinstance(values[0], str):
        label2id = RALF_STYLE_LABEL2ID[dataset]
        return torch.tensor(
            [label2id[str(value)] for value in values], dtype=torch.long
        )
    return torch.as_tensor(labels, dtype=torch.long)


def normalize_org_sample(
    sample: Mapping[str, object], dataset_name: DatasetName | str
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
    if {"label", "center_x", "center_y", "width", "height"}.issubset(sample):
        labels = _labels_to_tensor(sample["label"], dataset)
        bbox = torch.stack(
            [
                torch.as_tensor(sample["center_x"], dtype=torch.float32),
                torch.as_tensor(sample["center_y"], dtype=torch.float32),
                torch.as_tensor(sample["width"], dtype=torch.float32),
                torch.as_tensor(sample["height"], dtype=torch.float32),
            ],
            dim=-1,
        )
        return {
            "bbox": bbox,
            "labels": labels,
            "mask": torch.ones(labels.shape, dtype=torch.bool),
        }
    if dataset in {DatasetName.cgl, DatasetName.cgl_v2}:
        annotations_obj = sample.get("annotations", {})
        annotations = annotations_obj if isinstance(annotations_obj, Mapping) else {}
        bbox = torch.as_tensor(annotations.get("bbox", []), dtype=torch.float32)
        if bbox.numel() == 0:
            bbox = bbox.reshape(0, 4)
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
        if bbox.numel() == 0:
            bbox = bbox.reshape(0, 4)
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
        raise ImportError(
            "Install the optional reference dependencies to load org datasets"
        ) from exc
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
    dataset_name: Literal["cgl", "cgl_v2", "pku", "pku_posterlayout"] = "cgl",
) -> RalfRetrievedBatch:
    """Build explicit retrieved layout tensors from dataset indexes.

    Args:
        dataset: Indexable dataset whose rows match `normalize_org_sample`.
        indexes: Tensor of retrieved row indexes with shape `(batch, candidates)`.
        max_seq_length: Maximum elements retained per layout.
        dataset_name: Dataset key used for row normalization. PKU labels are remapped
            from org dataset ids (`text=0`, `logo=1`, `underlay=2`) to the checkpoint
            ids (`logo=0`, `text=1`, `underlay=2`) used by converted RALF.

    Returns:
        Retrieved batch with layout fields filled and image tensors as zeros.
    """
    normalized_dataset = normalize_dataset_name(dataset_name)
    bbox_rows = []
    label_rows = []
    mask_rows = []
    for row in indexes.tolist():
        bbox_candidates = []
        label_candidates = []
        mask_candidates = []
        for idx in row:
            sample = normalize_org_sample(dataset[int(idx)], normalized_dataset)
            bbox = torch.zeros(max_seq_length, 4)
            labels = torch.zeros(max_seq_length, dtype=torch.long)
            mask = torch.zeros(max_seq_length, dtype=torch.bool)
            sample_labels = _remap_retrieval_labels(
                cast(torch.Tensor, sample["labels"]).long(), normalized_dataset
            )
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

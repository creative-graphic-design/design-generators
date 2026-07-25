"""Processor for Coarse-to-Fine layouts and hierarchy tensors."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from transformers import BatchEncoding, ProcessorMixin

from laygen.common.bbox import BoxFormat
from laygen.common.labels import DatasetName, id2label_for_dataset, label2id_for_dataset
from laygen.common.labels import normalize_dataset_name
from laygen.modeling_outputs import LayoutGenerationOutput

from .geometry import discretize_ltwh, public_to_ltwh
from .hierarchy import build_cut_hierarchy, flatten_hierarchy, CoarseToFineHierarchy
from .types import OutputType, normalize_output_type


class CoarseToFineProcessor(ProcessorMixin):
    """Convert public layouts to Coarse-to-Fine hierarchy tensors."""

    attributes: list[str] = []
    processor_class = "CoarseToFineProcessor"

    def __init__(
        self,
        dataset: DatasetName | str = DatasetName.rico25,
        *,
        x_grid: int = 128,
        y_grid: int = 128,
        max_num_elements: int = 20,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize label maps and discretization settings.

        Args:
            dataset: Canonical layout dataset.
            x_grid: Number of x/width bins.
            y_grid: Number of y/height bins.
            max_num_elements: Padded flat sequence length.
            id2label: Optional public label map.

        Examples:
            >>> CoarseToFineProcessor(dataset="publaynet").id2label[0]
            'text'
        """
        normalized_dataset = normalize_dataset_name(dataset)
        self.dataset = str(normalized_dataset)
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.max_num_elements = max_num_elements
        self.id2label = id2label or id2label_for_dataset(normalized_dataset)
        self.label2id = label2id_for_dataset(normalized_dataset)

    @classmethod
    def from_config(
        cls,
        dataset: DatasetName | str = DatasetName.rico25,
        *,
        x_grid: int = 128,
        y_grid: int = 128,
        max_num_elements: int = 20,
        id2label: dict[int, str] | None = None,
    ) -> "CoarseToFineProcessor":
        """Construct a processor without external files."""
        return cls(
            dataset=dataset,
            x_grid=x_grid,
            y_grid=y_grid,
            max_num_elements=max_num_elements,
            id2label=id2label,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize processor state to a JSON-compatible dictionary."""
        return {
            "processor_class": self.processor_class,
            "dataset": self.dataset,
            "x_grid": self.x_grid,
            "y_grid": self.y_grid,
            "max_num_elements": self.max_num_elements,
            "id2label": {str(key): value for key, value in self.id2label.items()},
        }

    def save_pretrained(
        self,
        save_directory: str | PathLike[str],
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> list[str]:
        """Save processor metadata next to a converted checkpoint."""
        _ = (push_to_hub, kwargs)
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        out_file = path / "preprocessor_config.json"
        out_file.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return [str(out_file)]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: object,
    ) -> "CoarseToFineProcessor":
        """Load processor metadata from a local ``save_pretrained`` directory."""
        _ = (cache_dir, force_download, local_files_only, token, revision, kwargs)
        path = Path(pretrained_model_name_or_path) / "preprocessor_config.json"
        data = json.loads(path.read_text())
        return cls(
            dataset=str(data["dataset"]),
            x_grid=int(data["x_grid"]),
            y_grid=int(data["y_grid"]),
            max_num_elements=int(data["max_num_elements"]),
            id2label={int(key): str(value) for key, value in data["id2label"].items()},
        )

    def _labels_to_vendor(
        self, labels: Int[torch.Tensor, "..."]
    ) -> Int[torch.Tensor, "..."]:
        return cast(torch.LongTensor, labels.long() + 1)

    def __call__(
        self,
        labels: list[list[int | str]]
        | Int[torch.Tensor, "batch elements"]
        | Int[np.ndarray, "batch elements"]
        | None = None,
        bbox: list[list[list[float]]]
        | Float[torch.Tensor, "batch elements 4"]
        | Float[np.ndarray, "batch elements 4"]
        | None = None,
        mask: Bool[torch.Tensor, "batch elements"]
        | Bool[np.ndarray, "batch elements"]
        | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding:
        """Build discrete flat tensors from public layout inputs.

        Args:
            labels: Public zero-based labels or label strings.
            bbox: Public normalized boxes.
            mask: Optional valid-element mask.
            box_format: Input box format.
            normalized: Must be true; pixel boxes are dataset-loader work.
            return_tensors: Only ``"pt"`` is supported.

        Returns:
            BatchEncoding with internal labels, discrete boxes, and masks.

        Raises:
            ValueError: If labels or boxes are missing.
        """
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported")
        if not normalized:
            raise ValueError("CoarseToFineProcessor expects normalized boxes")
        if labels is None or bbox is None:
            raise ValueError("labels and bbox are required for processor encoding")
        label_tensor = self._coerce_labels(labels)
        bbox_tensor = torch.as_tensor(bbox, dtype=torch.float32)
        encoded_mask = (
            cast(torch.BoolTensor, torch.ones(label_tensor.shape, dtype=torch.bool))
            if mask is None
            else cast(torch.BoolTensor, torch.as_tensor(mask, dtype=torch.bool))
        )
        ltwh = public_to_ltwh(bbox_tensor, box_format=box_format)
        discrete = discretize_ltwh(ltwh, num_x_grid=self.x_grid, num_y_grid=self.y_grid)
        vendor_labels = self._labels_to_vendor(label_tensor)
        return BatchEncoding(
            {"labels": vendor_labels, "bbox": discrete, "mask": encoded_mask}
        )

    def _coerce_labels(
        self,
        labels: list[list[int | str]]
        | Int[torch.Tensor, "batch elements"]
        | Int[np.ndarray, "batch elements"],
    ) -> Int[torch.Tensor, "batch elements"]:
        if isinstance(labels, torch.Tensor | np.ndarray):
            return cast(torch.LongTensor, torch.as_tensor(labels, dtype=torch.long))
        rows: list[list[int]] = []
        for row in labels:
            values: list[int] = []
            for label in row:
                if isinstance(label, int):
                    values.append(label)
                else:
                    values.append(self.label2id[label])
            rows.append(values)
        max_len = max((len(row) for row in rows), default=1)
        padded = [row + [0] * (max_len - len(row)) for row in rows]
        return cast(torch.LongTensor, torch.tensor(padded, dtype=torch.long))

    def build_hierarchy_batch(
        self,
        labels: Int[torch.Tensor, "batch elements"],
        bbox: Float[torch.Tensor, "batch elements 4"],
        mask: Bool[torch.Tensor, "batch elements"],
    ) -> BatchEncoding:
        """Build padded hierarchy tensors for training/reference batches."""
        batch = labels.size(0)
        flat_enc = self(labels=labels, bbox=bbox, mask=mask, return_tensors="pt")
        encodings = []
        for idx in range(batch):
            valid = mask[idx].bool()
            encodings.append(
                build_cut_hierarchy(
                    public_to_ltwh(bbox[idx, valid]),
                    self._labels_to_vendor(labels[idx, valid]),
                    num_labels=len(self.id2label),
                    discrete_x_grid=self.x_grid,
                    discrete_y_grid=self.y_grid,
                )
            )
        max_groups = max(enc.group_bounding_box.size(0) for enc in encodings) + 2
        max_decode_groups = max_groups - 2
        max_group_elems = (
            max(max(group.size(0) for group in enc.grouped_labels) for enc in encodings)
            + 2
        )
        group_boxes = []
        group_labels = []
        group_masks = []
        grouped_boxes = []
        grouped_labels = []
        grouped_masks = []
        for enc in encodings:
            group_count = enc.group_bounding_box.size(0)
            sos_group_box = torch.zeros((1, 4), dtype=torch.long)
            eos_group_box = torch.zeros((1, 4), dtype=torch.long)
            padded_group_box = torch.cat(
                (sos_group_box, enc.group_bounding_box.long(), eos_group_box), dim=0
            )
            padded_group_box = F.pad(
                padded_group_box, (0, 0, 0, max_groups - padded_group_box.size(0))
            )
            hist = torch.zeros(
                (group_count + 2, len(self.id2label) + 2), dtype=torch.float32
            )
            hist[0, 0] = 1.0
            hist[1 : group_count + 1, 1:-1] = enc.label_in_one_group.float()
            hist[group_count + 1, -1] = 1.0
            hist = F.pad(hist, (0, 0, 0, max_groups - hist.size(0)))
            group_mask = torch.arange(max_groups) < group_count + 2
            per_group_boxes = []
            per_group_labels = []
            per_group_masks = []
            for group_idx in range(group_count):
                box = enc.grouped_bbox[group_idx].long()
                label = enc.grouped_labels[group_idx].long()
                label = torch.cat(
                    (
                        torch.tensor([len(self.id2label) + 1]),
                        label,
                        torch.tensor([len(self.id2label) + 2]),
                    )
                )
                box = torch.cat(
                    (
                        torch.zeros((1, 4), dtype=torch.long),
                        box,
                        torch.zeros((1, 4), dtype=torch.long),
                    )
                )
                valid_count = min(label.size(0), max_group_elems)
                per_group_boxes.append(
                    F.pad(
                        box[:max_group_elems], (0, 0, 0, max_group_elems - valid_count)
                    )
                )
                per_group_labels.append(
                    F.pad(label[:max_group_elems], (0, max_group_elems - valid_count))
                )
                per_group_masks.append(torch.arange(max_group_elems) < valid_count)
            while len(per_group_boxes) < max_decode_groups:
                per_group_boxes.append(
                    torch.zeros((max_group_elems, 4), dtype=torch.long)
                )
                per_group_labels.append(
                    torch.zeros((max_group_elems,), dtype=torch.long)
                )
                per_group_masks.append(
                    torch.zeros((max_group_elems,), dtype=torch.bool)
                )
            group_boxes.append(padded_group_box)
            group_labels.append(hist)
            group_masks.append(group_mask)
            grouped_boxes.append(torch.stack(per_group_boxes[:max_decode_groups]))
            grouped_labels.append(torch.stack(per_group_labels[:max_decode_groups]))
            grouped_masks.append(torch.stack(per_group_masks[:max_decode_groups]))
        flat_enc.update(
            {
                "group_bounding_box": torch.stack(group_boxes),
                "label_in_one_group": torch.stack(group_labels),
                "group_mask": torch.stack(group_masks),
                "grouped_bbox": torch.stack(grouped_boxes),
                "grouped_labels": torch.stack(grouped_labels),
                "grouped_mask": torch.stack(grouped_masks),
            }
        )
        return flat_enc

    def post_process_hierarchy(
        self,
        hierarchy: CoarseToFineHierarchy,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Convert decoded hierarchy tensors to the shared output schema."""
        output = flatten_hierarchy(
            hierarchy,
            id2label=dict(self.id2label),
            max_num_elements=self.max_num_elements,
        )
        if not return_intermediates:
            output.intermediates = None
        if normalize_output_type(output_type) is OutputType.dict:
            return dict(output)
        return output

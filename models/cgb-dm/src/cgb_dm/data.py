"""Dataset utilities for CGB-DM original zip extracts."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset

from laygen.common.bbox import ltrb_to_xywh

from .processing_cgb_dm import CGBDMProcessor


@dataclass(frozen=True)
class CGBDMDataPaths:
    """Paths for one CGB-DM split in an extracted dataset tree."""

    root: Path
    split: str = "train"

    @property
    def inpaint_dir(self) -> Path:
        """Return the inpaint image directory."""
        return self.root / self.split / "inpaint"

    @property
    def saliency_dir(self) -> Path:
        """Return the first saliency directory."""
        return self.root / self.split / "saliency"

    @property
    def saliency_sub_dir(self) -> Path:
        """Return the second saliency directory."""
        return self.root / self.split / "saliency_sub"

    @property
    def annotation_csv(self) -> Path:
        """Return the element annotation CSV path."""
        return self.root / "csv" / f"{self.split}.csv"

    @property
    def saliency_csv(self) -> Path:
        """Return the saliency-box CSV path."""
        return self.root / "csv" / f"{self.split}_sal.csv"


class CGBDMOriginalDataset(Dataset[dict[str, torch.Tensor]]):
    """Read an extracted CGB-DM dataset split without downloading assets.

    Args:
        root: Extracted dataset root.
        split: Dataset split name.
        processor: Processor used for image/layout normalization.

    Examples:
        >>> CGBDMDataPaths(Path("/tmp/data")).annotation_csv.name
        'train.csv'
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: Literal["train", "val", "test"] | str = "train",
        processor: CGBDMProcessor | None = None,
    ) -> None:
        """Initialize file lists and CSV indexes."""
        self.paths = CGBDMDataPaths(Path(root), split)
        self.processor = processor or CGBDMProcessor()
        self.names = sorted(path.name for path in self.paths.inpaint_dir.iterdir())
        self.annotations = _read_grouped_boxes(self.paths.annotation_csv)
        self.saliency_boxes = _read_grouped_boxes(self.paths.saliency_csv)

    def __len__(self) -> int:
        """Return number of image rows."""
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one normalized CGB-DM training row."""
        name = self.names[index]
        image_path = self.paths.inpaint_dir / name
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        saliency = Image.open(self.paths.saliency_dir / name).convert("L")
        saliency_sub = Image.open(self.paths.saliency_sub_dir / name).convert("L")
        content = self.processor(
            image,
            saliency_isnet=saliency,
            saliency_basnet=saliency_sub,
            saliency_box=_normalize_ltrb(self.saliency_boxes[name][0], width, height),
        )
        boxes, labels = zip(*self.annotations[name], strict=False)
        layout = self.processor.encode_layout(
            bbox=[[_normalize_ltrb(box, width, height).tolist() for box in boxes]],
            labels=[list(labels)],
        )["layout"][0]
        return {
            "pixel_values": content["pixel_values"][0],
            "layout": layout,
            "saliency_box": content["saliency_box"][0],
        }


def _read_grouped_boxes(
    path: Path,
) -> dict[str, list[tuple[tuple[float, float, float, float], int]]]:
    rows: dict[str, list[tuple[tuple[float, float, float, float], int]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = ast.literal_eval(row["box_elem"])
            box = (
                float(parsed[0]),
                float(parsed[1]),
                float(parsed[2]),
                float(parsed[3]),
            )
            cls = int(row.get("cls_elem", 0))
            rows.setdefault(row["poster_path"], []).append((box, cls))
    return rows


def _normalize_ltrb(
    box: tuple[float, float, float, float]
    | tuple[tuple[float, float, float, float], int],
    width: int,
    height: int,
) -> torch.Tensor:
    if isinstance(box[0], tuple):
        box = box[0]
    tensor = torch.tensor(box, dtype=torch.float32)
    tensor = tensor / torch.tensor((width, height, width, height), dtype=torch.float32)
    return ltrb_to_xywh(tensor).clamp(0.0, 1.0)

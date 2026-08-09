"""Small DLT datasets used by smoke training configs and tests."""

import random
from pathlib import Path
from typing import TypedDict

import h5py
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch.utils.data import Dataset


class DLTExample(TypedDict):
    """One DLT training example with conditioning masks."""

    box: Float[torch.Tensor, "elements 4"]
    box_cond: Float[torch.Tensor, "elements 4"]
    cat: Int[torch.Tensor, "elements"]
    mask: Bool[torch.Tensor, "elements"]
    mask_box: Int[torch.Tensor, "elements 4"]
    mask_cat: Int[torch.Tensor, "elements"]


class DLTStepTrace(TypedDict, total=False):
    """Diagnostic tensors captured from one DLT training step."""

    box: Float[torch.Tensor, "batch elements 4"]
    box_cond: Float[torch.Tensor, "batch elements 4"]
    cat: Int[torch.Tensor, "batch elements"]
    mask: Bool[torch.Tensor, "batch elements"]
    mask_box: Int[torch.Tensor, "batch elements 4"]
    mask_cat: Int[torch.Tensor, "batch elements"]
    noise: Float[torch.Tensor, "batch elements 4"]
    t: Int[torch.Tensor, "batch"]
    noised_box: Float[torch.Tensor, "batch elements 4"]
    noised_cat: Int[torch.Tensor, "batch elements"]
    pred_box: Float[torch.Tensor, "batch elements 4"]
    pred_cat: Float[torch.Tensor, "batch elements categories"]
    masked_l2: Float[torch.Tensor, ""]
    masked_ce: Float[torch.Tensor, ""]
    loss: Float[torch.Tensor, ""]


class SyntheticDLTDataset(Dataset[DLTExample]):
    """Deterministic synthetic DLT batches that never download data."""

    def __init__(
        self,
        *,
        length: int = 8,
        max_num_comp: int = 4,
        categories_num: int = 7,
        seed: int = 0,
    ) -> None:
        """Initialize a synthetic dataset."""
        self.length = length
        self.max_num_comp = max_num_comp
        self.categories_num = categories_num
        self.seed = seed

    def __len__(self) -> int:
        """Return dataset length."""
        return self.length

    def __getitem__(self, index: int) -> DLTExample:
        """Return one deterministic synthetic DLT sample."""
        generator = torch.Generator().manual_seed(self.seed + index)
        box = torch.rand(self.max_num_comp, 4, generator=generator) * 4.0 - 2.0
        cat = torch.randint(
            1, self.categories_num - 1, (self.max_num_comp,), generator=generator
        )
        mask = torch.ones(self.max_num_comp, dtype=torch.bool)
        mask_box = torch.ones(self.max_num_comp, 4, dtype=torch.long)
        mask_cat = torch.ones(self.max_num_comp, dtype=torch.long)
        return {
            "box": box.float(),
            "box_cond": box.float(),
            "cat": cat.long(),
            "mask": mask,
            "mask_box": mask_box,
            "mask_cat": mask_cat,
        }


class H5DLTDataset(Dataset[DLTExample]):
    """DLT PubLayNet dataset backed by LayoutFlow-style HDF5 files."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_num_comp: int = 9,
    ) -> None:
        """Index valid HDF5 rows without loading the full file into memory."""
        self.path = Path(path)
        self.max_num_comp = max_num_comp
        with h5py.File(self.path, "r") as data:
            self.keys = [
                key
                for key in sorted(data.keys(), key=int)
                if 1 < int(data[key]["length"][()]) <= self.max_num_comp
                and 1
                < int(_valid_ltwh_mask(np.asarray(data[key]["bbox"])).sum())
                <= self.max_num_comp
            ]

    def __len__(self) -> int:
        """Return the number of valid layouts."""
        return len(self.keys)

    def __getitem__(self, index: int) -> DLTExample:
        """Return one DLT training sample."""
        box, cat, _, _ = self.get_data_by_ix(index)
        mask_box, mask_cat = _mask_instance(box.shape)
        box, cat, mask_box, mask_cat = _pad_instance(
            box, cat, mask_box, mask_cat, self.max_num_comp
        )
        return {
            "box": torch.tensor(box, dtype=torch.float32),
            "box_cond": torch.tensor(box.copy(), dtype=torch.float32),
            "cat": torch.tensor(cat, dtype=torch.long),
            "mask": torch.tensor(cat != 0, dtype=torch.bool),
            "mask_box": torch.tensor(mask_box, dtype=torch.long),
            "mask_cat": torch.tensor(mask_cat, dtype=torch.long),
        }

    def get_data_by_ix(
        self, index: int
    ) -> tuple[
        Float[np.ndarray, "elements 4"],
        Int[np.ndarray, "elements"],
        list[int],
        str,
    ]:
        """Return one unpadded layout with reference-style element shuffling."""
        with h5py.File(self.path, "r") as data:
            key = self.keys[index]
            row = data[key]
            box = np.asarray(row["bbox"], dtype=np.float32)
            cat = np.asarray(row["categories"], dtype=int)
            length = int(row["length"][()])
        box = box[:length]
        cat = cat[:length]
        valid = _valid_ltwh_mask(box)
        box = box[valid]
        cat = cat[valid]
        length = box.shape[0]
        order = list(range(length))
        random.shuffle(order)
        box = box[order]
        cat = cat[order]
        box = _ltwh_to_scaled_xywh(box)
        return box, cat.astype(np.int64), order, str(key)


def _ltwh_to_scaled_xywh(
    box: Float[np.ndarray, "elements 4"],
) -> Float[np.ndarray, "elements 4"]:
    xywh = box.copy()
    xywh[:, 0] = box[:, 0] + box[:, 2] / 2.0
    xywh[:, 1] = box[:, 1] + box[:, 3] / 2.0
    return ((xywh * 2.0) - 1.0) * 2.0


def _valid_ltwh_mask(
    box: Float[np.ndarray, "elements 4"],
) -> Bool[np.ndarray, "elements"]:
    x1 = box[:, 0]
    y1 = box[:, 1]
    x2 = x1 + box[:, 2]
    y2 = y1 + box[:, 3]
    return (x1 >= 0) & (y1 >= 0) & (x2 <= 1) & (y2 <= 1) & (x2 > x1) & (y2 > y1)


def _mask_instance(
    bbox_shape: tuple[int, int],
) -> tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]:
    index = int(np.random.choice(6, 1, p=[0.2, 0.1, 0.05, 0.25, 0.1, 0.3])[0])
    if index == 0:
        return _mask_loc(bbox_shape, r_mask=np.random.uniform(0.5, 1.0, size=1)[0])
    if index == 1:
        return _mask_size(bbox_shape, r_mask=np.random.uniform(0.5, 1.0, size=1)[0])
    if index == 2:
        return _mask_cat(bbox_shape, r_mask=np.random.uniform(0.5, 1.0, size=1)[0])
    if index == 3:
        return _mask_whole_box(
            bbox_shape, r_mask=np.random.uniform(0.5, 1.0, size=1)[0]
        )
    if index == 4:
        return _mask_random(
            bbox_shape,
            r_mask_box=np.random.uniform(0.5, 1.0, size=1)[0],
            r_mask_cat=np.random.uniform(0.5, 1.0, size=1)[0],
        )
    return _mask_all(bbox_shape)


def _choice_mask(
    bbox_shape: tuple[int, int], r_mask: float
) -> Int[np.ndarray, "selected"]:
    count = int(bbox_shape[0] * r_mask)
    if count:
        return np.random.choice(range(bbox_shape[0]), count, replace=False)
    return np.asarray([], dtype=int)


def _mask_loc(
    bbox_shape: tuple[int, int], r_mask: float = 1.0
) -> tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]:
    rows = _choice_mask(bbox_shape, r_mask)
    mask = np.zeros(bbox_shape)
    mask[rows, :2] = 1
    return mask, np.zeros(bbox_shape[0]).astype("long")


def _mask_size(
    bbox_shape: tuple[int, int], r_mask: float = 1.0
) -> tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]:
    rows = _choice_mask(bbox_shape, r_mask)
    mask = np.zeros(bbox_shape)
    mask[rows, 2:] = 1
    return mask, np.zeros(bbox_shape[0]).astype("long")


def _mask_cat(
    bbox_shape: tuple[int, int], r_mask: float = 1.0
) -> tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]:
    rows = _choice_mask(bbox_shape, r_mask)
    mask_cat = np.zeros(bbox_shape[0]).astype("long")
    mask_cat[rows] = 1
    return np.zeros(bbox_shape), mask_cat


def _mask_whole_box(
    bbox_shape: tuple[int, int], r_mask: float = 1.0
) -> tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]:
    rows = _choice_mask(bbox_shape, r_mask)
    mask = np.zeros(bbox_shape)
    mask[rows, :4] = 1
    return mask, np.zeros(bbox_shape[0]).astype("long")


def _mask_random(
    bbox_shape: tuple[int, int], r_mask_box: float = 1.0, r_mask_cat: float = 1.0
) -> tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]:
    mask_options = [_mask_loc, _mask_size, [_mask_loc, _mask_size], _mask_whole_box]
    option = mask_options[np.random.choice(range(len(mask_options)), 1)[0]]
    if isinstance(option, list):
        mask_box = np.zeros(bbox_shape)
        for func in option:
            part, _ = func(bbox_shape, r_mask_box)
            mask_box += part
        base_cat_mask = np.zeros(bbox_shape[0]).astype("long")
    else:
        mask_box, base_cat_mask = option(bbox_shape, r_mask_box)
    _, full_cat_mask = _mask_cat(bbox_shape, r_mask_cat)
    return mask_box, [base_cat_mask, full_cat_mask][np.random.choice(2, 1)[0]]


def _mask_all(
    bbox_shape: tuple[int, int],
) -> tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]:
    return np.ones(bbox_shape), np.ones(bbox_shape[0]).astype("long")


def _pad_instance(
    box: Float[np.ndarray, "elements 4"],
    cat: Int[np.ndarray, "elements"],
    mask_box: Float[np.ndarray, "elements 4"],
    mask_cat: Int[np.ndarray, "elements"],
    max_num_comp: int,
) -> tuple[
    Float[np.ndarray, "max_elements 4"],
    Int[np.ndarray, "max_elements"],
    Float[np.ndarray, "max_elements 4"],
    Int[np.ndarray, "max_elements"],
]:
    pad = max_num_comp - box.shape[0]
    box = np.pad(box, pad_width=((0, pad), (0, 0)), constant_values=0.0)
    cat = np.pad(cat, pad_width=(0, pad), constant_values=0.0)
    mask_box = np.pad(mask_box, pad_width=((0, pad), (0, 0)), constant_values=0.0)
    mask_cat = np.pad(mask_cat, pad_width=(0, pad), constant_values=0.0)
    return box, cat, mask_box, mask_cat


def collate_dlt_batch(
    examples: list[DLTExample],
) -> DLTExample:
    """Stack DLT examples into one batch."""
    return {
        "box": torch.stack([example["box"] for example in examples]),
        "box_cond": torch.stack([example["box_cond"] for example in examples]),
        "cat": torch.stack([example["cat"] for example in examples]),
        "mask": torch.stack([example["mask"] for example in examples]),
        "mask_box": torch.stack([example["mask_box"] for example in examples]),
        "mask_cat": torch.stack([example["mask_cat"] for example in examples]),
    }

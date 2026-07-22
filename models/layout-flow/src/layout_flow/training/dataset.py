"""HDF5 dataset and collation helpers for LayoutFlow training."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch.utils.data import Dataset, default_collate

from laygen.common.bbox import BoxFormat

_SPLIT_FILES: Final[dict[str, dict[str, str]]] = {
    "rico25": {
        "train": "ldm_rico_train.h5",
        "validation": "ldm_rico_val.h5",
        "test": "ldm_rico_test.h5",
    },
    "publaynet": {
        "train": "publaynet_train.h5",
        "validation": "publaynet_val.h5",
        "test": "publaynet_test.h5",
    },
}
RawSample: TypeAlias = dict[str, torch.Tensor | str]


class LayoutFlowH5Dataset(Dataset[dict[str, torch.Tensor | str]]):
    """Vendor-compatible HDF5 dataset for LayoutFlow training."""

    def __init__(
        self,
        *,
        data_path: str | Path,
        dataset_name: str,
        split: str = "train",
        lex_order: bool = False,
        permute_elements: bool = False,
        inoue_split: bool = False,
    ) -> None:
        """Open one LayoutFlow HDF5 split.

        Args:
            data_path: Directory containing vendor HDF5 files.
            dataset_name: ``rico25`` or ``publaynet``.
            split: ``train``, ``validation``, or ``test``.
            lex_order: Whether to use vendor lexical-order files.
            permute_elements: Whether to permute elements at access time.
            inoue_split: Whether PubLayNet uses Inoue split filenames.

        Raises:
            ValueError: If the dataset or split is unsupported.
            FileNotFoundError: If the expected HDF5 file is absent.
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.dataset_name = dataset_name
        self.split = split
        self.permute_elements = permute_elements
        file_name = self._file_name(dataset_name, split, lex_order, inoue_split)
        path = self.data_path / file_name
        if not path.exists():
            raise FileNotFoundError(path)
        import h5pickle as h5py

        self.data = h5py.File(str(path))
        self.keys = list(self.data.keys())

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.keys)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        """Return a raw sample from the HDF5 file."""
        key = self.keys[index]
        sample = self.data[key]
        sample_dict: dict[str, torch.Tensor | str] = {"id": str(key)}
        for feature in sample.keys():
            value = torch.from_numpy(np.array(sample[feature]))
            sample_dict["type" if feature == "categories" else feature] = value
        if self.permute_elements:
            length = int(torch.as_tensor(sample_dict["length"]).item())
            randperm = torch.randperm(length)
            sample_dict["type"] = torch.as_tensor(sample_dict["type"])[randperm]
            sample_dict["bbox"] = torch.as_tensor(sample_dict["bbox"])[randperm]
        return sample_dict

    @staticmethod
    def _file_name(
        dataset_name: str,
        split: str,
        lex_order: bool,
        inoue_split: bool,
    ) -> str:
        if dataset_name == "rico25":
            prefix = "ldm_lex_rico" if lex_order else "ldm_rico"
            return f"{prefix}_{split if split != 'validation' else 'val'}.h5"
        if dataset_name == "publaynet":
            split_key = "val" if split == "validation" else split
            if inoue_split:
                return f"publaynet_{split_key}_inoue.h5"
            if lex_order:
                return f"ldm_lex_publaynet_{split_key}.h5"
            return _SPLIT_FILES[dataset_name][split]
        raise ValueError(f"Unsupported LayoutFlow dataset_name: {dataset_name}")


def collate_layout_flow_batch(
    batch: Sequence[RawSample],
    *,
    max_length: int | None = None,
    box_format: BoxFormat | str = BoxFormat.xywh,
) -> dict[str, torch.Tensor | list[str]]:
    """Collate LayoutFlow samples with vendor-compatible padding.

    Args:
        batch: Raw sample dictionaries.
        max_length: Optional fixed maximum sequence length.
        box_format: Output box format. ``xywh`` converts vendor ``ltwh`` boxes to
            center coordinates.

    Returns:
        Collated batch with ``bbox``, ``type``, ``mask``, ``length``, and optional
        ``id``.

    Raises:
        ValueError: If box format is unsupported.

    Examples:
        >>> sample = {"bbox": torch.tensor([[0.0, 0.0, 0.2, 0.4]]), "type": torch.tensor([1]), "length": torch.tensor(1)}
        >>> collate_layout_flow_batch([sample], max_length=2)["bbox"].shape
        torch.Size([1, 2, 4])
    """
    total_elems = [len(example["type"]) for example in batch]
    target_length = max(total_elems) if max_length is None else max_length
    collated: list[RawSample] = []
    fmt = BoxFormat(box_format)
    for example, total in zip(batch, total_elems, strict=True):
        item = dict(example)
        length = int(torch.as_tensor(item["length"]).squeeze().item())
        clipped = min(length, target_length)
        item["length"] = torch.tensor(clipped, dtype=torch.int)
        item["mask"] = _mask_tensor(clipped, target_length)
        item["type"] = _copy_1d(torch.as_tensor(item["type"]), target_length, torch.int)
        bbox = _copy_bbox(
            torch.as_tensor(item["bbox"], dtype=torch.float32), total, target_length
        )
        if fmt is BoxFormat.xywh:
            bbox[:, 0] += bbox[:, 2] / 2
            bbox[:, 1] += bbox[:, 3] / 2
        elif fmt is BoxFormat.ltrb:
            bbox[:, 2] += bbox[:, 0]
            bbox[:, 3] += bbox[:, 1]
        elif fmt is not BoxFormat.ltwh:
            raise ValueError(f"Unsupported box_format: {box_format}")
        item["bbox"] = bbox
        collated.append(item)
    ids = [str(item.pop("id")) for item in collated if "id" in item]
    output = default_collate(collated)
    if ids:
        output["id"] = ids
    return output


def _mask_tensor(length: int, max_length: int) -> Bool[torch.Tensor, "elements 1"]:
    mask = torch.zeros(max_length, 1, dtype=torch.bool)
    mask[:length] = True
    return mask


def _copy_1d(
    tensor: torch.Tensor, max_length: int, dtype: torch.dtype
) -> Int[torch.Tensor, "elements"]:
    out = torch.zeros(max_length, dtype=dtype)
    out[: min(tensor.shape[0], max_length)] = tensor[:max_length].to(dtype=dtype)
    return out


def _copy_bbox(
    tensor: torch.Tensor, total: int, max_length: int
) -> Float[torch.Tensor, "elements 4"]:
    out = torch.zeros(max_length, 4, dtype=torch.float32)
    out[: min(total, max_length)] = tensor[:max_length]
    return out

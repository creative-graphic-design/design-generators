"""Retrieval containers and adapters for RALF."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Bool, Float, Int


@dataclass
class RalfRetrievedBatch:
    """Batch of explicit retrieved examples for RALF.

    Args:
        image: Retrieved RGB images with shape `(batch, candidates, channels, height, width)`.
        saliency: Retrieved saliency maps with shape `(batch, candidates, 1, height, width)`.
        bbox: Retrieved normalized center `xywh` boxes.
        labels: Retrieved dataset-local labels.
        mask: Retrieved valid-element masks.
        indexes: Optional selected cache indexes.
    """

    image: Float[torch.Tensor, "batch candidates channels height width"]
    saliency: Float[torch.Tensor, "batch candidates 1 height width"]
    bbox: Float[torch.Tensor, "batch candidates elements 4"]
    labels: Int[torch.Tensor, "batch candidates elements"]
    mask: Bool[torch.Tensor, "batch candidates elements"]
    indexes: Int[torch.Tensor, "batch candidates"] | None = None


class RalfRetrievalTable:
    """Lookup table from query ids to retrieved training indexes.

    Args:
        table: Mapping from query ids to ordered retrieved ids.
        top_k: Number of retrieved ids returned per query.

    Examples:
        >>> table = RalfRetrievalTable({"a": [3, 4, 5]}, top_k=2)
        >>> table.lookup(["a"]).tolist()
        [[3, 4]]
    """

    def __init__(self, table: Mapping[int | str, Sequence[int]], top_k: int) -> None:
        """Initialize lookup table."""
        self.table = {
            str(key): [int(value) for value in values] for key, values in table.items()
        }
        self.top_k = int(top_k)

    @classmethod
    def from_pretrained(
        cls, path: str | Path, top_k: int | None = None
    ) -> "RalfRetrievalTable":
        """Load `retrieval_table.json` from a checkpoint directory."""
        root = Path(path)
        with (root / "retrieval_table.json").open() as f:
            payload = json.load(f)
        return cls(payload["table"], top_k=top_k or int(payload["top_k"]))

    def save_pretrained(self, save_directory: str | Path) -> tuple[str]:
        """Save the table next to converted checkpoint metadata."""
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "retrieval_table.json"
        with path.open("w") as f:
            json.dump(
                {"top_k": self.top_k, "table": self.table}, f, indent=2, sort_keys=True
            )
        return (str(path),)

    def lookup(self, ids: Sequence[int | str]) -> Int[torch.Tensor, "batch candidates"]:
        """Return retrieved indexes for query ids."""
        rows = []
        for item in ids:
            values = self.table[str(item)][: self.top_k]
            if len(values) < self.top_k:
                values = values + [-1] * (self.top_k - len(values))
            rows.append(values)
        return torch.tensor(rows, dtype=torch.long)


def retrieved_batch_to_vendor_dict(
    batch: RalfRetrievedBatch,
) -> dict[str, torch.Tensor]:
    """Convert explicit retrieved examples to vendor-style field names."""
    x, y, w, h = batch.bbox.unbind(dim=-1)
    output = {
        "image": batch.image,
        "saliency": batch.saliency,
        "center_x": x,
        "center_y": y,
        "width": w,
        "height": h,
        "label": batch.labels,
        "mask": batch.mask,
    }
    if batch.indexes is not None:
        output["index"] = batch.indexes
    return output


def vendor_dict_to_retrieved_batch(
    data: Mapping[str, torch.Tensor],
) -> RalfRetrievedBatch:
    """Convert vendor retrieved fields to `RalfRetrievedBatch`."""
    bbox = torch.stack(
        (
            data["center_x"],
            data["center_y"],
            data["width"],
            data["height"],
        ),
        dim=-1,
    )
    return RalfRetrievedBatch(
        image=data["image"],
        saliency=data["saliency"],
        bbox=bbox,
        labels=data["label"].long(),
        mask=data["mask"].bool(),
        indexes=data.get("index"),
    )

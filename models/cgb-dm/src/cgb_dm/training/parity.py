"""S0-S2 parity adapters for CGB-DM."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Literal, cast

import torch
from jaxtyping import Float
from cgb_dm.data import CGBDMOriginalDataset


class CGBDMStepTraceAdapter:
    """Adapter exposing comparable CGB-DM training-step trace tensors."""

    trace_points = (
        "pixel_values",
        "layout",
        "saliency_box",
        "t",
        "noise",
        "fix_mask",
        "noisy_layout",
        "predicted_epsilon",
        "cgb_weight",
        "loss",
    )

    def comparable_batch(
        self,
        batch: Mapping[str, Float[torch.Tensor, "..."]]
        | tuple[
            Float[torch.Tensor, "..."],
            Float[torch.Tensor, "..."],
            Float[torch.Tensor, "..."],
        ],
    ) -> Mapping[str, Float[torch.Tensor, "..."]]:
        """Normalize dict or tuple batches to comparable tensor mappings."""
        if isinstance(batch, tuple):
            image, layout, saliency_box = batch
            result: dict[str, Float[torch.Tensor, "..."]] = {}
            result["pixel_values"] = cast(Float[torch.Tensor, "..."], image)
            result["layout"] = cast(Float[torch.Tensor, "..."], layout)
            result["saliency_box"] = cast(Float[torch.Tensor, "..."], saliency_box)
            return result
        return batch


def capture_source_order(data_root: str | Path, *, split: str = "train") -> list[str]:
    """Capture the filename order used by the original CGB-DM training loader."""
    return list(os.listdir(Path(data_root) / split / "inpaint"))


def write_source_order_manifest(
    *,
    data_root: str | Path,
    output: str | Path,
    dataset: str,
    split: str = "train",
    seed: int = 1,
) -> Path:
    """Write a regenerated source-order manifest outside the repository."""
    names = capture_source_order(data_root, split=split)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "split": split,
                "seed": seed,
                "source": "reference train_dataset os.listdir order",
                "data_root": str(data_root),
                "names": names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_source_order_manifest(path: str | Path) -> list[str]:
    """Load names from a regenerated source-order manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [str(name) for name in payload["names"]]


def build_reference_dataset(
    data_root: str | Path,
    *,
    manifest: str | Path,
    split: Literal["train", "val", "test"] = "train",
) -> CGBDMOriginalDataset:
    """Build a CGB-DM dataset that replays captured source order and encoding."""
    return CGBDMOriginalDataset(
        data_root,
        split=split,
        name_manifest=manifest,
        encoding="reference",
    )

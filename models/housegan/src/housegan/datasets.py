"""Local dataset adapters for House-GAN vectorized floorplan assets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np

from .graph_schema import (
    HouseGanRelation,
    HouseGanRoomNode,
    HouseGanSceneGraph,
    relation_from_bboxes,
)

TARGET_SETS: dict[str, tuple[int, int]] = {
    "A": (1, 3),
    "B": (4, 6),
    "C": (7, 9),
    "D": (10, 12),
    "E": (13, 100),
}


def load_housegan_numpy(path: str | Path) -> np.ndarray:
    """Load a local House-GAN ``.npy`` asset without downloading data."""
    return np.load(Path(path), allow_pickle=True)


def normalize_vendor_graph_row(
    row: object,
    *,
    canvas_size: tuple[int, int] = (256, 256),
) -> HouseGanSceneGraph:
    """Convert one vendor ``[room_types, ltrb_boxes]`` row to a scene graph."""
    del canvas_size
    room_types, room_bbs = cast(tuple[object, object], row)
    bboxes = np.asarray(room_bbs, dtype=np.float32) / 256.0
    labels = [int(value) - 1 for value in cast(list[int], room_types)]
    nodes = tuple(
        HouseGanRoomNode(
            id=index,
            label=label,
            bbox=cast(tuple[float, float, float, float], tuple(map(float, bbox))),
        )
        for index, (label, bbox) in enumerate(zip(labels, bboxes, strict=True))
    )
    return HouseGanSceneGraph(
        nodes=nodes, relations=relation_from_bboxes(bboxes.tolist())
    )


def split_target_set(
    graphs: list[HouseGanSceneGraph],
    *,
    target_set: Literal["A", "B", "C", "D", "E"],
    split: Literal["train", "eval"],
) -> list[HouseGanSceneGraph]:
    """Apply House-GAN target-set graph-size splits."""
    low, high = TARGET_SETS[target_set]
    rows: list[HouseGanSceneGraph] = []
    for graph in graphs:
        in_range = low <= len(graph.nodes) <= high
        if (split == "eval" and in_range) or (split == "train" and not in_range):
            rows.append(graph)
    return rows


def build_edges_from_bboxes(
    bbox_ltrb: np.ndarray,
    *,
    threshold: float = 0.03,
) -> list[HouseGanRelation]:
    """Build public adjacency relations from normalized ``ltrb`` boxes."""
    return list(relation_from_bboxes(bbox_ltrb.tolist(), threshold=threshold))

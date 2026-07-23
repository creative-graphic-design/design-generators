"""Scene-graph schema normalization for House-GAN."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch


@dataclass(frozen=True)
class HouseGanRoomNode:
    """Room node in a House-GAN scene graph."""

    id: int
    label: int | str
    bbox: tuple[float, float, float, float] | None = None
    attributes: Mapping[str, object] | None = None


@dataclass(frozen=True)
class HouseGanRelation:
    """Adjacency relation between two room nodes."""

    source: int
    target: int
    adjacent: bool
    weight: float | None = None


@dataclass(frozen=True)
class HouseGanSceneGraph:
    """Flat room relation graph used by House-GAN."""

    nodes: tuple[HouseGanRoomNode, ...]
    relations: tuple[HouseGanRelation, ...] | None = None


def normalize_scene_graph(
    scene_graph: HouseGanSceneGraph | Mapping[str, object] | None,
    *,
    labels: object | None,
    relations: object | None,
    id2label: Mapping[int, str],
) -> HouseGanSceneGraph:
    """Normalize public scene-graph payloads.

    Args:
        scene_graph: Dataclass or mapping with ``nodes`` and ``edges`` fields.
        labels: Optional labels used when no full scene graph is supplied.
        relations: Optional relation payload.
        id2label: Public label map.

    Returns:
        Normalized scene graph preserving node order.

    Raises:
        ValueError: If no nodes can be resolved or labels are invalid.

    Examples:
        >>> graph = normalize_scene_graph(None, labels=[0, 1], relations=[], id2label={0: "a", 1: "b"})
        >>> len(graph.nodes)
        2
    """
    if isinstance(scene_graph, HouseGanSceneGraph):
        return scene_graph
    if scene_graph is None:
        if labels is None:
            raise ValueError("scene_graph or labels must be provided")
        label_values = _to_label_sequence(labels)
        nodes = tuple(
            HouseGanRoomNode(id=index, label=label)
            for index, label in enumerate(label_values)
        )
        return HouseGanSceneGraph(
            nodes=nodes,
            relations=_normalize_relations(relations),
        )
    nodes_payload = cast(
        Sequence[object],
        scene_graph.get("nodes", scene_graph.get("rooms", ())),
    )
    nodes: list[HouseGanRoomNode] = []
    for index, raw_node in enumerate(nodes_payload):
        item = cast(Mapping[str, object], raw_node)
        raw_id = item.get("id", index)
        raw_label = item.get("label_id", item.get("label"))
        if raw_label is None:
            raise ValueError("Each House-GAN node requires a label")
        raw_bbox = item.get("bbox")
        bbox = tuple(cast(Sequence[float], raw_bbox)) if raw_bbox is not None else None
        nodes.append(
            HouseGanRoomNode(
                id=int(cast(int | str, raw_id)),
                label=cast(int | str, raw_label),
                bbox=cast(tuple[float, float, float, float] | None, bbox),
                attributes=cast(Mapping[str, object] | None, item.get("attributes")),
            )
        )
    edges = scene_graph.get("edges", scene_graph.get("relations", relations))
    if not nodes:
        raise ValueError("House-GAN relation graphs require at least one node")
    _ = id2label
    return HouseGanSceneGraph(nodes=tuple(nodes), relations=_normalize_relations(edges))


def complete_signed_edges(
    nodes: Sequence[HouseGanRoomNode],
    relations: Sequence[HouseGanRelation] | None,
    *,
    default_adjacent: bool = False,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Build signed complete graph triples.

    Args:
        nodes: Room nodes in preserved graph order.
        relations: Sparse public relations.
        default_adjacent: Whether missing pairs become adjacent.
        device: Optional tensor device.

    Returns:
        ``LongTensor`` with rows ``[source_index, sign, target_index]``.
    """
    node_index = {node.id: index for index, node in enumerate(nodes)}
    relation_map: dict[tuple[int, int], bool] = {}
    for relation in relations or ():
        left = node_index[relation.source]
        right = node_index[relation.target]
        key = (left, right) if left < right else (right, left)
        relation_map[key] = relation.adjacent
    edges: list[list[int]] = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            adjacent = relation_map.get((left, right), default_adjacent)
            edges.append([left, 1 if adjacent else -1, right])
    return cast(torch.LongTensor, torch.tensor(edges, dtype=torch.long, device=device))


def graph_to_node_features(
    nodes: Sequence[HouseGanRoomNode],
    *,
    label2id: Mapping[str, int],
    num_labels: int,
    device: torch.device | None = None,
) -> torch.FloatTensor:
    """Convert public room labels to 10-way one-hot features."""
    label_ids = [_label_to_id(node.label, label2id=label2id) for node in nodes]
    labels_t = torch.tensor(label_ids, dtype=torch.long, device=device)
    if labels_t.numel() and (
        int(labels_t.min().item()) < 0 or int(labels_t.max().item()) >= num_labels
    ):
        raise ValueError("House-GAN labels must be dataset-local ids in range")
    return cast(
        torch.FloatTensor,
        torch.nn.functional.one_hot(labels_t, num_classes=num_labels).to(
            dtype=torch.float32
        ),
    )


def relation_from_bboxes(
    bbox_ltrb: Sequence[Sequence[float]],
    *,
    threshold: float = 0.03,
) -> tuple[HouseGanRelation, ...]:
    """Derive House-GAN adjacency relations from normalized ``ltrb`` boxes."""
    relations: list[HouseGanRelation] = []
    for left in range(len(bbox_ltrb)):
        for right in range(left + 1, len(bbox_ltrb)):
            relations.append(
                HouseGanRelation(
                    source=left,
                    target=right,
                    adjacent=_is_adjacent(bbox_ltrb[left], bbox_ltrb[right], threshold),
                )
            )
    return tuple(relations)


def _to_label_sequence(labels: object) -> list[int | str]:
    if isinstance(labels, torch.Tensor):
        values = labels.detach().cpu().tolist()
    else:
        values = labels
    if (
        isinstance(values, Sequence)
        and values
        and isinstance(values[0], Sequence)
        and not isinstance(values[0], str | bytes)
    ):
        values = values[0]
    return [cast(int | str, item) for item in cast(Sequence[object], values)]


def _normalize_relations(raw_relations: object | None) -> tuple[HouseGanRelation, ...]:
    if raw_relations is None:
        return ()
    relations: list[HouseGanRelation] = []
    for item in cast(Sequence[object], raw_relations):
        if isinstance(item, HouseGanRelation):
            relations.append(item)
            continue
        if isinstance(item, Mapping):
            source = int(cast(int | str, item.get("source", item.get("subject"))))
            target = int(cast(int | str, item.get("target", item.get("object"))))
            predicate = item.get("adjacent", item.get("predicate", item.get("sign")))
            adjacent = _predicate_to_adjacent(predicate)
            relations.append(
                HouseGanRelation(
                    source=source,
                    target=target,
                    adjacent=adjacent,
                    weight=cast(float | None, item.get("weight")),
                )
            )
            continue
        values = list(cast(Sequence[object], item))
        if len(values) != 3:
            raise ValueError("Relation tuples must have three values")
        source, middle, target = values
        if middle in (-1, 1, "-1", "1"):
            adjacent = int(cast(int | str, middle)) > 0
        else:
            adjacent = _predicate_to_adjacent(middle)
        relations.append(
            HouseGanRelation(
                source=int(cast(int | str, source)),
                target=int(cast(int | str, target)),
                adjacent=adjacent,
            )
        )
    return tuple(relations)


def _predicate_to_adjacent(predicate: object) -> bool:
    if isinstance(predicate, bool):
        return predicate
    if isinstance(predicate, int):
        return predicate > 0
    normalized = str(predicate).lower().replace("-", "_")
    if normalized in {"adjacent", "touching", "connected", "1", "true"}:
        return True
    if normalized in {"not_adjacent", "non_adjacent", "none", "-1", "false"}:
        return False
    raise ValueError(f"Unknown House-GAN relation predicate: {predicate}")


def _label_to_id(label: int | str, *, label2id: Mapping[str, int]) -> int:
    if isinstance(label, int):
        return label
    if label in label2id:
        return label2id[label]
    lowered = label.lower()
    for name, label_id in label2id.items():
        if name.lower() == lowered:
            return label_id
    raise ValueError(f"Unknown House-GAN room label: {label}")


def _is_adjacent(
    box_a: Sequence[float],
    box_b: Sequence[float],
    threshold: float,
) -> bool:
    x0, y0, x1, y1 = box_a
    x2, y2, x3, y3 = box_b
    h1, h2 = x1 - x0, x3 - x2
    w1, w2 = y1 - y0, y3 - y2
    xc1, xc2 = (x0 + x1) / 2.0, (x2 + x3) / 2.0
    yc1, yc2 = (y0 + y1) / 2.0, (y2 + y3) / 2.0
    delta_x = abs(xc2 - xc1) - (h1 + h2) / 2.0
    delta_y = abs(yc2 - yc1) - (w1 + w2) / 2.0
    return max(delta_x, delta_y) < threshold

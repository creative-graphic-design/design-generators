"""Public scene-graph dataclasses for LayoutTransformer processors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutObject:
    """One scene-graph object node.

    Args:
        id: Stable object id within a scene graph.
        label: Dataset-local object label id or label string.
        bbox: Optional normalized center ``xywh`` constraint.

    Examples:
        >>> LayoutObject(id="person-1", label="person").id
        'person-1'
    """

    id: int | str
    label: int | str
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class LayoutRelation:
    """One directed scene-graph edge.

    Args:
        subject: Source object id.
        predicate: Relation id or label.
        object: Target object id.
        bbox_delta: Optional relation geometry.
        score: Optional edge confidence.

    Examples:
        >>> LayoutRelation("a", "left of", "b").predicate
        'left of'
    """

    subject: int | str
    predicate: int | str
    object: int | str
    bbox_delta: tuple[float, float, float, float] | None = None
    score: float | None = None


@dataclass(frozen=True)
class SceneGraphInput:
    """Normalized scene graph payload.

    Args:
        objects: Scene-graph object nodes.
        relations: Scene-graph relation edges.
        id2label: Optional public object label mapping.
        relation_id2label: Optional relation label mapping.

    Examples:
        >>> graph = SceneGraphInput(
        ...     objects=(LayoutObject("a", "person"), LayoutObject("b", "table")),
        ...     relations=(LayoutRelation("a", "left of", "b"),),
        ... )
        >>> len(graph.relations)
        1
    """

    objects: tuple[LayoutObject, ...]
    relations: tuple[LayoutRelation, ...]
    id2label: dict[int, str] | None = None
    relation_id2label: dict[int, str] | None = None

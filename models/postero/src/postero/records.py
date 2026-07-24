"""Typed in-memory records consumed by PosterO prompts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import BaseModel, ConfigDict, Field


class AvailableRegion(BaseModel):
    """One available poster area in pixel ``ltrb`` coordinates."""

    bbox_ltrb: tuple[float, float, float, float]

    model_config = ConfigDict(frozen=True)


class PosterLayoutElement(BaseModel):
    """One poster layout element in pixel ``ltrb`` coordinates."""

    label: int | str
    bbox_ltrb: tuple[float, float, float, float]
    rotation: float | None = None
    path_data: str | None = None

    model_config = ConfigDict(frozen=True)


class PosterORecord(BaseModel):
    """One poster prompt or exemplar record.

    Args:
        id: Stable record id used in parity reports.
        dataset: Dataset key.
        poster_path: Optional source image path.
        canvas_size: Canvas size as ``(width, height)``.
        available_regions: Content-aware available regions.
        elements: Ground-truth or generated layout elements.
        features: Optional precomputed retrieval feature vector.
        metrics: Optional quality metrics used by pool filtering.
        metadata: Extra non-runtime metadata.
    """

    id: str
    dataset: str
    poster_path: str | None = None
    canvas_size: tuple[int, int] = (513, 750)
    available_regions: list[AvailableRegion] = Field(default_factory=list)
    elements: list[PosterLayoutElement] = Field(default_factory=list)
    features: list[float] | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


def record_from_mapping(
    row: Mapping[str, object], *, id2label: Mapping[int, str]
) -> PosterORecord:
    """Build a ``PosterORecord`` from a lightweight mapping.

    Args:
        row: Mapping with optional ``available_regions`` and ``elements`` lists.
        id2label: Public id-to-label mapping used for integer validation.

    Returns:
        Normalized ``PosterORecord``.

    Raises:
        ValueError: If an integer label is not present in ``id2label``.
    """
    elements = [
        _element_from_mapping(element, id2label=id2label)
        for element in _sequence(row.get("elements", ()))
    ]
    regions = [
        AvailableRegion(bbox_ltrb=_bbox(_mapping(region).get("bbox_ltrb", region)))
        for region in _sequence(row.get("available_regions", ()))
    ]
    return PosterORecord(
        id=str(row["id"]),
        dataset=str(row.get("dataset", "")),
        poster_path=str(row["poster_path"]) if row.get("poster_path") else None,
        canvas_size=_canvas_size(row.get("canvas_size", (513, 750))),
        available_regions=regions,
        elements=elements,
        features=[_float(value) for value in _sequence(row.get("features", ()))],
        metrics={
            str(key): _float(value)
            for key, value in _mapping(row.get("metrics")).items()
        },
        metadata=dict(_mapping(row.get("metadata"))),
    )


def record_to_mapping(record: PosterORecord) -> dict[str, object]:
    """Serialize a record to a JSON-compatible mapping."""
    return record.model_dump(mode="json")


def labels_for_record(record: PosterORecord) -> list[int | str]:
    """Return element labels in record order."""
    return [element.label for element in record.elements]


def _element_from_mapping(
    value: object, *, id2label: Mapping[int, str]
) -> PosterLayoutElement:
    if isinstance(value, PosterLayoutElement):
        return value
    mapping = _mapping(value)
    label = mapping["label"]
    if isinstance(label, int) and label not in id2label:
        msg = f"Unknown label id for PosterO record: {label}"
        raise ValueError(msg)
    return PosterLayoutElement(
        label=label if isinstance(label, int) else str(label),
        bbox_ltrb=_bbox(mapping.get("bbox_ltrb", mapping.get("bbox"))),
        rotation=_float(mapping["rotation"])
        if mapping.get("rotation") is not None
        else None,
        path_data=str(mapping["path_data"]) if mapping.get("path_data") else None,
    )


def _bbox(value: object) -> tuple[float, float, float, float]:
    values = [_float(item) for item in _sequence(value)]
    if len(values) != 4:
        msg = "bbox values must contain four numbers"
        raise ValueError(msg)
    return (values[0], values[1], values[2], values[3])


def _canvas_size(value: object) -> tuple[int, int]:
    values = [int(_float(item)) for item in _sequence(value)]
    if len(values) != 2:
        msg = "canvas_size values must contain width and height"
        raise ValueError(msg)
    return (values[0], values[1])


def _float(value: object) -> float:
    return float(cast(float | int | str, value))


def _sequence(value: object) -> Sequence[object]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    msg = f"Expected a sequence, got {type(value).__name__}"
    raise TypeError(msg)


def _mapping(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    msg = f"Expected a mapping, got {type(value).__name__}"
    raise TypeError(msg)

"""Closed string modes used by the PosterO package."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import TypeVar

EnumT = TypeVar("EnumT", bound=StrEnum)


class PosterOStructure(StrEnum):
    """Prompt structure options."""

    plain = auto()
    hierarchical = auto()


class PosterOInjection(StrEnum):
    """Available-area injection options."""

    none = auto()
    top = auto()
    pulse = auto()
    pulse_wh = auto()


class PosterOPoolStrategy(StrEnum):
    """Candidate-pool construction strategies."""

    all = auto()
    metric_filter = auto()
    metric_describe = auto()
    metric_filter_describe = auto()


class PosterORankStrategy(StrEnum):
    """Exemplar ranking strategies."""

    random = auto()
    rank_by_label = auto()
    rank_by_denbox = auto()
    rank_by_feature = auto()


class OutputType(StrEnum):
    """Public return container names."""

    dataclass = auto()
    dict = auto()


class PromptTarget(StrEnum):
    """Structured response target requested from the language model."""

    rect_only = auto()
    generalized_svg = auto()


def coerce_enum(value: str | EnumT, enum_type: type[EnumT]) -> EnumT:
    """Normalize a public string-or-enum value.

    Args:
        value: String or enum value.
        enum_type: Expected enum type.

    Returns:
        Normalized enum value.

    Raises:
        ValueError: If the string does not belong to ``enum_type``.

    Examples:
        >>> coerce_enum("plain", PosterOStructure)
        <PosterOStructure.plain: 'plain'>
    """
    if isinstance(value, enum_type):
        return value
    return enum_type(value)

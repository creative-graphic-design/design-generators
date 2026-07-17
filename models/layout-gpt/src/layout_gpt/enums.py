"""Closed string modes used by LayoutGPT."""

from enum import StrEnum
from typing import TypeVar

EnumT = TypeVar("EnumT", bound=StrEnum)


class LayoutGPTSetting(StrEnum):
    """Supported NSR-1K LayoutGPT settings."""

    COUNTING = "counting"
    SPATIAL = "spatial"


class ICLType(StrEnum):
    """Supported in-context exemplar selection modes."""

    FIXED_RANDOM = "fixed-random"
    K_SIMILAR = "k-similar"


class ConditionType(StrEnum):
    """Public condition names accepted by the LayoutGPT agent."""

    TEXT = "text"
    UNCONDITIONAL = "unconditional"


class BoxFormat(StrEnum):
    """Public bounding-box format names."""

    XYWH = "xywh"
    LTWH = "ltwh"
    LTRB = "ltrb"


class OutputType(StrEnum):
    """Public return type names."""

    DATACLASS = "dataclass"
    DICT = "dict"


def coerce_enum(value: str | EnumT, enum_type: type[EnumT]) -> EnumT:
    """Convert a string or enum value into the requested StrEnum."""
    if isinstance(value, enum_type):
        return value
    return enum_type(value)

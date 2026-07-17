"""Closed string modes that are specific to LayoutGPT."""

from enum import StrEnum, auto
from typing import TypeVar

EnumT = TypeVar("EnumT", bound=StrEnum)


class LayoutGPTSetting(StrEnum):
    """Supported NSR-1K LayoutGPT settings."""

    counting = auto()
    spatial = auto()


class ICLType(StrEnum):
    """Supported in-context exemplar selection modes."""

    fixed_random = "fixed-random"
    k_similar = "k-similar"


class OutputType(StrEnum):
    """Public return type names."""

    dataclass = auto()
    dict = auto()


def coerce_enum(value: str | EnumT, enum_type: type[EnumT]) -> EnumT:
    """Convert a string or enum value into the requested StrEnum."""
    if isinstance(value, enum_type):
        return value
    return enum_type(value)

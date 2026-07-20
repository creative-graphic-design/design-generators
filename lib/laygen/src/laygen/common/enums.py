"""Shared enum normalization helpers."""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar

EnumT = TypeVar("EnumT", bound=StrEnum)


def normalize_enum_value(
    value: EnumT | str,
    enum_type: type[EnumT],
    *,
    option_name: str,
) -> EnumT:
    """Normalize a public string-or-enum option to a ``StrEnum`` value."""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported {option_name}: {value}") from exc

"""Serialization helpers for shared layout-generation metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, TypeAlias, cast  # noqa: TID251 - YAML sanitization accepts arbitrary mapping and sequence payloads.


YamlScalar: TypeAlias = str | int | float | bool | None
YamlValue: TypeAlias = YamlScalar | list["YamlValue"] | dict[YamlScalar, "YamlValue"]
YamlInputValue: TypeAlias = YamlScalar | Enum | Mapping[Any, Any] | Sequence[Any]


def sanitize_for_yaml(value: YamlInputValue) -> YamlValue:
    """Convert enum-rich metadata into objects accepted by ``yaml.safe_dump``.

    Args:
        value: Metadata value that may contain ``Enum`` instances, mappings,
            sequences, or dataclass instances.

    Returns:
        A recursively sanitized value containing only YAML-safe scalar and
        container types.

    Examples:
        >>> from laygen.common import DatasetName
        >>> sanitize_for_yaml({"dataset": DatasetName.rico25})
        {'dataset': 'rico25'}
    """
    if isinstance(value, Enum):
        return str(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return sanitize_for_yaml(cast(YamlInputValue, asdict(value)))
    if isinstance(value, Mapping):
        return cast(
            YamlValue,
            {
                sanitize_for_yaml(cast(YamlInputValue, key)): sanitize_for_yaml(
                    cast(YamlInputValue, item)
                )
                for key, item in value.items()
            },
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [sanitize_for_yaml(cast(YamlInputValue, item)) for item in value]
    return cast(YamlScalar, value)

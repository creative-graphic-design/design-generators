"""Closed public option sets for Coarse-to-Fine."""

from __future__ import annotations

from enum import StrEnum, auto

from laygen.common.enums import normalize_enum_value


class OutputType(StrEnum):
    """Supported output containers."""

    dataclass = auto()
    dict = auto()


def normalize_output_type(output_type: OutputType | str) -> OutputType:
    """Normalize a public output type value.

    Args:
        output_type: Enum value or string.

    Returns:
        Normalized output type.

    Raises:
        ValueError: If the output type is unknown.

    Examples:
        >>> str(normalize_output_type("dict"))
        'dict'
    """
    return normalize_enum_value(
        output_type,
        OutputType,
        option_name="output_type",
    )

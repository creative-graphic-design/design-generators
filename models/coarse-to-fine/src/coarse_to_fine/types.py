"""Closed public option sets for Coarse-to-Fine."""

from __future__ import annotations

from enum import StrEnum, auto


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
    if isinstance(output_type, OutputType):
        return output_type
    try:
        return OutputType(output_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported output_type: {output_type}") from exc

"""Path checks for RADM's original-code data mapper."""

from __future__ import annotations

import os
from os import PathLike


def validate_dot_free_data_root(value: str | PathLike[str]) -> str:
    """Make a visible data-root path absolute without resolving symlinks."""
    raw_path = os.fspath(value)
    if not raw_path:
        raise ValueError("RADM data root must not be empty")

    path = os.path.abspath(raw_path)
    if any("." in part for part in path.split(os.sep) if part):
        raise ValueError(f"RADM data root must have dot-free path components: {path!r}")

    return path

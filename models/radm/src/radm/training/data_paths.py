"""Path checks for RADM's original-code data mapper."""

from __future__ import annotations

import os
from os import PathLike


def validate_dot_free_data_root(value: str | PathLike[str]) -> str:
    """Preserve a visible data-root path after rejecting dotted components."""
    path = os.fspath(value)
    if not path or any("." in part for part in path.split(os.sep) if part):
        raise ValueError(
            f"RADM data root must have dot-free path components: {path!r}"
        )

    return path

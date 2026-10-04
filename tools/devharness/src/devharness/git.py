"""Shared best-effort Git execution for repository checkers."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path


def git_output(root: Path, command: Sequence[str]) -> str | None:
    """Return stdout for a successful Git command, or ``None`` on failure."""
    result = subprocess.run(
        command,
        check=False,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return None

    return result.stdout

"""Repository discovery helpers shared by development checkers."""

from __future__ import annotations

from pathlib import Path
import tomllib


def find_repo_root(start: Path | None = None) -> Path:
    """Find the nearest directory containing the workspace declaration."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        pyproject = candidate / "pyproject.toml"
        if not pyproject.is_file():
            continue

        try:
            document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            continue

        tool = document.get("tool")
        if not isinstance(tool, dict):
            continue

        uv = tool.get("uv")
        if not isinstance(uv, dict):
            continue

        workspace = uv.get("workspace")
        if isinstance(workspace, dict):
            return candidate

    raise ValueError(
        "Unable to find repository root with [tool.uv.workspace] in pyproject.toml"
    )

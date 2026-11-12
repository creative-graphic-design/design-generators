"""Workspace module paths used by the RADM CGL training and evaluation flow."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys


WORKSPACE_SOURCE_RELATIVE_ROOTS = {
    "radm": Path("models") / "radm" / "src",
    "traingen": Path("lib") / "traingen" / "src",
    "traingen_parity": Path("lib") / "traingen-parity" / "src",
    "laygen": Path("lib") / "laygen" / "src",
    "posgen": Path("lib") / "posgen" / "src",
}


def workspace_source_roots(root: Path) -> dict[str, Path]:
    """Return workspace source roots under the caller's checkout."""
    root = root.resolve()
    return {
        package: root / relative_root
        for package, relative_root in WORKSPACE_SOURCE_RELATIVE_ROOTS.items()
    }


def verify_workspace_sources(root: Path) -> dict[str, str]:
    """Require RADM and its workspace dependencies to resolve in this checkout."""
    source_paths: dict[str, str] = {}
    for package, source_root in workspace_source_roots(root).items():
        module = sys.modules.get(package)
        module_path = getattr(module, "__file__", None)
        if not isinstance(module_path, str):
            spec = find_spec(package)
            module_path = spec.origin if spec is not None else None

        if not module_path:
            raise RuntimeError(f"{package} cannot be resolved from this checkout")

        resolved_path = Path(module_path).resolve()
        if not resolved_path.is_relative_to(source_root.resolve()):
            raise RuntimeError(
                f"{package} resolves outside repository source tree: {resolved_path}"
            )

        source_paths[package] = str(resolved_path)

    return source_paths

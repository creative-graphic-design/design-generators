"""Workspace module paths used by the RADM CGL training and evaluation flow."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[5]
WORKSPACE_SOURCE_ROOTS = {
    "radm": ROOT / "models" / "radm" / "src",
    "traingen": ROOT / "lib" / "traingen" / "src",
    "traingen_parity": ROOT / "lib" / "traingen-parity" / "src",
    "laygen": ROOT / "lib" / "laygen" / "src",
    "posgen": ROOT / "lib" / "posgen" / "src",
}


def verify_workspace_sources() -> dict[str, str]:
    """Require RADM and its workspace dependencies to resolve in this checkout."""
    source_paths: dict[str, str] = {}
    for package, source_root in WORKSPACE_SOURCE_ROOTS.items():
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

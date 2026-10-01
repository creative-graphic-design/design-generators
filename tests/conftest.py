from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_repo_checks() -> ModuleType:
    """Load the private checker package for import-mode=importlib tests."""
    package_dir = Path(__file__).resolve().parents[1] / "scripts" / "_repo_checks"
    spec = importlib.util.spec_from_file_location(
        "_repo_checks",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None
    assert spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return package


_load_repo_checks()

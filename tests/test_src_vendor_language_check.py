from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType


def load_check_src_vendor_language() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_src_vendor_language.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_src_vendor_language", module_path
    )
    assert spec is not None
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_src_vendor_language = load_check_src_vendor_language()


def write_source(root: Path, text: str, name: str = "example.py") -> Path:
    path = root / "models" / "posterllama" / "src" / "posterllama" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_current_entries_detects_vendor_in_identifiers_and_text(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        '''
VENDOR_CONDITION_ALIASES = {}

def normalize_vendor_condition_aliases() -> None:
    """Reject vendor wording in docstrings."""
    # vendor wording in comments is also rejected.
''',
    )

    entries = check_src_vendor_language.current_entries(tmp_path)

    assert (
        "models/posterllama/src/posterllama/example.py\t1\t"
        "VENDOR_CONDITION_ALIASES = {}"
    ) in entries
    assert (
        "models/posterllama/src/posterllama/example.py\t1\t"
        "def normalize_vendor_condition_aliases() -> None:"
    ) in entries
    assert (
        "models/posterllama/src/posterllama/example.py\t1\t"
        '"""Reject vendor wording in docstrings."""'
    ) in entries
    assert (
        "models/posterllama/src/posterllama/example.py\t1\t"
        "# vendor wording in comments is also rejected."
    ) in entries


def test_source_files_excludes_conversion_and_vendor_state_modules(
    tmp_path: Path,
) -> None:
    included = write_source(tmp_path, "")
    write_source(tmp_path, "VENDOR = True\n", name="conversion_posterllama.py")
    write_source(tmp_path, "VENDOR = True\n", name="vendor_state.py")

    assert check_src_vendor_language.source_files(tmp_path) == [included]

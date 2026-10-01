from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest


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


def test_write_baseline_preserves_empty_source_vendor_byte_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "src_vendor_language_baseline.txt"
    monkeypatch.setattr(check_src_vendor_language, "BASELINE_PATH", baseline)
    monkeypatch.setattr(check_src_vendor_language, "current_entries", lambda: set())
    monkeypatch.setattr(
        sys, "argv", ["check_src_vendor_language.py", "--write-baseline"]
    )

    assert check_src_vendor_language.main() == 0
    assert baseline.read_bytes() == b"\n"

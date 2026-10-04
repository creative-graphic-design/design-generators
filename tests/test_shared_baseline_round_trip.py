from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest

from devharness.baselines import read_entry_baseline, write_entry_baseline
from devharness.checks.jaxtyping_annotations.constants import baseline_paths


ROOT = Path(__file__).resolve().parents[1]


def load_checker(name: str) -> ModuleType:
    module_path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_module_naming = load_checker("check_module_naming")
check_training_doc_template = load_checker("check_training_doc_template")
check_training_stage_evidence = load_checker("check_training_stage_evidence")


SHARED_WRITER_BASELINES = (
    *baseline_paths(ROOT),
    check_module_naming.BASELINE_PATH,
    check_training_doc_template.BASELINE_PATH,
    check_training_stage_evidence.BASELINE_PATH,
)


@pytest.mark.parametrize("baseline_path", SHARED_WRITER_BASELINES)
def test_shared_writer_baseline_round_trip(baseline_path: Path, tmp_path: Path) -> None:
    original = baseline_path.read_bytes()
    rewritten = tmp_path / baseline_path.name
    rewritten.write_bytes(original)

    write_entry_baseline(rewritten, read_entry_baseline(baseline_path))

    assert rewritten.read_bytes() == original

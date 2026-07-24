from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest


def load_check_jaxtyping_annotations() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check_jaxtyping_annotations.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_jaxtyping_annotations", module_path
    )
    assert spec is not None
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_jaxtyping_annotations = load_check_jaxtyping_annotations()


def write_source(root: Path, text: str) -> Path:
    path = root / "models" / "layout-dm" / "src" / "layout_dm" / "example.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_current_entries_detects_raw_annotations_only(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        """
from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Float

"docstring mentions torch.Tensor"
# comment mentions np.ndarray

def ok(x: Float[torch.Tensor, "batch channels"]) -> None:
    if isinstance(x, torch.Tensor):
        pass

def bad(
    x: torch.Tensor,
    y: torch.Tensor | None,
    z: Optional[np.ndarray],
) -> tuple[torch.Tensor, np.ndarray]:
    literal: "torch.Tensor" = "ignored"
    local: np.ndarray
    return x, z
""",
    )

    entries = check_jaxtyping_annotations.current_entries(tmp_path)

    assert entries == {
        "models/layout-dm/src/layout_dm/example.py\t1\tOptional[np.ndarray]",
        "models/layout-dm/src/layout_dm/example.py\t1\tnp.ndarray",
        "models/layout-dm/src/layout_dm/example.py\t1\ttorch.Tensor",
        "models/layout-dm/src/layout_dm/example.py\t1\ttorch.Tensor | None",
        "models/layout-dm/src/layout_dm/example.py\t1\ttuple[torch.Tensor, np.ndarray]",
    }


def test_check_fails_on_new_raw_annotation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_jaxtyping_annotations, "baseline_reference_entries", lambda *_: None
    )
    write_source(
        tmp_path,
        """
import torch

def bad(x: torch.Tensor) -> None:
    pass
""",
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("", encoding="utf-8")

    assert (
        check_jaxtyping_annotations.check_jaxtyping_annotations(tmp_path, baseline) == 1
    )

    stderr = capsys.readouterr().err
    assert "New raw tensor/ndarray annotations" in stderr
    assert "torch.Tensor" in stderr


def test_check_rejects_baseline_growth_for_new_annotation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_jaxtyping_annotations, "baseline_reference_entries", lambda *_: set()
    )
    write_source(
        tmp_path,
        """
import torch

def bad(x: torch.Tensor) -> None:
    pass
""",
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(
        "models/layout-dm/src/layout_dm/example.py\t1\ttorch.Tensor\n",
        encoding="utf-8",
    )

    assert (
        check_jaxtyping_annotations.check_jaxtyping_annotations(tmp_path, baseline) == 1
    )

    stderr = capsys.readouterr().err
    assert "New jaxtyping baseline entries" in stderr


def test_check_passes_when_baseline_matches_or_shrinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_source(
        tmp_path,
        """
import torch

def bad(x: torch.Tensor) -> None:
    pass
""",
    )
    baseline = tmp_path / "baseline.txt"
    check_jaxtyping_annotations.write_baseline(
        baseline,
        check_jaxtyping_annotations.current_entries(tmp_path),
    )
    base_entries = check_jaxtyping_annotations.baseline_entries(baseline)
    monkeypatch.setattr(
        check_jaxtyping_annotations,
        "baseline_reference_entries",
        lambda *_: base_entries,
    )

    assert (
        check_jaxtyping_annotations.check_jaxtyping_annotations(tmp_path, baseline) == 0
    )

    write_source(
        tmp_path,
        """
import torch
from jaxtyping import Float

def fixed(x: Float[torch.Tensor, "batch"]) -> None:
    pass
""",
    )
    baseline.write_text("", encoding="utf-8")

    assert (
        check_jaxtyping_annotations.check_jaxtyping_annotations(tmp_path, baseline) == 0
    )

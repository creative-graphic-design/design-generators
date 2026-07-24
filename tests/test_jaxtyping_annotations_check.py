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
import numpy.typing as npt
import torch
import torch as t
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

"docstring mentions torch.Tensor"
# comment mentions np.ndarray

def ok(x: Float[torch.Tensor, "batch channels"]) -> None:
    if isinstance(x, torch.Tensor):
        pass

def bad(
    x: torch.Tensor,
    imported: Tensor,
    alias: t.Tensor,
    y: torch.Tensor | None,
    z: Optional[np.ndarray],
    arr: npt.NDArray,
    direct_arr: numpy.typing.NDArray,
    imported_arr: NDArray,
    quoted: "torch.Tensor",
) -> tuple[torch.Tensor, np.ndarray]:
    literal: "torch.Tensor" = "ignored"
    local: np.ndarray
    return x, z
""",
    )

    entries = check_jaxtyping_annotations.current_entries(tmp_path)

    assert entries == {
        "models/layout-dm/src/layout_dm/example.py\tOptional[np.ndarray]\tz: Optional[np.ndarray],",
        "models/layout-dm/src/layout_dm/example.py\tTensor\timported: Tensor,",
        "models/layout-dm/src/layout_dm/example.py\tNDArray\timported_arr: NDArray,",
        "models/layout-dm/src/layout_dm/example.py\tnumpy.typing.NDArray\tdirect_arr: numpy.typing.NDArray,",
        "models/layout-dm/src/layout_dm/example.py\tnp.ndarray\tlocal: np.ndarray",
        "models/layout-dm/src/layout_dm/example.py\tnpt.NDArray\tarr: npt.NDArray,",
        "models/layout-dm/src/layout_dm/example.py\tt.Tensor\talias: t.Tensor,",
        'models/layout-dm/src/layout_dm/example.py\ttorch.Tensor\tliteral: "torch.Tensor" = "ignored"',
        'models/layout-dm/src/layout_dm/example.py\ttorch.Tensor\tquoted: "torch.Tensor",',
        "models/layout-dm/src/layout_dm/example.py\ttorch.Tensor\tx: torch.Tensor,",
        "models/layout-dm/src/layout_dm/example.py\ttorch.Tensor | None\ty: torch.Tensor | None,",
        "models/layout-dm/src/layout_dm/example.py\ttuple[torch.Tensor, np.ndarray]\t) -> tuple[torch.Tensor, np.ndarray]:",
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
        "models/layout-dm/src/layout_dm/example.py\ttorch.Tensor\tdef bad(x: torch.Tensor) -> None:\n",
        encoding="utf-8",
    )

    assert (
        check_jaxtyping_annotations.check_jaxtyping_annotations(tmp_path, baseline) == 1
    )

    stderr = capsys.readouterr().err
    assert "New jaxtyping baseline entries" in stderr


def test_line_based_baseline_rejects_same_annotation_swap(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source(
        tmp_path,
        """
import torch

def old(x: torch.Tensor) -> None:
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

    write_source(
        tmp_path,
        """
import torch

def new_unrelated_api(y: torch.Tensor) -> None:
    pass
""",
    )

    assert (
        check_jaxtyping_annotations.check_jaxtyping_annotations(tmp_path, baseline) == 1
    )

    stderr = capsys.readouterr().err
    assert "New raw tensor/ndarray annotations" in stderr
    assert "new_unrelated_api" in stderr


def test_existing_baseline_rejects_new_annotation_and_baseline_entry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source(
        tmp_path,
        """
import torch

def added_in_pr(x: torch.Tensor) -> None:
    pass
""",
    )
    baseline = tmp_path / "baseline.txt"
    check_jaxtyping_annotations.write_baseline(
        baseline,
        check_jaxtyping_annotations.current_entries(tmp_path),
    )
    monkeypatch.setattr(
        check_jaxtyping_annotations, "baseline_reference_entries", lambda *_: set()
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

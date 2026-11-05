from __future__ import annotations

import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from devharness import baselines
from devharness.checks.jaxtyping_annotations import (
    baseline as baseline_checks,
    check as jaxtyping_check,
    object_annotations,
    raw_annotations,
    shaped_aliases,
    weak_cast_types,
)
from devharness.checks.jaxtyping_annotations.constants import baseline_paths


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_jaxtyping_cli(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PATH"] = (
        f"{Path(sys.executable).parent}{os.pathsep}{environment['PATH']}"
    )
    return subprocess.run(
        ["devharness", "check", "jaxtyping-annotations", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _archive_fixture(tmp_path: Path) -> Path:
    archive = subprocess.run(
        ["git", "archive", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    fixture = tmp_path / "repository"
    fixture.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tar:
        tar.extractall(fixture)
    return fixture


def _initialize_git_fixture(fixture: Path) -> None:
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch", "main"],
        cwd=fixture,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "devharness test"],
        cwd=fixture,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "devharness@example.invalid"],
        cwd=fixture,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=fixture, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "--message", "fixture"],
        cwd=fixture,
        check=True,
    )
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"],
        cwd=fixture,
        check=True,
    )


def test_baseline_reference_entries_reads_merge_base_and_show_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "scripts" / "baseline.txt"
    baseline.parent.mkdir()
    baseline.write_text("working-entry\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_git_output(root: Path, command: list[str]) -> str | None:
        del root
        calls.append(command)
        if command[:2] == ["git", "merge-base"]:
            return "merge-base-sha\n"
        return "# comment\nreference-entry\n\n"

    monkeypatch.setattr(baseline_checks, "git_output", fake_git_output)

    assert baseline_checks.baseline_reference_entries(tmp_path, baseline) == {
        "reference-entry"
    }
    assert calls == [
        ["git", "merge-base", "origin/main", "HEAD"],
        ["git", "show", "merge-base-sha:scripts/baseline.txt"],
    ]

    monkeypatch.setattr(baseline_checks, "git_output", lambda *_: None)

    assert baseline_checks.baseline_reference_entries(tmp_path, baseline) is None


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

    entries = raw_annotations.current_entries(tmp_path)

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


def test_current_alias_entries_detects_module_level_jaxtyping_aliases_only(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        """
from __future__ import annotations

from typing import Literal, TypeAlias
import typing as typ

import torch
from jaxtyping import Float, Int
import jaxtyping as jt

Mode: TypeAlias = Literal["x"]
PublicBbox = Float[torch.Tensor, "batch 4"]
PublicLabels: TypeAlias = Int[torch.Tensor, "batch"]
Qualified: typ.TypeAlias = jt.Float[torch.Tensor, "batch"]

class Output:
    bbox: Float[torch.Tensor, "batch 4"]

def make_alias() -> None:
    LocalBbox = Float[torch.Tensor, "batch 4"]
""",
    )

    entries = shaped_aliases.current_entries(tmp_path)

    assert entries == {
        "models/layout-dm/src/layout_dm/example.py\tPublicBbox\tFloat[torch.Tensor, 'batch 4']\tPublicBbox = Float[torch.Tensor, \"batch 4\"]",
        "models/layout-dm/src/layout_dm/example.py\tPublicLabels\tInt[torch.Tensor, 'batch']\tPublicLabels: TypeAlias = Int[torch.Tensor, \"batch\"]",
        "models/layout-dm/src/layout_dm/example.py\tQualified\tjt.Float[torch.Tensor, 'batch']\tQualified: typ.TypeAlias = jt.Float[torch.Tensor, \"batch\"]",
    }


def test_current_object_entries_detects_function_signature_object_annotations(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        """
from __future__ import annotations

from builtins import object as builtin_object
from typing import TypeAlias

Metadata: TypeAlias = dict[str, object]

class Output:
    intermediates: object | None = None

def ok(**kwargs: object) -> None:
    local: object = kwargs

def bad(
    value: object,
    values: list[object],
    alias: builtin_object,
    *args: object,
    **kwargs: object,
) -> dict[str, object]:
    return {"value": value}
""",
    )

    entries = object_annotations.current_entries(tmp_path)

    assert entries == {
        "models/layout-dm/src/layout_dm/example.py\tbuiltin_object\talias: builtin_object,",
        "models/layout-dm/src/layout_dm/example.py\tdict[str, object]\t) -> dict[str, object]:",
        "models/layout-dm/src/layout_dm/example.py\tlist[object]\tvalues: list[object],",
        "models/layout-dm/src/layout_dm/example.py\tobject\t*args: object,",
        "models/layout-dm/src/layout_dm/example.py\tobject\t**kwargs: object,",
        "models/layout-dm/src/layout_dm/example.py\tobject\tdef ok(**kwargs: object) -> None:",
        "models/layout-dm/src/layout_dm/example.py\tobject\tvalue: object,",
    }


def test_current_object_entries_detects_type_checking_object_fallbacks(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        """
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

if TYPE_CHECKING:
    PublicBbox = Float[torch.Tensor, "batch 4"]
else:
    PublicBbox = object

RuntimeValue = object
lowercase_value = object
""",
    )

    entries = object_annotations.current_entries(tmp_path)

    assert entries == {
        "models/layout-dm/src/layout_dm/example.py\tobject\tPublicBbox = object",
    }


def test_current_weak_cast_entries_detects_nested_object_and_any_casts(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        """
from __future__ import annotations

from builtins import object as builtin_object
from collections.abc import Mapping, Sequence
from typing import Any, cast
import typing as typ

def example(value: object) -> None:
    cast(Sequence[object], value)
    cast(Mapping[str, Any] | None, value)
    typ.cast(list[builtin_object], value)
    cast(str | int, value)
""",
    )

    entries = weak_cast_types.current_entries(tmp_path)

    assert entries == {
        "models/layout-dm/src/layout_dm/example.py\tMapping[str, Any] | None\tcast(Mapping[str, Any] | None, value)",
        "models/layout-dm/src/layout_dm/example.py\tSequence[object]\tcast(Sequence[object], value)",
        "models/layout-dm/src/layout_dm/example.py\tlist[builtin_object]\ttyp.cast(list[builtin_object], value)",
    }


def test_check_fails_on_new_raw_annotation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baseline_checks, "baseline_reference_entries", lambda *_: None)
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

    assert baseline_checks.check_jaxtyping_annotations(tmp_path, baseline) == 1

    stderr = capsys.readouterr().err
    assert "New raw tensor/ndarray annotations" in stderr
    assert "torch.Tensor" in stderr


def test_check_rejects_baseline_growth_for_new_annotation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baseline_checks, "baseline_reference_entries", lambda *_: set())
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

    assert baseline_checks.check_jaxtyping_annotations(tmp_path, baseline) == 1

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
    baselines.write_entry_baseline(
        baseline,
        raw_annotations.current_entries(tmp_path),
    )
    base_entries = baselines.read_entry_baseline(baseline)
    monkeypatch.setattr(
        baseline_checks,
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

    assert baseline_checks.check_jaxtyping_annotations(tmp_path, baseline) == 1

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
    baselines.write_entry_baseline(
        baseline,
        raw_annotations.current_entries(tmp_path),
    )
    monkeypatch.setattr(baseline_checks, "baseline_reference_entries", lambda *_: set())

    assert baseline_checks.check_jaxtyping_annotations(tmp_path, baseline) == 1

    stderr = capsys.readouterr().err
    assert "New jaxtyping baseline entries" in stderr


def test_check_fails_on_new_jaxtyping_alias(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baseline_checks, "baseline_reference_entries", lambda *_: None)
    write_source(
        tmp_path,
        """
from typing import TypeAlias

import torch
from jaxtyping import Float

PublicBbox: TypeAlias = Float[torch.Tensor, "batch 4"]
""",
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("", encoding="utf-8")

    assert baseline_checks.check_jaxtyping_aliases(tmp_path, baseline) == 1

    stderr = capsys.readouterr().err
    assert "New jaxtyping shaped-type aliases" in stderr
    assert "PublicBbox" in stderr


def test_check_fails_on_new_object_annotation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baseline_checks, "baseline_reference_entries", lambda *_: None)
    write_source(
        tmp_path,
        """
def bad(value: list[object]) -> None:
    pass
""",
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("", encoding="utf-8")

    assert baseline_checks.check_object_annotations(tmp_path, baseline) == 1

    stderr = capsys.readouterr().err
    assert "New object annotations in function signatures" in stderr
    assert "list[object]" in stderr


def test_check_fails_on_new_weak_cast_type(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baseline_checks, "baseline_reference_entries", lambda *_: None)
    write_source(
        tmp_path,
        """
from collections.abc import Sequence
from typing import cast

def bad(value: object) -> None:
    cast(Sequence[object], value)
""",
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("", encoding="utf-8")

    assert baseline_checks.check_weak_cast_types(tmp_path, baseline) == 1

    stderr = capsys.readouterr().err
    assert "New bare object/Any references in cast target types" in stderr
    assert "Sequence[object]" in stderr


def test_check_rejects_baseline_growth_for_new_jaxtyping_alias(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baseline_checks, "baseline_reference_entries", lambda *_: set())
    write_source(
        tmp_path,
        """
import torch
from jaxtyping import Float

PublicBbox = Float[torch.Tensor, "batch 4"]
""",
    )
    baseline = tmp_path / "baseline.txt"
    baselines.write_entry_baseline(
        baseline,
        shaped_aliases.current_entries(tmp_path),
    )

    assert baseline_checks.check_jaxtyping_aliases(tmp_path, baseline) == 1

    stderr = capsys.readouterr().err
    assert "New jaxtyping alias baseline entries" in stderr


def test_check_rejects_baseline_growth_for_new_object_annotation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baseline_checks, "baseline_reference_entries", lambda *_: set())
    write_source(
        tmp_path,
        """
def bad(value: object) -> None:
    pass
""",
    )
    baseline = tmp_path / "baseline.txt"
    baselines.write_entry_baseline(
        baseline,
        object_annotations.current_entries(tmp_path),
    )

    assert baseline_checks.check_object_annotations(tmp_path, baseline) == 1

    stderr = capsys.readouterr().err
    assert "New object annotation baseline entries" in stderr


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
    baselines.write_entry_baseline(
        baseline,
        raw_annotations.current_entries(tmp_path),
    )
    base_entries = baselines.read_entry_baseline(baseline)
    monkeypatch.setattr(
        baseline_checks,
        "baseline_reference_entries",
        lambda *_: base_entries,
    )

    assert baseline_checks.check_jaxtyping_annotations(tmp_path, baseline) == 0

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

    assert baseline_checks.check_jaxtyping_annotations(tmp_path, baseline) == 0


def test_jaxtyping_cli_rewrites_baselines_and_accepts_shrink(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    raw_path = baseline_paths(fixture)[0]
    raw_path.write_text(
        "models/layout-dm/src/layout_dm/removed.py\ttorch.Tensor\t"
        "def removed(value: torch.Tensor) -> None:\n",
        encoding="utf-8",
    )
    _initialize_git_fixture(fixture)

    expected_baselines = [path.read_bytes() for path in baseline_paths(REPO_ROOT)]

    write_result = _run_jaxtyping_cli(fixture, "--write-baseline")

    assert write_result.returncode == 0
    assert write_result.stdout == ""
    assert write_result.stderr == ""
    assert [path.read_bytes() for path in baseline_paths(fixture)] == expected_baselines

    check_result = _run_jaxtyping_cli(fixture)

    assert check_result.returncode == 0
    assert check_result.stdout == ""
    assert check_result.stderr == ""


def test_jaxtyping_cli_reports_new_annotation_with_exact_diagnostic(
    tmp_path: Path,
) -> None:
    fixture = _archive_fixture(tmp_path)
    violation = write_source(
        fixture,
        """
import torch

def bad(value: torch.Tensor) -> None:
    pass
""",
    )

    result = _run_jaxtyping_cli(fixture)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == (
        "New raw tensor/ndarray annotations in package source:\n"
        f"  + {violation.relative_to(fixture).as_posix()}\t"
        "torch.Tensor\tdef bad(value: torch.Tensor) -> None:\n"
    )


def test_jaxtyping_cli_passes_without_git_reference(tmp_path: Path) -> None:
    fixture = _archive_fixture(tmp_path)

    result = _run_jaxtyping_cli(fixture)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_jaxtyping_cli_reports_missing_workspace_root() -> None:
    result = _run_jaxtyping_cli(Path("/proc"))

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == (
        "Unable to find repository root with [tool.uv.workspace] in pyproject.toml\n"
    )


def test_jaxtyping_main_runs_from_a_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for path in baseline_paths(tmp_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jaxtyping_check, "find_repo_root", lambda _: tmp_path)

    assert jaxtyping_check.main([]) == 0
    assert jaxtyping_check.main(["--write-baseline"]) == 0

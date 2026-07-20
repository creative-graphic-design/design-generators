from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
from subprocess import CompletedProcess
from types import ModuleType

import pytest


def load_check_duplicate_code() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_duplicate_code.py"
    )
    spec = importlib.util.spec_from_file_location("check_duplicate_code", module_path)
    assert spec is not None
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_duplicate_code = load_check_duplicate_code()


def test_is_python_target_accepts_workspace_python_files() -> None:
    assert check_duplicate_code.is_python_target("lib/laygen/src/laygen/foo.py")
    assert check_duplicate_code.is_python_target(
        "models/layout-dm/src/layout_dm/foo.py"
    )
    assert check_duplicate_code.is_python_target("scripts/check_duplicate_code.py")
    assert not check_duplicate_code.is_python_target("vendor/example.py")
    assert not check_duplicate_code.is_python_target("README.md")


def test_python_targets_discovers_workspace_python_files(tmp_path: Path) -> None:
    (tmp_path / "lib" / "pkg").mkdir(parents=True)
    (tmp_path / "lib" / "pkg" / "foo.py").write_text("", encoding="utf-8")
    (tmp_path / "models" / "model").mkdir(parents=True)
    (tmp_path / "models" / "model" / "bar.py").write_text("", encoding="utf-8")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "copied.py").write_text("", encoding="utf-8")

    assert check_duplicate_code.python_targets(tmp_path) == [
        "lib/pkg/foo.py",
        "models/model/bar.py",
    ]


def test_changed_python_targets_reads_branch_and_local_diffs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = {
        ("git", "diff", "--name-only", "--diff-filter=ACMR", "origin/main...HEAD"): (
            "lib/laygen/src/laygen/foo.py\nREADME.md\n"
        ),
        ("git", "diff", "--name-only", "--diff-filter=ACMR"): (
            "scripts/check_duplicate_code.py\n"
        ),
        ("git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"): (
            "vendor/copied.py\nmodels/layout-dm/src/layout_dm/bar.py\n"
        ),
        ("git", "ls-files", "--others", "--exclude-standard"): "scripts/new_tool.py\n",
    }

    def fake_run(
        command: list[str],
        *,
        check: bool,
        cwd: Path,
        stdout: int,
        stderr: int,
        text: bool,
    ) -> CompletedProcess[str]:
        assert check is False
        assert cwd == tmp_path
        assert stdout == check_duplicate_code.subprocess.PIPE
        assert stderr == check_duplicate_code.subprocess.DEVNULL
        assert text is True
        return CompletedProcess(command, 0, outputs[tuple(command)])

    monkeypatch.setattr(check_duplicate_code.subprocess, "run", fake_run)

    assert check_duplicate_code.changed_python_targets(tmp_path) == [
        "lib/laygen/src/laygen/foo.py",
        "models/layout-dm/src/layout_dm/bar.py",
        "scripts/check_duplicate_code.py",
        "scripts/new_tool.py",
    ]


def test_module_labels_for_path_returns_pylint_variants() -> None:
    assert check_duplicate_code.module_labels_for_path(
        "models/layout-dm/src/layout_dm/pipeline.py"
    ) == {
        "pipeline",
        "models.layout-dm.src.layout_dm.pipeline",
        "layout_dm.pipeline",
        "src.layout_dm.pipeline",
    }


def test_build_command_uses_duplicate_code_only() -> None:
    assert check_duplicate_code.build_command(["lib", "models"]) == [
        "pylint",
        "lib",
        "models",
        "--disable=all",
        "--enable=duplicate-code",
        "--ignore=vendor",
    ]


def test_parse_duplicate_blocks_splits_pylint_reports() -> None:
    output = """************* Module example
foo.py:1:0: R0801: Similar lines in 2 files
==pkg.a:[1:3]
    a = 1
==pkg.b:[4:6]
    a = 1 (duplicate-code)
bar.py:2:0: R0801: Similar lines in 2 files
==pkg.c:[10:12]
    b = 2
==pkg.d:[20:22]
    b = 2 (duplicate-code)

------------------------------------------------------------------
Your code has been rated at 9.99/10
"""

    blocks = check_duplicate_code.parse_duplicate_blocks(output)

    assert len(blocks) == 2
    assert "==pkg.a:[1:3]" in blocks[0]
    assert "==pkg.c:[10:12]" in blocks[1]


def test_block_module_labels_extracts_report_modules() -> None:
    block = """foo.py:1:0: R0801: Similar lines in 2 files
==pkg.a:[1:3]
    a = 1
==pkg.b:[4:6]
    a = 1 (duplicate-code)"""

    assert check_duplicate_code.block_module_labels(block) == {"pkg.a", "pkg.b"}


def test_label_matches_module_suffixes() -> None:
    assert check_duplicate_code.label_matches("layout_dm.pipeline", "pipeline")
    assert check_duplicate_code.label_matches(
        "layout_dm.pipeline", "layout_dm.pipeline"
    )
    assert check_duplicate_code.label_matches(
        "src.layout_dm.pipeline", "layout_dm.pipeline"
    )
    assert not check_duplicate_code.label_matches(
        "layout_dm.pipeline", "other.pipeline"
    )


def test_accepted_duplicate_block_allows_non_src_pair() -> None:
    block = """foo.py:1:0: R0801: Similar lines in 2 files
==generate_reference_outputs:[1:3]
    a = 1
==test_layout_flow_parity:[4:6]
    a = 1 (duplicate-code)"""

    assert check_duplicate_code.is_accepted_duplicate_block(block)


def test_accepted_duplicate_pairs_reject_src_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_duplicate_code,
        "ACCEPTED_DUPLICATE_PAIRS",
        (
            (
                "models/ds-gan/src/ds_gan/processing_ds_gan.py",
                "models/layout-dm/tests/test_processing.py",
            ),
        ),
    )

    with pytest.raises(AssertionError, match="must not include src"):
        check_duplicate_code._accepted_duplicate_pairs()


def test_main_returns_success_when_no_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_duplicate_code, "python_targets", lambda root: [])
    assert check_duplicate_code.main([]) == 0


def test_check_duplicate_code_fails_when_changed_module_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    block = """foo.py:1:0: R0801: Similar lines in 2 files
==layout_dm.pipeline:[1:3]
    a = 1
==pkg.b:[4:6]
    a = 1 (duplicate-code)"""

    monkeypatch.setattr(
        check_duplicate_code, "python_targets", lambda root: ["lib/foo.py"]
    )
    monkeypatch.setattr(
        check_duplicate_code,
        "changed_python_targets",
        lambda root: ["models/layout-dm/src/layout_dm/pipeline.py"],
    )
    monkeypatch.setattr(
        check_duplicate_code,
        "run_pylint",
        lambda root, targets: CompletedProcess(
            check_duplicate_code.build_command(targets),
            8,
            stdout=block,
            stderr="",
        ),
    )

    assert check_duplicate_code.check_duplicate_code(tmp_path) == 1


def test_check_duplicate_code_passes_when_only_unchanged_modules_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    block = """foo.py:1:0: R0801: Similar lines in 2 files
==pkg.a:[1:3]
    a = 1
==pkg.b:[4:6]
    a = 1 (duplicate-code)"""
    monkeypatch.setattr(
        check_duplicate_code, "python_targets", lambda root: ["lib/foo.py"]
    )
    monkeypatch.setattr(
        check_duplicate_code,
        "changed_python_targets",
        lambda root: ["models/layout-dm/src/layout_dm/pipeline.py"],
    )
    monkeypatch.setattr(
        check_duplicate_code,
        "run_pylint",
        lambda root, targets: CompletedProcess(
            check_duplicate_code.build_command(targets),
            8,
            stdout=block,
            stderr="",
        ),
    )

    assert check_duplicate_code.check_duplicate_code(tmp_path) == 0

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_checker() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check_semantic_blank_lines.py"
    )
    spec = importlib.util.spec_from_file_location("check_semantic_blank_lines", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load check_semantic_blank_lines.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = load_checker()


def write_source(root: Path, text: str) -> None:
    path = root / "models" / "example" / "src" / "example" / "module.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_baseline(path: Path, entries: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Baseline entries for tests.\n"
        + "\n".join(sorted(entries))
        + ("\n" if entries else ""),
        encoding="utf-8",
    )


def initialize_git_repository(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.name", "semantic-checker-test"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "semantic-checker-test@example.com"],
        cwd=root,
        check=True,
    )


def test_heuristic_flags_more_than_fifteen_statement_lines(tmp_path: Path) -> None:
    statements = "\n".join(f"    value_{index} = {index}" for index in range(16))
    write_source(tmp_path, f"def dense():\n{statements}\n")

    entries = checker.current_entries(tmp_path)

    assert any("heuristic" in entry for entry in entries)


def test_heuristic_ignores_nested_class_body(tmp_path: Path) -> None:
    assignments = "\n".join(f"        value_{index} = {index}" for index in range(16))
    write_source(
        tmp_path,
        f"def container():\n    class Nested:\n{assignments}\n    return Nested\n",
    )

    entries = checker.current_entries(tmp_path)

    assert not any("heuristic" in entry for entry in entries)


def test_exact_raise_rule_requires_blank_line_before_ordinary_code(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        """
def missing():
    if True:
        raise ValueError(
            "failure"
        )
    return 1
""",
    )

    entries = checker.current_entries(tmp_path)

    assert any("raise-block" in entry for entry in entries)


def test_exact_raise_rule_requires_blank_line_across_comment_and_dedent(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        """
def missing():
    if True:
        raise ValueError()
    # The return is ordinary code after the raise block.
    return 1
""",
    )

    entries = checker.current_entries(tmp_path)

    assert any("raise-block" in entry for entry in entries)


def test_exact_raise_rule_allows_blank_line_before_ordinary_code(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        """
def present():
    raise ValueError()

    return 1
""",
    )

    entries = checker.current_entries(tmp_path)

    assert not any("raise-block" in entry for entry in entries)


def test_exact_raise_rule_exempts_block_continuations_and_terminal_raise(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        """
def continuations(value):
    try:
        raise ValueError()
    except ValueError:
        pass

    try:
        raise ValueError()
    finally:
        pass

    if value == 1:
        raise ValueError()
    elif value == 2:
        pass
    else:
        pass

    match value:
        case 1:
            raise ValueError()
        case _:
            pass


def terminal():
    raise ValueError()
""",
    )

    entries = checker.current_entries(tmp_path)

    assert not any("raise-block" in entry for entry in entries)


def test_case_identifiers_are_ordinary_code(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        """
def assignment():
    raise ValueError()
    case = 1


def call():
    raise ValueError()
    case(value)


def attribute():
    raise ValueError()
    case.value = 1
""",
    )

    entries = checker.current_entries(tmp_path)

    assert sum("raise-block" in entry for entry in entries) == 3


def test_checker_reports_entries_not_in_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_source(
        tmp_path,
        """
def missing():
    raise ValueError()
    return 1
""",
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("", encoding="utf-8")

    assert checker.check_semantic_blank_lines(tmp_path, baseline) == 1
    assert "raise-block" in capsys.readouterr().err


def test_checker_rejects_baseline_entries_absent_from_current_scan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_source(tmp_path, "def terminal():\n    raise ValueError()\n")
    baseline = tmp_path / "baseline.txt"
    stale_entry = "models/example/src/example/module.py\t99\theuristic\tstale"
    baseline.write_text(stale_entry + "\n", encoding="utf-8")

    assert checker.check_semantic_blank_lines(tmp_path, baseline) == 1
    assert "Baseline entries absent from current scan" in capsys.readouterr().err


def test_checker_rejects_adding_fresh_violation_to_committed_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "scripts" / "semantic_blank_lines_baseline.txt"
    write_baseline(baseline, set())
    write_source(tmp_path, "def terminal():\n    raise ValueError()\n")
    initialize_git_repository(tmp_path)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "baseline"],
        cwd=tmp_path,
        check=True,
    )

    write_source(
        tmp_path,
        """
def missing():
    raise ValueError()
    return 1
""",
    )
    write_baseline(baseline, checker.current_entries(tmp_path))

    assert checker.check_semantic_blank_lines(tmp_path, baseline) == 1
    assert "Baseline entries added relative to HEAD" in capsys.readouterr().err

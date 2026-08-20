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


def write_baseline(path: Path, counts: dict[tuple[str, str], int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Baseline counts for tests.\n"
        + "\n".join(
            f"{path_name}\t{rule}\t{count}"
            for (path_name, rule), count in sorted(counts.items())
        )
        + ("\n" if counts else ""),
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

    entries = checker.current_counts(tmp_path)

    assert any("heuristic" in entry for entry in entries)


def test_heuristic_ignores_nested_class_body(tmp_path: Path) -> None:
    assignments = "\n".join(f"        value_{index} = {index}" for index in range(16))
    write_source(
        tmp_path,
        f"def container():\n    class Nested:\n{assignments}\n    return Nested\n",
    )

    entries = checker.current_counts(tmp_path)

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

    entries = checker.current_counts(tmp_path)

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

    entries = checker.current_counts(tmp_path)

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

    entries = checker.current_counts(tmp_path)

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

    entries = checker.current_counts(tmp_path)

    assert not any("raise-block" in entry for entry in entries)


def test_block_end_rule_covers_each_compound_statement_kind(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        """
def compound_statements(items, enabled):
    if enabled:
        pass
    value = 1
    for item in items:
        value += item
    value += 1
    while enabled:
        enabled = False
    value += 1
    try:
        value += 1
    except Exception:
        value += 2
    value += 1
    with open("example.txt") as handle:
        value += len(handle.read())
    return value
""",
    )

    entries = checker.current_counts(tmp_path)

    assert entries[("models/example/src/example/module.py", "block-end")] == 5


def test_block_end_rule_flags_nested_suites_at_enclosing_indents(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        """
def nested():
    if True:
        if True:
            pass
        value = 1
    return value
""",
    )

    entries = checker.current_counts(tmp_path)

    assert entries[("models/example/src/example/module.py", "block-end")] == 2


def test_block_end_rule_ignores_suite_at_function_end(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        """
def terminal_if():
    if True:
        return 1


def terminal_with():
    with open("example.txt") as handle:
        return handle.read()
""",
    )

    entries = checker.current_counts(tmp_path)

    assert not any("block-end" in entry for entry in entries)


def test_block_end_rule_exempts_continuation_clauses(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        """
def continuations(value):
    if value == 1:
        pass
    elif value == 2:
        pass
    else:
        pass

    try:
        pass
    except ValueError:
        pass
    finally:
        pass

    return value
""",
    )

    entries = checker.current_counts(tmp_path)

    assert not any("block-end" in entry for entry in entries)


def test_block_end_rule_handles_with_statements(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        """
def missing():
    with open("example.txt") as handle:
        value = handle.read()
    return value


def present():
    with open("example.txt") as handle:
        value = handle.read()

    return value
""",
    )

    entries = checker.current_counts(tmp_path)

    assert entries[("models/example/src/example/module.py", "block-end")] == 1


def test_block_end_rule_treats_case_identifiers_as_ordinary_code(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        """
def assignment():
    if True:
        pass
    case = 1


def call():
    if True:
        pass
    case(value)


def attribute():
    if True:
        pass
    case.value = 1
""",
    )

    entries = checker.current_counts(tmp_path)

    assert entries[("models/example/src/example/module.py", "block-end")] == 3


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

    entries = checker.current_counts(tmp_path)

    assert entries[("models/example/src/example/module.py", "raise-block")] == 3


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
    assert "Baseline counts differ from current scan" in capsys.readouterr().err


def test_checker_rejects_baseline_entries_absent_from_current_scan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_source(tmp_path, "def terminal():\n    raise ValueError()\n")
    baseline = tmp_path / "baseline.txt"
    write_baseline(
        baseline,
        {("models/example/src/example/module.py", "heuristic"): 1},
    )

    assert checker.check_semantic_blank_lines(tmp_path, baseline) == 1
    assert "Baseline counts differ from current scan" in capsys.readouterr().err


def test_checker_rejects_adding_fresh_violation_to_committed_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "scripts" / "semantic_blank_lines_baseline.txt"
    write_baseline(baseline, {})
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
    write_baseline(baseline, checker.current_counts(tmp_path))

    assert checker.check_semantic_blank_lines(tmp_path, baseline) == 1
    assert (
        "Baseline counts added or increased relative to HEAD" in capsys.readouterr().err
    )


def test_checker_accepts_initial_baseline_for_new_rule(tmp_path: Path) -> None:
    baseline = tmp_path / "scripts" / "semantic_blank_lines_baseline.txt"
    write_source(
        tmp_path,
        "def missing():\n    if True:\n        pass\n    return 1\n",
    )
    write_baseline(baseline, {})
    initialize_git_repository(tmp_path)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "empty baseline"],
        cwd=tmp_path,
        check=True,
    )

    write_baseline(baseline, checker.current_counts(tmp_path))

    assert checker.check_semantic_blank_lines(tmp_path, baseline) == 0


def test_checker_accepts_legacy_location_baseline_as_initial_counts(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "scripts" / "semantic_blank_lines_baseline.txt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(
        "models/example/src/example/module.py\t1\traise-block\t"
        "ordinary code follows raise on line 2 without a blank line\n",
        encoding="utf-8",
    )
    write_source(
        tmp_path,
        "def missing():\n    raise ValueError()\n    return 1\n",
    )
    initialize_git_repository(tmp_path)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "legacy baseline"],
        cwd=tmp_path,
        check=True,
    )
    write_baseline(
        baseline,
        {("models/example/src/example/module.py", "raise-block"): 1},
    )

    assert checker.check_semantic_blank_lines(tmp_path, baseline) == 0


def test_checker_accepts_same_file_rule_swap_at_count_granularity(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "scripts" / "semantic_blank_lines_baseline.txt"
    write_source(
        tmp_path,
        "def first():\n    raise ValueError()\n    return 1\n",
    )
    write_baseline(baseline, checker.current_counts(tmp_path))
    initialize_git_repository(tmp_path)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "count baseline"],
        cwd=tmp_path,
        check=True,
    )

    write_source(
        tmp_path,
        "def second():\n    raise TypeError()\n    return 2\n",
    )

    assert checker.check_semantic_blank_lines(tmp_path, baseline) == 0

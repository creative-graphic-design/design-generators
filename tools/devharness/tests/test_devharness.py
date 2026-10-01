from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from devharness import baselines, git as git_checks, markdown


def test_read_entry_baseline_preserves_entries_and_ignores_empty_comments(
    tmp_path: Path,
) -> None:
    path = tmp_path / "baseline.txt"
    path.write_text("\n# comment\n entry \nvalue\n", encoding="utf-8")

    assert baselines.read_entry_baseline(path) == {" entry ", "value"}


def test_read_entry_baseline_requires_a_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        baselines.read_entry_baseline(tmp_path / "missing.txt")


def test_write_entry_baseline_is_sorted_and_empty_bytes_are_stable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "baseline.txt"

    baselines.write_entry_baseline(path, [])
    assert path.read_bytes() == b""

    baselines.write_entry_baseline(path, {"z", "a"})
    assert path.read_bytes() == b"a\nz\n"


def test_diff_entry_baseline_returns_sorted_unexpected_and_stale_entries() -> None:
    current = {"unexpected-two", "unexpected-one", "shared"}
    baseline = {"stale-two", "stale-one", "shared"}

    assert baselines.diff_entry_baseline(current, baseline) == (
        ["unexpected-one", "unexpected-two"],
        ["stale-one", "stale-two"],
    )


def test_print_entries_writes_exact_diagnostics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    baselines.print_entries("Header", "+", ["one", "two"])

    assert capsys.readouterr().err == "Header\n  + one\n  + two\n"


def test_git_output_returns_stdout_and_hides_failed_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        cwd: Path,
        stdout: int,
        stderr: int,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, cwd))
        assert check is False
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.DEVNULL
        assert text is True
        return subprocess.CompletedProcess(
            command, 0, stdout="exact\n", stderr="hidden"
        )

    monkeypatch.setattr(git_checks.subprocess, "run", fake_run)

    assert git_checks.git_output(tmp_path, ["git", "status"]) == "exact\n"
    assert calls == [(["git", "status"], tmp_path)]


def test_git_output_returns_none_for_nonzero_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        command: list[str],
        *,
        check: bool,
        cwd: Path,
        stdout: int,
        stderr: int,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="hidden")

    monkeypatch.setattr(git_checks.subprocess, "run", fake_run)

    assert git_checks.git_output(tmp_path, ["git", "status"]) is None


def test_markdown_primitives_preserve_tables_fences_and_heading_order() -> None:
    assert markdown.split_markdown_row(r"| a\|b | c |") == ["a|b", "c"]
    assert markdown.is_table_delimiter("| :--- | ---: |")

    text = """\
## First
one
```markdown
## Hidden
| hidden |
~~~
still hidden
```
### Second
two
~~~
# Hidden too
```
still hidden too
~~~
## Third
three
"""

    assert list(markdown.iter_unfenced_lines(text)) == [
        "## First",
        "one",
        "### Second",
        "two",
        "## Third",
        "three",
    ]
    assert [
        (level, heading)
        for level, heading, _ in markdown.iter_heading_sections_with_level(text)
    ] == [
        (2, "First"),
        (3, "Second"),
        (2, "Third"),
    ]

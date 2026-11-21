"""Risk-classifier regression tests for the report-only factory pilot."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest


def load_report_pr_risk() -> ModuleType:
    """Import the repository script without installing a package."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "report_pr_risk.py"
    spec = importlib.util.spec_from_file_location("report_pr_risk", path)
    assert spec is not None
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["report_pr_risk"] = module
    spec.loader.exec_module(module)
    return module


risk = load_report_pr_risk()


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (("docs/getting-started.md",), "auto_candidate"),
        (("docs/api/layout-dm.md",), "auto_candidate"),
        (("docs/conventions.md",), "review"),
        (("docs/training-reproduction.md",), "review"),
        (("models/layout-dm/README.md",), "review"),
        (("models/layout-dm/src/layout_dm/modeling.py",), "review"),
        ((".github/workflows/ci.yml",), "review"),
        ((".agents/skills/a/SKILL.md",), "review"),
        (("tests/test_something.py",), "review"),
        (("scripts/check_pr_issue_reference.py",), "review"),
        (("docs/api/layout-dm.md", "models/layout-dm/README.md"), "review"),
        ((), "block"),
        (("../unsafe.md",), "block"),
    ],
)
def test_candidate_is_never_merge_authorization(
    paths: tuple[str, ...], expected: str
) -> None:
    """A path-based advisory cannot grant permission to merge."""
    result = risk.classify("a" * 40, "b" * 40, paths)
    assert result.lane.value == expected
    assert result.merge_authorized is False


def test_parse_status_includes_both_rename_and_copy_paths() -> None:
    """A renamed policy file must remain visible to the classifier."""
    data = b"R100\0AGENTS.md\0docs/getting-started.md\0C075\0foo.py\0bar.py\0"
    paths = risk.read_changed_paths(data)
    assert paths == ("AGENTS.md", "bar.py", "docs/getting-started.md", "foo.py")
    assert risk.classify("a" * 40, "b" * 40, paths).lane.value == "review"


@pytest.mark.parametrize(
    "data",
    [
        b"M\0not-terminated",
        b"R100\0only-one-path\0",
        b"Q\0foo\0",
        b"M\0../foo\0",
        b"M\0a\\b\0",
        b"M\0a\n.md\0",
    ],
)
def test_unsafe_diff_fails_closed(data: bytes) -> None:
    """Malformed or ambiguous Git output must not become AUTO candidate."""
    with pytest.raises(ValueError):
        risk.read_changed_paths(data)


def test_summary_does_not_interpolate_changed_filenames() -> None:
    """Avoid injecting untrusted path text into the Actions Markdown summary."""
    report = risk.classify("a" * 40, "b" * 40, ("docs/untrusted|path.md",))
    summary = risk.render_summary(report)
    assert "untrusted|path" not in summary
    assert "merge authorized: **no**" in summary


def test_git_diff_uses_merge_base_and_preserves_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real Git protocol on a tiny local repository."""

    def git(*args: str) -> str:
        """Execute Git within the temporary fixture repository."""
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Factory Test")
    git("config", "user.email", "factory@example.org")
    (tmp_path / "AGENTS.md").write_text("policy\n", encoding="utf-8")
    git("add", "AGENTS.md")
    git("commit", "-qm", "initial")
    base = git("rev-parse", "HEAD")
    (tmp_path / "docs").mkdir()
    git("mv", "AGENTS.md", "docs/getting-started.md")
    git("commit", "-qm", "rename")
    head = git("rev-parse", "HEAD")

    monkeypatch.chdir(tmp_path)
    paths = risk.changed_paths(base, head)
    assert paths == ("AGENTS.md", "docs/getting-started.md")
    assert risk.classify(base, head, paths).lane.value == "review"


def test_requires_full_git_commit_shas() -> None:
    """Do not allow arbitrary user-controlled Git revision expressions."""
    with pytest.raises(ValueError):
        risk.changed_paths("HEAD; echo bad", "b" * 40)

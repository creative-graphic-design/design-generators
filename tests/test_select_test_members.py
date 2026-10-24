from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from importlib.machinery import SourceFileLoader

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = REPOSITORY_ROOT / "scripts" / "select_test_members.py"
LOADER = SourceFileLoader("select_test_members", str(SELECTOR_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
SELECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = SELECTOR
LOADER.exec_module(SELECTOR)


def run_selector_for_paths(
    changed_paths: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> list[dict[str, str]]:
    changed_output = b"\0".join(path.encode() for path in changed_paths) + b"\0"

    def fake_git_diff(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=changed_output)

    monkeypatch.setattr(SELECTOR.subprocess, "run", fake_git_diff)
    assert SELECTOR.main(["--base", "base-sha", "--head", "head-sha"]) == 0
    return json.loads(capsys.readouterr().out)


def expected_test_packages() -> list[str]:
    return [
        member.package
        for member in SELECTOR.discover_members(REPOSITORY_ROOT)
        if member.has_tests
    ]


def test_member_change_also_selects_devharness(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selected = run_selector_for_paths(
        ["models/layoutformerpp/src/layoutformerpp/modeling_layoutformerpp.py"],
        monkeypatch,
        capsys,
    )
    packages = [member["package"] for member in selected]
    assert "layoutformerpp" in packages
    assert "devharness" in packages


def test_laygen_change_selects_declared_transitive_dependents(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selected = run_selector_for_paths(
        ["lib/laygen/src/laygen/common.py"],
        monkeypatch,
        capsys,
    )
    packages = [member["package"] for member in selected]
    assert {"laygen", "posgen", "canvas-vae", "layout-dm", "devharness"}.issubset(
        packages
    )


def test_dependency_group_include_error_names_member_and_group() -> None:
    with pytest.raises(ValueError, match=r"posgen.*'test'.*include-group"):
        SELECTOR._requirements({}, {"test": [{"include-group": "lint"}]}, "posgen")


@pytest.mark.parametrize("changed_path", ["uv.lock", ".github/workflows/ci.yml"])
def test_non_member_change_selects_all_test_members(
    changed_path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selected = run_selector_for_paths([changed_path], monkeypatch, capsys)
    assert [member["package"] for member in selected] == expected_test_packages()


def test_markdown_only_change_selects_no_members(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        run_selector_for_paths(["models/layoutformerpp/README.md"], monkeypatch, capsys)
        == []
    )


def test_run_without_diff_base_selects_all_test_members(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert SELECTOR.main([]) == 0
    selected = json.loads(capsys.readouterr().out)
    assert [member["package"] for member in selected] == expected_test_packages()


def test_cli_requires_both_diff_commits() -> None:
    with pytest.raises(SystemExit, match="2"):
        SELECTOR.main(["--base", "base-sha"])

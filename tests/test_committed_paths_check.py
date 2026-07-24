from __future__ import annotations

import importlib.util
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest


def load_check_committed_paths() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_committed_paths.py"
    )
    spec = importlib.util.spec_from_file_location("check_committed_paths", module_path)
    assert spec is not None
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_committed_paths = load_check_committed_paths()
SLASH = "/"
BACKSLASH = "\\"


def absolute_path(*parts: str) -> str:
    return SLASH + SLASH.join(parts)


def windows_path(*parts: str) -> str:
    return "C:" + BACKSLASH + BACKSLASH.join(parts)


def init_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE)


def track(root: Path, *paths: str) -> None:
    subprocess.run(["git", "add", *paths], cwd=root, check=True, stdout=subprocess.PIPE)


def test_check_committed_paths_rejects_tracked_host_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    init_repo(tmp_path)
    bad_path = tmp_path / "docs" / "bad.md"
    bad_path.parent.mkdir()
    bad_path.write_text(
        "\n".join(
            [
                "rooted=" + absolute_path("root", "ghq", "project"),
                "linux=" + absolute_path("home", "alice", "workspace"),
                "mac=" + absolute_path("Users", "alice", "workspace"),
                "ghq " + absolute_path("tmp", "ghq", "github.com", "org", "repo"),
                "windows=" + windows_path("Users", "alice", "workspace"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    track(tmp_path, "docs/bad.md")

    assert check_committed_paths.check_committed_paths(tmp_path) == 1

    stderr = capsys.readouterr().err
    assert "Host-specific absolute paths" in stderr
    assert "docs/bad.md:1" in stderr
    assert "docs/bad.md:2" in stderr
    assert "docs/bad.md:3" in stderr
    assert "docs/bad.md:4" in stderr
    assert "docs/bad.md:5" in stderr


def test_check_committed_paths_allows_non_absolute_ghq_shapes(
    tmp_path: Path,
) -> None:
    init_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            [
                "~" + absolute_path("ghq", "github.com", "org", "repo"),
                "$HOME" + absolute_path("ghq", "github.com", "org", "repo"),
                "docs" + absolute_path("ghq", "github.com", "org", "repo"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    track(tmp_path, "README.md")

    assert check_committed_paths.check_committed_paths(tmp_path) == 0


def test_check_committed_paths_rejects_delimited_absolute_ghq_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    init_repo(tmp_path)
    path = tmp_path / "docs" / "ghq.md"
    path.parent.mkdir()
    path.write_text(
        "\n".join(
            [
                "x=" + absolute_path("tmp", "ghq", "github.com", "org", "repo"),
                "path:" + absolute_path("tmp", "ghq", "github.com", "org", "repo"),
                "x=(" + absolute_path("tmp", "ghq", "github.com", "org", "repo") + ")",
                "x=[" + absolute_path("tmp", "ghq", "github.com", "org", "repo") + "]",
                "x={" + absolute_path("tmp", "ghq", "github.com", "org", "repo") + "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    track(tmp_path, "docs/ghq.md")

    assert check_committed_paths.check_committed_paths(tmp_path) == 1

    stderr = capsys.readouterr().err
    assert "docs/ghq.md:1" in stderr
    assert "docs/ghq.md:2" in stderr
    assert "docs/ghq.md:3" in stderr
    assert "docs/ghq.md:4" in stderr
    assert "docs/ghq.md:5" in stderr


def test_check_committed_paths_passes_for_clean_tracked_text(tmp_path: Path) -> None:
    init_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        "Use a repo-relative path or a tilde-prefixed ghq checkout hint.\n",
        encoding="utf-8",
    )
    track(tmp_path, "README.md")

    assert check_committed_paths.check_committed_paths(tmp_path) == 0


def test_check_committed_paths_ignores_excluded_and_untracked_files(
    tmp_path: Path,
) -> None:
    init_repo(tmp_path)
    vendor_file = tmp_path / "vendor" / "fixture.txt"
    vendor_file.parent.mkdir()
    vendor_file.write_text(absolute_path("root", "vendor") + "\n", encoding="utf-8")
    lock_file = tmp_path / "uv.lock"
    lock_file.write_text(
        absolute_path("home", "alice", "cache") + "\n", encoding="utf-8"
    )
    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(
        b"\0" + absolute_path("Users", "alice", "binary").encode("utf-8")
    )
    untracked = tmp_path / "untracked.txt"
    untracked.write_text(absolute_path("root", "untracked") + "\n", encoding="utf-8")
    track(tmp_path, "vendor/fixture.txt", "uv.lock", "binary.bin")

    assert check_committed_paths.check_committed_paths(tmp_path) == 0

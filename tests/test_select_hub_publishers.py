from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = REPOSITORY_ROOT / "scripts" / "select_hub_publishers.py"
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "hub-publish.yml"
LOADER = SourceFileLoader("select_hub_publishers", str(SELECTOR_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
SELECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = SELECTOR
LOADER.exec_module(SELECTOR)


def make_model_package(root: Path, name: str, *, publishes: bool) -> None:
    package_directory = root / "models" / name
    package_directory.mkdir(parents=True)
    (package_directory / "pyproject.toml").write_text(f"[project]\nname = '{name}'\n")
    if publishes:
        publisher_path = package_directory / "scripts" / "publish_to_hub.py"
        publisher_path.parent.mkdir()
        publisher_path.write_text("raise SystemExit(0)\n")


def test_selects_only_changed_packages_with_a_publisher(tmp_path: Path) -> None:
    make_model_package(tmp_path, "collage-diffusion", publishes=True)
    make_model_package(tmp_path, "other-model", publishes=True)
    make_model_package(tmp_path, "no-publisher", publishes=False)

    selected = SELECTOR.select_publishers(
        tmp_path,
        [
            "models/collage-diffusion/README.md",
            "models/no-publisher/src/no_publisher.py",
            "docs/hub-publishing.md",
        ],
    )

    assert selected == [
        SELECTOR.HubPublisher(
            package="collage-diffusion",
            directory=Path("models") / "collage-diffusion",
        )
    ]


def test_returns_no_publishers_for_changes_outside_model_packages(
    tmp_path: Path,
) -> None:
    make_model_package(tmp_path, "collage-diffusion", publishes=True)

    assert SELECTOR.select_publishers(tmp_path, ["docs/architecture.md"]) == []


def test_no_diff_selects_all_configured_publishers(tmp_path: Path) -> None:
    make_model_package(tmp_path, "z-model", publishes=True)
    make_model_package(tmp_path, "a-model", publishes=True)

    assert SELECTOR.select_publishers(tmp_path, None) == [
        SELECTOR.HubPublisher(package="a-model", directory=Path("models") / "a-model"),
        SELECTOR.HubPublisher(package="z-model", directory=Path("models") / "z-model"),
    ]


def test_discover_skips_non_packages(tmp_path: Path) -> None:
    models_directory = tmp_path / "models"
    models_directory.mkdir()
    (models_directory / "not-a-directory").write_text("file")
    (models_directory / "no-pyproject").mkdir()

    assert SELECTOR.discover_publishers(tmp_path) == []


def test_changed_paths_reads_null_delimited_git_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git_diff(
        command: list[str], *, check: bool, stdout: int
    ) -> subprocess.CompletedProcess[bytes]:
        assert command == [
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            "base",
            "head",
        ]
        assert check is True
        assert stdout == subprocess.PIPE
        return subprocess.CompletedProcess(command, 0, b"models/example/README.md\0")

    monkeypatch.setattr(SELECTOR.subprocess, "run", fake_git_diff)

    assert SELECTOR._changed_paths("base", "head") == ["models/example/README.md"]


@pytest.mark.parametrize(
    ("newer_commits", "expected"),
    [(b"", True), (b"newer-package-commit\n", False)],
)
def test_freshness_check_rejects_newer_main_commits_for_the_package(
    monkeypatch: pytest.MonkeyPatch,
    newer_commits: bytes,
    expected: bool,
) -> None:
    def fake_git_log(
        command: list[str], *, check: bool, stdout: int, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert command == [
            "git",
            "log",
            "--first-parent",
            "--format=%H",
            "event-revision..origin/main",
            "--",
            "models/collage-diffusion",
        ]
        assert check is True
        assert stdout == subprocess.PIPE
        assert text is True
        return subprocess.CompletedProcess(
            command,
            0,
            newer_commits.decode("ascii"),
        )

    monkeypatch.setattr(SELECTOR.subprocess, "run", fake_git_log)

    assert (
        SELECTOR.is_package_change_current_on_main(
            Path("models") / "collage-diffusion", "event-revision", "origin/main"
        )
        is expected
    )


def test_main_serializes_package_freshness(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        SELECTOR,
        "is_package_change_current_on_main",
        lambda package_directory, event_revision, main_ref: False,
    )

    assert (
        SELECTOR.main(
            [
                "--check-freshness",
                "--package-directory",
                "models/collage-diffusion",
                "--event-revision",
                "event-revision",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out) == {"is_current": False}


@pytest.mark.parametrize(
    "arguments",
    [
        ["--check-freshness"],
        [
            "--check-freshness",
            "--package-directory",
            "models/collage-diffusion",
            "--event-revision",
            "event-revision",
            "--base",
            "base",
            "--head",
            "head",
        ],
        ["--package-directory", "models/collage-diffusion"],
    ],
)
def test_main_rejects_invalid_freshness_arguments(arguments: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        SELECTOR.main(arguments)


def test_main_serializes_selected_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_model_package(tmp_path, "collage-diffusion", publishes=True)
    monkeypatch.setattr(SELECTOR, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        SELECTOR,
        "_changed_paths",
        lambda base, head: ["models/collage-diffusion/README.md"],
    )

    assert SELECTOR.main(["--base", "base", "--head", "head"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "include": [
            {"package": "collage-diffusion", "directory": "models/collage-diffusion"}
        ]
    }


def test_main_requires_base_and_head_together() -> None:
    with pytest.raises(SystemExit, match="2"):
        SELECTOR.main(["--base", "base"])


def test_workflow_serializes_publication_and_gates_stale_jobs() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "group: hub-publish-" + "$" + "{{ matrix.package }}" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "fetch-depth: 0" in workflow
    assert "git fetch --no-tags origin main:refs/remotes/origin/main" in workflow
    assert "--check-freshness" in workflow
    assert workflow.count("if: steps.freshness.outputs.publish == 'true'") == 3
    assert workflow.count("PYTHONPATH: tools/devharness/src") == 2

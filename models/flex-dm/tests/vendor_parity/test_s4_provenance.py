"""Tests for capture worktree provenance."""

import subprocess
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest


def test_worktree_provenance_records_head_and_rejects_dirty(
    monkeypatch, tmp_path
) -> None:
    worktree_provenance = run_path(
        str(Path(__file__).with_name("capture_flex_dm_training_s4.py"))
    )["_worktree_provenance"]

    commands = iter(
        [
            ["git", "rev-parse", "--show-toplevel"],
            ["git", "rev-parse", "HEAD"],
            ["git", "status", "--porcelain", "--untracked-files=all"],
        ]
        * 3
    )
    calls = iter(
        (
            str(tmp_path) + "\n",
            "a" * 40 + "\n",
            "",
            str(tmp_path) + "\n",
            "a" * 40 + "\n",
            "?? dirty.txt\n",
            str(tmp_path) + "\n",
            "a" * 40 + "\n",
            "?? dirty.txt\n",
        )
    )

    def git_run(command, **kwargs):
        assert command == next(commands)
        assert kwargs["cwd"] == tmp_path
        return SimpleNamespace(stdout=next(calls))

    monkeypatch.setattr(subprocess, "run", git_run)

    clean = worktree_provenance(tmp_path)
    assert clean == {
        "git_head_sha": "a" * 40,
        "git_top_level": str(tmp_path),
        "git_worktree_clean": True,
    }

    with pytest.raises(RuntimeError, match="requires a clean git worktree"):
        worktree_provenance(tmp_path)

    dirty_dry_run = worktree_provenance(tmp_path, dry_run=True)
    assert dirty_dry_run == {
        "git_head_sha": "a" * 40,
        "git_top_level": str(tmp_path),
        "git_worktree_clean": False,
    }


def test_package_module_paths_record_and_reject_outside_modules(tmp_path) -> None:
    from types import ModuleType

    package_module_paths = run_path(
        str(Path(__file__).with_name("capture_flex_dm_training_s4.py"))
    )["_package_module_paths"]
    root = tmp_path / "repo"
    modules = {
        name: ModuleType(name)
        for name in (
            "flex_dm",
            "flex_dm.training.dataset",
            "traingen",
            "traingen_parity.tensorflow_compat",
        )
    }
    for name, module in modules.items():
        module.__file__ = str(root / f"{name.replace('.', '/')}.py")

    paths = package_module_paths(root, modules)

    assert paths == {
        name: str(Path(module.__file__).resolve()) for name, module in modules.items()
    }
    modules["flex_dm.training.dataset"].__file__ = str(tmp_path / "s3" / "dataset.py")
    with pytest.raises(RuntimeError, match="resolves outside git worktree"):
        package_module_paths(root, modules)

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
            ["git", "rev-parse", "HEAD"],
            ["git", "status", "--porcelain", "--untracked-files=all"],
        ]
        * 3
    )
    calls = iter(
        (
            "a" * 40 + "\n",
            "",
            "a" * 40 + "\n",
            "?? dirty.txt\n",
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
    assert clean == {"git_head_sha": "a" * 40, "git_worktree_clean": True}

    with pytest.raises(RuntimeError, match="requires a clean git worktree"):
        worktree_provenance(tmp_path)

    dirty_dry_run = worktree_provenance(tmp_path, dry_run=True)
    assert dirty_dry_run == {"git_head_sha": "a" * 40, "git_worktree_clean": False}

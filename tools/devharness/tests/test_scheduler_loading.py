"""Tests for the Diffusers scheduler loading policy check."""

from __future__ import annotations

from pathlib import Path

import pytest

from devharness.checks.scheduler_loading import check as scheduler_loading
from devharness.checks.scheduler_loading.check import (
    check,
    find_scheduler_assignment_violations,
)


def _write(root: Path, relative_path: str, text: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_finds_scheduler_assignments_in_documents_and_runtime_sources(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "models/example/README.md",
        "pipe.scheduler = Scheduler.from_config(\n    pipe.scheduler.config\n)\n",
    )
    _write(
        tmp_path,
        "models/example/REPRODUCING.md",
        'pipe.scheduler = Scheduler.from_pretrained("repo", subfolder="scheduler")\n',
    )
    _write(
        tmp_path,
        "docs/conventions.md",
        "pipe.scheduler = package.Scheduler.from_config(config)\n",
    )
    _write(
        tmp_path,
        "models/example/src/example/pipeline.py",
        "self.pipe.scheduler = schedulers.EulerScheduler.from_config(config)\n",
    )
    _write(
        tmp_path,
        "lib/laygen/src/laygen/pipeline.py",
        'pipe.scheduler = package.Scheduler.from_pretrained("repo")\n',
    )

    assert find_scheduler_assignment_violations(tmp_path) == [
        "docs/conventions.md:1",
        "lib/laygen/src/laygen/pipeline.py:1",
        "models/example/README.md:1",
        "models/example/REPRODUCING.md:1",
        "models/example/src/example/pipeline.py:1",
    ]


def test_rejects_readme_replacement_example_and_allows_scheduler_injection(
    tmp_path: Path,
) -> None:
    readme = """```python
pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
```"""
    _write(tmp_path, "models/example/README.md", readme)

    assert find_scheduler_assignment_violations(tmp_path) == [
        "models/example/README.md:2"
    ]

    readme = """```python
scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
```"""
    _write(tmp_path, "models/example/README.md", readme)

    assert find_scheduler_assignment_violations(tmp_path) == []


def test_horizontal_whitespace_does_not_match_across_lines(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "models/example/src/example/pipeline.py",
        "pipe\n.scheduler = Scheduler\n.from_config(config)\n",
    )

    assert find_scheduler_assignment_violations(tmp_path) == []


def test_skips_test_and_vendor_source_directories(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "models/example/src/tests/test_pipeline.py",
        "pipe.scheduler = Scheduler.from_config(config)\n",
    )
    _write(
        tmp_path,
        "lib/laygen/src/vendor/pipeline.py",
        "pipe.scheduler = Scheduler.from_config(config)\n",
    )

    assert find_scheduler_assignment_violations(tmp_path) == []


def test_allows_existing_scheduler_references_and_other_factories(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "models/example/src/example/pipeline.py",
        "pipe.scheduler = scheduler\npipe.scheduler = Scheduler.from_other(config)\n",
    )

    assert find_scheduler_assignment_violations(tmp_path) == []


def test_check_reports_forbidden_assignments(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(
        tmp_path,
        "models/example/README.md",
        "pipe.scheduler = Scheduler.from_config(config)\n",
    )

    assert check(tmp_path) == 1
    assert capsys.readouterr().err == (
        "Pass Diffusers pipeline schedulers to `from_pretrained(..., scheduler=...)`; "
        "do not assign them after loading:\n"
        "  models/example/README.md:1\n"
    )


def test_check_passes_when_no_assignments_are_present(tmp_path: Path) -> None:
    assert check(tmp_path) == 0


def test_main_runs_check_at_discovered_repository_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduler_loading, "find_repo_root", lambda _: tmp_path)
    monkeypatch.setattr(scheduler_loading, "check", lambda root: 17)

    assert scheduler_loading.main([]) == 17


def test_main_reports_missing_repository_root(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raise_missing_root(_: Path) -> Path:
        raise ValueError("missing workspace root")

    monkeypatch.setattr(scheduler_loading, "find_repo_root", raise_missing_root)

    assert scheduler_loading.main([]) == 1
    assert capsys.readouterr().err == "missing workspace root\n"

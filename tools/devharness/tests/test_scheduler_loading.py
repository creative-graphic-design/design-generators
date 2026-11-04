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


def test_finds_assignments_in_python_fences_and_runtime_sources(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "models/example/README.md",
        "```python\npipe.scheduler = (\n"
        "    Scheduler.from_config(pipe.scheduler.config)\n)\n```\n",
    )
    _write(
        tmp_path,
        "models/example/REPRODUCING.md",
        "```py title=replace scheduler\n"
        'pipe.scheduler = package.Scheduler.from_pretrained("repo")\n````\n',
    )
    _write(
        tmp_path,
        "docs/conventions.md",
        "  ``` python\npipe.scheduler = package.Scheduler.from_config(config)\n  ```\n",
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
        "docs/conventions.md:2",
        "lib/laygen/src/laygen/pipeline.py:1",
        "models/example/README.md:2",
        "models/example/REPRODUCING.md:2",
        "models/example/src/example/pipeline.py:1",
    ]


def test_allows_scheduler_injection_and_ignores_prose_and_other_fences(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "models/example/README.md",
        "The old form was `pipe.scheduler = Scheduler.from_config(config)`.\n"
        "```text\npipe.scheduler = Scheduler.from_config(config)\n```\n"
        "```python\n"
        'scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")\n'
        "pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)\n"
        "```\n",
    )

    assert find_scheduler_assignment_violations(tmp_path) == []


def test_reports_python_code_fence_parse_errors_instead_of_skipping(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "docs/conventions.md",
        "```python\nif :\n```\n",
    )

    assert find_scheduler_assignment_violations(tmp_path) == [
        "docs/conventions.md:2: cannot parse Python code fence: invalid syntax"
    ]


def test_reports_unterminated_python_code_fence(tmp_path: Path) -> None:
    _write(tmp_path, "models/example/README.md", "```python\npass\n")

    assert find_scheduler_assignment_violations(tmp_path) == [
        "models/example/README.md:1: cannot parse Python code fence: "
        "unterminated Python code fence"
    ]


def test_reports_python_source_parse_errors(tmp_path: Path) -> None:
    _write(tmp_path, "lib/laygen/src/laygen/pipeline.py", "if :\n")

    assert find_scheduler_assignment_violations(tmp_path) == [
        "lib/laygen/src/laygen/pipeline.py:1: cannot parse Python source: invalid syntax"
    ]


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


def test_rejects_only_supported_scheduler_factory_assignments(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "models/example/src/example/pipeline.py",
        "pipe.scheduler = scheduler\n"
        "pipe.scheduler = Scheduler.from_other(config)\n"
        "pipe.scheduler = (\n    Scheduler.from_config(config)\n)\n",
    )

    assert find_scheduler_assignment_violations(tmp_path) == [
        "models/example/src/example/pipeline.py:3"
    ]


def test_check_reports_forbidden_assignments(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(
        tmp_path,
        "models/example/README.md",
        "```python\npipe.scheduler = Scheduler.from_config(config)\n```\n",
    )

    assert check(tmp_path) == 1
    assert capsys.readouterr().err == (
        "Pass Diffusers pipeline schedulers to `from_pretrained(..., scheduler=...)`; "
        "do not assign them after loading, and keep scanned Python syntax valid:\n"
        "  models/example/README.md:2\n"
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

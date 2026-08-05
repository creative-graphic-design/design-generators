from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest


def load_check_training_doc_template() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check_training_doc_template.py"
    )
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location(
        "check_training_doc_template", module_path
    )
    assert spec is not None
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_training_doc_template = load_check_training_doc_template()


def write_package_docs(
    root: Path,
    package: str,
    readme_text: str,
    training_text: str,
) -> Path:
    package_dir = root / "models" / package
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "README.md").write_text(readme_text, encoding="utf-8")
    training_path = package_dir / "TRAINING.md"
    training_path.write_text(training_text, encoding="utf-8")
    return training_path


def readme_with_supported_checkpoints() -> str:
    return """
# Model Card

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 | `creative-graphic-design/example-rico25` | not-published |
| PubLayNet | `creative-graphic-design/example-publaynet` | not-published |
"""


def valid_training_doc() -> str:
    return """
# Example Training

## Scheduler and Recipe Notes

The scheduler steps once per optimizer update.

## Seed Policy

RICO25 and PubLayNet use training-seed n=3.

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `uv run pytest s0` | `.cache/pkg/s0.json` | PASS |

## Reproduction Results

| Dataset | System | Status | Seed scope | Primary metrics | Loss evidence | Artifact summary |
| --- | --- | --- | --- | --- | --- | --- |
| RICO25 | package | `s5-practical-reproduction` | training-seed n=3 | FID 1.0 | loss matched | `.cache/pkg/rico25` |
| PubLayNet | package | `not-yet-run (#253)` | training-seed n=3 | pending | pending | `.cache/pkg/publaynet` |

## Regeneration Metadata

```text
.cache/pkg/
```

## Training Commands

```bash
uv run --package example pytest
```
"""


def test_valid_training_doc_passes(tmp_path: Path) -> None:
    write_package_docs(
        tmp_path,
        "example",
        readme_with_supported_checkpoints(),
        valid_training_doc(),
    )

    assert check_training_doc_template.current_entries(tmp_path) == set()


def test_missing_required_section_fails(tmp_path: Path) -> None:
    write_package_docs(
        tmp_path,
        "example",
        readme_with_supported_checkpoints(),
        valid_training_doc().replace("## Seed Policy\n", "## Seeds\n"),
    )

    assert check_training_doc_template.current_entries(tmp_path) == {
        "models/example/TRAINING.md\tSeed Policy\tmissing required TRAINING.md section"
    }


def test_readme_supported_checkpoint_dataset_requires_result_row(
    tmp_path: Path,
) -> None:
    training_text = valid_training_doc().replace(
        "| PubLayNet | package | `not-yet-run (#253)` | training-seed n=3 | pending | pending | `.cache/pkg/publaynet` |\n",
        "",
    )
    write_package_docs(
        tmp_path,
        "example",
        readme_with_supported_checkpoints(),
        training_text,
    )

    assert check_training_doc_template.current_entries(tmp_path) == {
        "models/example/TRAINING.md\tpublaynet\tREADME Supported Checkpoints dataset missing from Reproduction Results"
    }


@pytest.mark.parametrize(
    "status",
    [
        "PASS",
        "statistically equivalent",
        "not-yet-run",
        "not-yet-run (<tracking ref>)",
        "blocked (-)",
    ],
)
def test_status_free_text_and_placeholder_reasons_fail(
    tmp_path: Path, status: str
) -> None:
    training_text = valid_training_doc().replace(
        "`s5-practical-reproduction`", status, 1
    )
    write_package_docs(
        tmp_path,
        "example",
        readme_with_supported_checkpoints(),
        training_text,
    )

    assert check_training_doc_template.current_entries(tmp_path) == {
        "models/example/TRAINING.md\tRICO25\tReproduction Results status is not an allowed enum value"
    }


@pytest.mark.parametrize(
    "status",
    [
        "s5-bit-parity",
        "s5-practical-reproduction",
        "recipe-unstable (documented)",
        "not-yet-run (#253)",
        "blocked (dataset license is unresolved)",
    ],
)
def test_allowed_status_enum_values_pass(tmp_path: Path, status: str) -> None:
    training_text = valid_training_doc().replace(
        "`s5-practical-reproduction`", status, 1
    )
    write_package_docs(
        tmp_path,
        "example",
        readme_with_supported_checkpoints(),
        training_text,
    )

    assert check_training_doc_template.current_entries(tmp_path) == set()


def test_check_reports_unexpected_and_stale_baseline_entries(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_package_docs(
        tmp_path,
        "example",
        readme_with_supported_checkpoints(),
        valid_training_doc().replace("## Training Commands\n", "## Commands\n"),
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("models/old/TRAINING.md\tSeed Policy\told\n", encoding="utf-8")

    assert (
        check_training_doc_template.check_training_doc_template(tmp_path, baseline) == 1
    )

    stderr = capsys.readouterr().err
    assert "New TRAINING.md template violations" in stderr
    assert "Stale TRAINING.md template baseline entries" in stderr


def test_check_passes_when_baseline_matches(tmp_path: Path) -> None:
    write_package_docs(
        tmp_path,
        "example",
        readme_with_supported_checkpoints(),
        valid_training_doc().replace("## Training Commands\n", "## Commands\n"),
    )
    baseline = tmp_path / "baseline.txt"
    check_training_doc_template.write_baseline(
        baseline, check_training_doc_template.current_entries(tmp_path)
    )

    assert (
        check_training_doc_template.check_training_doc_template(tmp_path, baseline) == 0
    )

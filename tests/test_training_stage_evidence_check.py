from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest


def load_check_training_stage_evidence() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check_training_stage_evidence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_training_stage_evidence", module_path
    )
    assert spec is not None
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_training_stage_evidence = load_check_training_stage_evidence()


def write_training_md(root: Path, package: str, text: str) -> Path:
    path = root / "models" / package / "TRAINING.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def complete_stage_evidence_table() -> str:
    return """
## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `uv run pytest s0` | `.cache/pkg/s0.json` | Static config parity passed. |
| S1 | `uv run pytest s1` | `.cache/pkg/s1.json` | Fixed-batch trace parity passed. |
| S2 | `uv run pytest s2` | `.cache/pkg/s2.json` | One optimizer step parity passed. |
| S3 | `uv run pytest s3` | `.cache/pkg/s3/metrics.csv` | Multi-batch deterministic run passed. |
| S4 | `uv run pytest s4` | `.cache/pkg/s4/stream.jsonl` | Loader stream parity passed. |
| S5 | `uv run train full` | `.cache/pkg/full-run/summary.csv` | training-seed n=3 accepted. |
"""


def test_parse_stage_evidence_accepts_complete_rows() -> None:
    evidence = check_training_stage_evidence.parse_stage_evidence(
        complete_stage_evidence_table()
    )

    rows, duplicates = evidence
    assert sorted(rows) == ["S0", "S1", "S2", "S3", "S4", "S5"]
    assert rows["S0"].is_complete
    assert duplicates == set()


def test_parse_stage_evidence_ignores_fenced_heading_and_table() -> None:
    evidence, duplicates = check_training_stage_evidence.parse_stage_evidence(
        f"""
```markdown
## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | placeholder | placeholder | placeholder |
```

{complete_stage_evidence_table()}
"""
    )

    assert sorted(evidence) == ["S0", "S1", "S2", "S3", "S4", "S5"]
    assert evidence["S0"].command == "`uv run pytest s0`"
    assert duplicates == set()


def test_fenced_reproduction_results_heading_does_not_satisfy_gate(
    tmp_path: Path,
) -> None:
    write_training_md(
        tmp_path,
        "layout-dm",
        f"""
# Training

```markdown
## Reproduction Results
```

S5 verdict is accepted.

{complete_stage_evidence_table()}
""",
    )

    assert check_training_stage_evidence.current_entries(tmp_path) == {
        "models/layout-dm/TRAINING.md\t*\tS5 result claim requires a Reproduction Results heading"
    }


def test_s5_claim_with_complete_stage_evidence_passes(tmp_path: Path) -> None:
    write_training_md(
        tmp_path,
        "layout-dm",
        f"""
# Training

## Reproduction Results

Package training reproduction is achieved with training-seed n=3.

{complete_stage_evidence_table()}
""",
    )

    assert check_training_stage_evidence.current_entries(tmp_path) == set()


def test_s5_claim_missing_prior_stage_rows_fails(tmp_path: Path) -> None:
    write_training_md(
        tmp_path,
        "layout-dm",
        """
# Training

## Reproduction Results

S5 verdict: full-run statistical comparison is accepted at training-seed n=3.

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S5 | `uv run train full` | `.cache/pkg/full-run/summary.csv` | PASS |
""",
    )

    entries = check_training_stage_evidence.current_entries(tmp_path)
    reason = "S5 result claim requires a complete evidence row for this stage"

    assert entries == {
        f"models/layout-dm/TRAINING.md\tS0\t{reason}",
        f"models/layout-dm/TRAINING.md\tS1\t{reason}",
        f"models/layout-dm/TRAINING.md\tS2\t{reason}",
        f"models/layout-dm/TRAINING.md\tS3\t{reason}",
        f"models/layout-dm/TRAINING.md\tS4\t{reason}",
    }


def test_s5_claim_with_prose_only_evidence_cells_fails(tmp_path: Path) -> None:
    write_training_md(
        tmp_path,
        "layout-dm",
        """
# Training

## Reproduction Results

training-seed n=3 is accepted.

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | passed manually | passed manually | PASS |
| S1 | `uv run pytest s1` | `.cache/pkg/s1.json` | PASS |
| S2 | `uv run pytest s2` | `.cache/pkg/s2.json` | PASS |
| S3 | `uv run pytest s3` | `.cache/pkg/s3.json` | PASS |
| S4 | `uv run pytest s4` | `.cache/pkg/s4.json` | PASS |
| S5 | `uv run train full` | `.cache/pkg/full-run/summary.csv` | PASS |
""",
    )

    assert check_training_stage_evidence.current_entries(tmp_path) == {
        "models/layout-dm/TRAINING.md\tS0\tstage evidence row has a placeholder command, artifact, or result"
    }


@pytest.mark.parametrize(
    "claim",
    [
        "S5 verdict is accepted.",
        "s-5 verdict is accepted.",
        "Stage 5 verdict is accepted.",
        "The RICO25 result uses training-seed n=3.",
        "Training reproduction is achieved.",
        "The run is statistically equivalent.",
        "The full run statistical comparison is complete.",
        "The full-run statistical comparison is complete.",
    ],
)
def test_has_s5_claim_detects_each_claim_pattern(claim: str) -> None:
    assert check_training_stage_evidence.has_s5_claim(claim)


def test_has_s5_claim_allows_pending_clause_without_hiding_claim() -> None:
    assert check_training_stage_evidence.has_s5_claim(
        "Training reproduction is achieved for RICO25; model card update pending."
    )
    assert check_training_stage_evidence.has_s5_claim(
        "Training reproduction is achieved for RICO25, model card update pending."
    )
    assert not check_training_stage_evidence.has_s5_claim(
        "S5 full-run comparison is pending and is not claimed in this PR."
    )
    assert not check_training_stage_evidence.has_s5_claim(
        "training-seed n=3 is not claimed in this PR."
    )


def test_has_s5_claim_detects_wrapped_training_seed() -> None:
    assert check_training_stage_evidence.has_s5_claim(
        "The result is reported with training-seed\nn=3."
    )


def test_s5_claim_under_alternate_heading_requires_results_heading(
    tmp_path: Path,
) -> None:
    write_training_md(
        tmp_path,
        "layout-dm",
        f"""
# Training

## Results

S5 verdict is accepted.

{complete_stage_evidence_table()}
""",
    )

    assert check_training_stage_evidence.current_entries(tmp_path) == {
        "models/layout-dm/TRAINING.md\t*\tS5 result claim requires a Reproduction Results heading"
    }


def test_artifact_path_allows_project_github_urls() -> None:
    assert check_training_stage_evidence.is_artifact_path(
        "https://github.com/creative-graphic-design/design-generators/issues/149#issuecomment-5060415006"
    )


@pytest.mark.parametrize(
    "artifact",
    [
        "../outside/trace.json",
        "models/layout-dm/TRAINING.md",
        ".cache/pkg/../trace.json",
    ],
)
def test_artifact_path_rejects_traversal_and_self_reference(artifact: str) -> None:
    assert not check_training_stage_evidence.is_artifact_path(artifact)


def test_duplicate_stage_rows_are_violations(tmp_path: Path) -> None:
    write_training_md(
        tmp_path,
        "layout-dm",
        """
# Training

## Reproduction Results

training-seed n=3 is accepted.

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `uv run pytest s0` | `.cache/pkg/s0-a.json` | PASS |
| S0 | `uv run pytest s0` | `.cache/pkg/s0-b.json` | PASS |
| S1 | `uv run pytest s1` | `.cache/pkg/s1.json` | PASS |
| S2 | `uv run pytest s2` | `.cache/pkg/s2.json` | PASS |
| S3 | `uv run pytest s3` | `.cache/pkg/s3.json` | PASS |
| S4 | `uv run pytest s4` | `.cache/pkg/s4.json` | PASS |
| S5 | `uv run train full` | `.cache/pkg/full-run/summary.csv` | PASS |
""",
    )

    assert check_training_stage_evidence.current_entries(tmp_path) == {
        "models/layout-dm/TRAINING.md\tS0\tstage evidence table contains duplicate rows for this stage"
    }


def test_pending_s5_text_without_result_claim_does_not_require_evidence(
    tmp_path: Path,
) -> None:
    write_training_md(
        tmp_path,
        "layout-dm",
        """
# Training

## Reproduction Results

S5 full-run comparison is pending and is not claimed in this PR.
""",
    )

    assert check_training_stage_evidence.current_entries(tmp_path) == set()


def test_check_fails_on_unexpected_and_stale_baseline_entries(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_training_md(
        tmp_path,
        "layout-dm",
        """
# Training

## Reproduction Results

training-seed n=3 is accepted.
""",
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("models/old/TRAINING.md\tS0\told\n", encoding="utf-8")

    assert (
        check_training_stage_evidence.check_training_stage_evidence(tmp_path, baseline)
        == 1
    )

    stderr = capsys.readouterr().err
    assert "New training stage evidence violations" in stderr
    assert "Stale training stage evidence baseline entries" in stderr


def test_check_passes_when_baseline_matches(tmp_path: Path) -> None:
    write_training_md(
        tmp_path,
        "layout-flow",
        """
# Training

## Reproduction Results

training-seed n=3 is accepted.

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `uv run pytest s0` | `.cache/pkg/trace.json` | PASS |
| S1 | `uv run pytest s1` | `.cache/pkg/trace.json` | PASS |
| S2 | `uv run pytest s2` | `.cache/pkg/trace.json` | PASS |
| S5 | `uv run train full` | `.cache/pkg/summary.csv` | PASS |
""",
    )
    baseline = tmp_path / "baseline.txt"
    check_training_stage_evidence.write_baseline(
        baseline, check_training_stage_evidence.current_entries(tmp_path)
    )

    assert (
        check_training_stage_evidence.check_training_stage_evidence(tmp_path, baseline)
        == 0
    )

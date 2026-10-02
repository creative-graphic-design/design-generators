from __future__ import annotations

from pathlib import Path

import pytest

from devharness.baselines import read_entry_baseline, write_entry_baseline


SHARED_WRITER_BASELINES = (
    Path("scripts/jaxtyping_baseline.txt"),
    Path("scripts/jaxtyping_alias_baseline.txt"),
    Path("scripts/object_annotation_baseline.txt"),
    Path("scripts/weak_cast_type_baseline.txt"),
    Path("scripts/module_naming_baseline.txt"),
    Path("scripts/training_doc_template_baseline.txt"),
    Path("scripts/training_stage_evidence_baseline.txt"),
)


@pytest.mark.parametrize("baseline_path", SHARED_WRITER_BASELINES)
def test_shared_writer_baseline_round_trip(baseline_path: Path, tmp_path: Path) -> None:
    original = baseline_path.read_bytes()
    rewritten = tmp_path / baseline_path.name
    rewritten.write_bytes(original)

    write_entry_baseline(rewritten, read_entry_baseline(baseline_path))

    assert rewritten.read_bytes() == original

"""Tests for coverage badge endpoint aggregation."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aggregate_member_coverage",
    REPO_ROOT / "scripts" / "aggregate_member_coverage.py",
)
assert SPEC is not None
aggregate_member_coverage = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = aggregate_member_coverage
SPEC.loader.exec_module(aggregate_member_coverage)


def test_aggregate_member_coverage_writes_minimum_endpoint(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "laygen.json").write_text(
        json.dumps({"totals": {"percent_covered": 92.3456}}),
        encoding="utf-8",
    )
    (input_dir / "layout-dm.json").write_text(
        json.dumps({"totals": {"percent_covered": 94.2}}),
        encoding="utf-8",
    )

    rows = aggregate_member_coverage.aggregate(input_dir, output_dir)

    assert [row.package for row in rows] == ["laygen", "layout-dm"]
    endpoint = json.loads((output_dir / "coverage-badge.json").read_text())
    assert endpoint == {
        "schemaVersion": 1,
        "label": "coverage",
        "message": "min 92.3% per member",
        "color": "brightgreen",
    }
    summary = json.loads((output_dir / "coverage-members.json").read_text())
    assert summary["minimum"] == {"package": "laygen", "percent": 92.3456}

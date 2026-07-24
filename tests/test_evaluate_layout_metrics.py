from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest


pytestmark = pytest.mark.integration


def test_evaluate_layout_metrics_match_layout_dm_vendor() -> None:
    if os.environ.get("LAYOUT_METRICS_PARITY_REQUIRE") != "1":
        pytest.skip(
            "Set LAYOUT_METRICS_PARITY_REQUIRE=1 to run network-backed "
            "evaluate.load layout metric parity."
        )

    script_path = (
        Path(__file__).parents[1] / "scripts/verify_evaluate_layout_metrics.py"
    )
    spec = importlib.util.spec_from_file_location(
        "verify_evaluate_layout_metrics", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rows = module.run_verification()
    mismatches = [row for row in rows if row.verdict == "mismatch"]
    assert mismatches == []

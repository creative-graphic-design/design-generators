from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("h5pickle")

from laygen.common.testing import skip_or_fail_vendor_parity

pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "models" / "layout-flow" / "scripts" / "training_stage_evidence.py"
OUTPUT_ROOT = ROOT / ".cache" / "layout-flow" / "stage-evidence"
VENDOR_ROOT = ROOT / "vendor" / "layout-flow"


def test_s4_loader_stream_matches_vendor() -> None:
    if not (VENDOR_ROOT / "src" / "datamodule" / "PubLayNet.py").exists():
        skip_or_fail_vendor_parity(
            "LayoutFlow vendor checkout is missing",
            missing_paths=[VENDOR_ROOT],
            regeneration_hint="run `git submodule update --init vendor/layout-flow`",
        )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "s4-loader-stream",
            "--output-root",
            str(OUTPUT_ROOT),
            "--vendor-root",
            str(VENDOR_ROOT),
        ],
        check=True,
        cwd=ROOT,
    )
    artifact = OUTPUT_ROOT / "s4-loader-stream" / "summary.json"
    assert artifact.exists()

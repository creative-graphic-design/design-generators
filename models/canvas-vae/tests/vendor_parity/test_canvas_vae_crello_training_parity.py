"""Crello model agreement with the pinned TensorFlow implementation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from laygen.common.testing import skip_or_fail_vendor_parity

pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]

DATA_DIR = Path(
    os.environ.get("CANVAS_VAE_DATA_DIR", ".cache/canvas-vae/crello/package-run-1")
)
FIXTURE_DIR = Path(
    os.environ.get("CANVAS_VAE_FIXTURE_DIR", ".cache/canvas-vae/crello/fixture")
)
REPORT_DIR = Path(
    os.environ.get(
        "CANVAS_VAE_REFERENCE_DIR", ".cache/canvas-vae/reference/crello-model"
    )
)
SCRIPT = Path("models/canvas-vae/scripts/compare_crello_training.py")


def _require_inputs(*paths: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        skip_or_fail_vendor_parity(
            "Crello model parity assets are missing.",
            missing_paths=missing,
            regeneration_hint=(
                "Run the Crello data contract commands in models/canvas-vae/REPRODUCING.md."
            ),
        )


def _run(phase: str) -> None:
    _require_inputs(
        *(DATA_DIR / f"{split}.jsonl" for split in ("train", "val", "test")),
        DATA_DIR / "vocabulary.json",
        FIXTURE_DIR / "manifest.json",
        FIXTURE_DIR / "manifest.sha256",
        FIXTURE_DIR / "posterior_means.npy",
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            phase,
            "--data-dir",
            str(DATA_DIR),
            "--fixture-dir",
            str(FIXTURE_DIR),
            "--report-dir",
            str(REPORT_DIR),
        ],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


def test_crello_s0_s3_distinct_batch_calibration() -> None:
    """Calibrate limits from three distinct canonical train batches."""
    _run("calibrate")


def test_crello_s0_s3_heldout_frozen_limits() -> None:
    """Check every canonical test batch against the frozen calibration limits."""
    _require_inputs(REPORT_DIR / "limits.json", REPORT_DIR / "limits.sha256")
    _run("heldout")

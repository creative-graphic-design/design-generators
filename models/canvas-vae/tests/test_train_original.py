"""Tests for the original CanvasVAE training adapter arguments."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "train_original.py"
SPEC = importlib.util.spec_from_file_location("train_original", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRAIN_ORIGINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_ORIGINAL)


def test_original_adapter_passes_crello_document_dataset_and_paths() -> None:
    arguments = TRAIN_ORIGINAL.build_trainer_arguments(
        "crello-document",
        Path("data/crello-document"),
        Path("jobs/crello-seed1"),
        ["--batch-size", "1024", "--num-epochs", "500"],
    )

    assert arguments == [
        "--dataset-name",
        "crello-document",
        "--data-dir",
        "data/crello-document",
        "--job-dir",
        "jobs/crello-seed1",
        "--batch-size",
        "1024",
        "--num-epochs",
        "500",
    ]

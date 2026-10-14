"""Tests for the original CanvasVAE training adapter arguments."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_tensorflow_tf32_helper_disables_and_checks_framework_flag() -> None:
    enabled = True

    def set_tf32(value: bool) -> None:
        nonlocal enabled
        enabled = value

    def is_tf32_enabled() -> bool:
        return enabled

    tf = SimpleNamespace(
        config=SimpleNamespace(
            experimental=SimpleNamespace(
                enable_tensor_float_32_execution=set_tf32,
                tensor_float_32_execution_enabled=is_tf32_enabled,
            )
        )
    )
    scripts = str(SCRIPT.parent)
    sys.path.insert(0, scripts)
    try:
        from generate_vendor_reference import disable_tensorflow_tf32

        disable_tensorflow_tf32(tf)
    finally:
        sys.path.remove(scripts)

    assert enabled is False


def test_tensorflow_tf32_helper_fails_if_framework_keeps_tf32_enabled() -> None:
    tf = SimpleNamespace(
        config=SimpleNamespace(
            experimental=SimpleNamespace(
                enable_tensor_float_32_execution=lambda _value: None,
                tensor_float_32_execution_enabled=lambda: True,
            )
        )
    )
    scripts = str(SCRIPT.parent)
    sys.path.insert(0, scripts)
    try:
        from generate_vendor_reference import disable_tensorflow_tf32

        with pytest.raises(RuntimeError, match="must be disabled"):
            disable_tensorflow_tf32(tf)
    finally:
        sys.path.remove(scripts)

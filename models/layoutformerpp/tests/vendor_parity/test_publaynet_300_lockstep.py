"""CPU-only checks for the lockstep trace helpers."""

import importlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LOCKSTEP = importlib.import_module("run_publaynet_300_lockstep")


def test_batch_hash_and_stream_parity_are_cpu_only() -> None:
    reference = {
        "names": ("sample",),
        "task_names": ("label",),
        "task_ids": (1,),
        "input_bytes": (b"input",),
        "output_bytes": (b"output",),
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[True, True]]),
        "labels": torch.tensor([[2, 1]]),
    }
    package = dict(reference)
    assert LOCKSTEP._batch_sha256(reference) == LOCKSTEP._batch_sha256(package)
    LOCKSTEP._equal_batches(reference, package)
    normalized = LOCKSTEP._batch(reference, torch.device("cpu"))
    assert normalized["task_ids"] is not None
    assert torch.equal(normalized["task_ids"], torch.tensor([1], dtype=torch.long))
    package["input_ids"] = torch.tensor([[1, 3]])
    with pytest.raises(RuntimeError, match="input_ids"):
        LOCKSTEP._equal_batches(reference, package)

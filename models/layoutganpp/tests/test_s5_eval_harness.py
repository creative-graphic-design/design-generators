from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from torch_geometric.data import Data, DataLoader as PyGDataLoader

SCRIPTS = Path(__file__).parents[1] / "scripts"
VENDOR_SPEC = importlib.util.spec_from_file_location(
    "layoutganpp_run_vendor_full", SCRIPTS / "run_vendor_full.py"
)
assert VENDOR_SPEC is not None and VENDOR_SPEC.loader is not None
VENDOR_RUN = importlib.util.module_from_spec(VENDOR_SPEC)
VENDOR_SPEC.loader.exec_module(VENDOR_RUN)


def test_seeded_vendor_loader_uses_pyg_factory_and_batches_pyg_data() -> None:
    examples = [Data(x=torch.tensor([[1.0]])), Data(x=torch.tensor([[2.0]]))]
    loader = VENDOR_RUN.seeded_training_loader(PyGDataLoader, 42003)

    assert loader.__wrapped__ is PyGDataLoader
    shuffled = loader(examples, batch_size=2, shuffle=True)
    assert shuffled.generator is not None
    assert shuffled.generator.initial_seed() == 42003

    batch = next(iter(shuffled))
    assert isinstance(batch, Data)
    assert batch.x.shape == (2, 1)

    unshuffled = loader(examples, batch_size=2, shuffle=False)
    assert unshuffled.generator is None


def test_vendor_progress_record_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    path.write_text("old\n")

    VENDOR_RUN.write_progress(path, 50)

    assert json.loads(path.read_text())["updates"] == 50
    assert not path.with_suffix(".tmp").exists()


def test_vendor_harness_refuses_existing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(VENDOR_RUN, "VENDOR_WORK", tmp_path)
    output_dir = tmp_path / "output/magazine/LayoutGAN++/existing"
    output_dir.mkdir(parents=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_vendor_full.py",
            "--dataset",
            "magazine",
            "--seed",
            "42004",
            "--data-seed",
            "42003",
            "--iterations",
            "200000",
            "--run-name",
            "existing",
            "--progress-json",
            str(tmp_path / "progress.json"),
        ],
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        VENDOR_RUN.main()

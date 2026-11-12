from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.vendor_parity

SCRIPTS = Path(__file__).parents[2] / "scripts"
VENDOR_SPEC = importlib.util.spec_from_file_location(
    "layoutganpp_run_vendor_full", SCRIPTS / "run_vendor_full.py"
)
assert VENDOR_SPEC is not None and VENDOR_SPEC.loader is not None
VENDOR_RUN = importlib.util.module_from_spec(VENDOR_SPEC)
VENDOR_SPEC.loader.exec_module(VENDOR_RUN)


def test_seeded_vendor_loader_uses_pyg_factory_and_batches_pyg_data() -> None:
    import torch
    from torch_geometric.data import Data, DataLoader as PyGDataLoader

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

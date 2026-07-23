from pathlib import Path

import torch

from ralf import RalfRetrievalTable
from ralf.retrieval import (
    RalfRetrievedBatch,
    retrieved_batch_to_model_inputs,
    model_inputs_to_retrieved_batch,
)


def test_retrieval_table_save_lookup_round_trip(tmp_path: Path) -> None:
    table = RalfRetrievalTable({"query": [1, 2, 3]}, top_k=2)

    table.save_pretrained(tmp_path)
    loaded = RalfRetrievalTable.from_pretrained(tmp_path)

    assert loaded.lookup(["query"]).tolist() == [[1, 2]]


def test_retrieval_table_pads_short_rows_and_overrides_top_k(tmp_path: Path) -> None:
    table = RalfRetrievalTable({"query": [7]}, top_k=3)

    table.save_pretrained(tmp_path)
    loaded = RalfRetrievalTable.from_pretrained(tmp_path, top_k=2)

    assert loaded.lookup(["query"]).tolist() == [[7, -1]]


def test_retrieved_batch_vendor_adapters() -> None:
    batch = RalfRetrievedBatch(
        image=torch.zeros(1, 2, 3, 1, 1),
        saliency=torch.zeros(1, 2, 1, 1, 1),
        bbox=torch.ones(1, 2, 3, 4),
        labels=torch.zeros(1, 2, 3, dtype=torch.long),
        mask=torch.ones(1, 2, 3, dtype=torch.bool),
        indexes=torch.tensor([[5, 6]]),
    )

    vendor = retrieved_batch_to_model_inputs(batch)
    restored = model_inputs_to_retrieved_batch(vendor)

    assert restored.indexes is not None
    assert restored.indexes.tolist() == [[5, 6]]
    assert torch.equal(restored.bbox, batch.bbox)

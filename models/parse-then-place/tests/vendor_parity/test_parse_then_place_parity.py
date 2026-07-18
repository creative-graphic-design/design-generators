from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer  # ty: ignore[possibly-missing-import]

from parse_then_place import (
    ParseThenPlaceConfig,
    ParseThenPlaceForConditionalGeneration,
    ParseThenPlaceProcessor,
)


pytestmark = pytest.mark.vendor_parity


def test_vendor_reference_metadata_exists() -> None:
    reference_root = os.environ.get("PARSE_THEN_PLACE_REFERENCE_DIR")
    if reference_root is None:
        pytest.skip("PARSE_THEN_PLACE_REFERENCE_DIR is required for vendor parity")
    metadata = Path(reference_root) / "reference_metadata.json"
    if not metadata.exists():
        pytest.skip("Generate references before running vendor parity")
    assert metadata.read_text(encoding="utf-8")


def test_stage2_reference_matches_converted_place() -> None:
    reference_root = os.environ.get("PARSE_THEN_PLACE_REFERENCE_DIR")
    original_root = os.environ.get("PARSE_THEN_PLACE_ORIGINAL_ROOT")
    if reference_root is None or original_root is None:
        pytest.skip(
            "PARSE_THEN_PLACE_REFERENCE_DIR and PARSE_THEN_PLACE_ORIGINAL_ROOT are required"
        )
    reference_dir = Path(reference_root)
    metadata_path = reference_dir / "reference_metadata.json"
    reference_path = reference_dir / "stage2_reference.json"
    if not metadata_path.exists() or not reference_path.exists():
        pytest.skip("Generate stage2 references before running vendor parity")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    dataset_name = str(metadata["dataset_name"])
    stage2_mode = str(metadata["stage2_mode"])
    ckpt_dir = Path(original_root) / "ckpt" / dataset_name / "stage2" / stage2_mode
    torch.manual_seed(int(metadata["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(metadata["seed"]))
    tokenizer = T5Tokenizer.from_pretrained(ckpt_dir)
    placement = T5ForConditionalGeneration.from_pretrained(ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    placement.to(device)  # ty: ignore[invalid-argument-type]
    model = ParseThenPlaceForConditionalGeneration(
        ParseThenPlaceConfig(
            dataset_name=dataset_name,
            stage2_mode=stage2_mode,
            num_return_sequences=int(metadata["num_return_sequences"]),
            temperature=float(metadata["temperature"]),
        ),
        placement=placement,
    )
    processor = ParseThenPlaceProcessor(
        placement_tokenizer=tokenizer,
        dataset_name=dataset_name,
    )
    encoded = processor.encode_placement_inputs(reference["source_texts"])
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    generated_ids = model.place(
        input_ids,
        attention_mask=attention_mask,
        generation_max_length=int(metadata["max_length"]),
        num_return_sequences=int(metadata["num_return_sequences"]),
        temperature=float(metadata["temperature"]),
    )
    expected_ids = torch.tensor(reference["generated_ids"], device=generated_ids.device)
    assert torch.equal(generated_ids, expected_ids)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    grouped = [
        decoded[
            idx * int(metadata["num_return_sequences"]) : (idx + 1)
            * int(metadata["num_return_sequences"])
        ]
        for idx in range(len(reference["source_texts"]))
    ]
    assert grouped == reference["decoded"]

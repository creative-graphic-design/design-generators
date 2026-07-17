from pathlib import Path

import pytest
import torch

from layout_dm import LayoutDMPipeline
from layout_dm.tokenization_layout_dm import LayoutDMTokenizer

DATASETS = ("rico25", "publaynet")


def _paths(dataset: str) -> tuple[Path, Path]:
    root = Path(__file__).parents[4]
    fixtures = Path(__file__).parent / "fixtures" / dataset
    converted = root / ".cache" / "layout-dm" / "converted" / f"layoutdm-{dataset}"
    return fixtures, converted


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", DATASETS)
def test_tokenizer_parity(dataset: str):
    fixture_dir, converted_dir = _paths(dataset)
    fixture = fixture_dir / "tokenizer_io.pt"
    if not fixture.exists() or not converted_dir.exists():
        pytest.skip("LayoutDM parity fixtures and converted weights are local-only")
    data = torch.load(fixture, map_location="cpu")
    tokenizer = LayoutDMTokenizer.from_pretrained(converted_dir)
    encoded = tokenizer.encode_layout(
        bbox=data["bbox"], labels=data["labels"], mask=data["mask"]
    )
    decoded = tokenizer.decode_layout(encoded["input_ids"])
    assert torch.equal(encoded["input_ids"], data["input_ids"])
    assert torch.equal(encoded["attention_mask"], data["attention_mask"])
    assert torch.equal(decoded["labels"], data["decoded_labels"])
    assert torch.equal(decoded["mask"], data["decoded_mask"])
    assert torch.allclose(decoded["bbox"], data["decoded_bbox"], atol=1e-7, rtol=0)


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", DATASETS)
def test_denoiser_logits_parity(dataset: str):
    fixture_dir, converted_dir = _paths(dataset)
    fixture = fixture_dir / "denoiser_forward.pt"
    if not fixture.exists() or not converted_dir.exists():
        pytest.skip("LayoutDM parity fixtures and converted weights are local-only")
    data = torch.load(fixture, map_location="cpu")
    pipe = LayoutDMPipeline.from_pretrained(converted_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    with torch.no_grad():
        logits = pipe.denoiser(
            input_ids=data["input_ids"].to(device),
            timesteps=data["timesteps"].to(device),
        ).logits.cpu()
    assert torch.allclose(logits, data["logits"], atol=1e-5, rtol=1e-5)


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", DATASETS)
def test_deterministic_sequence_parity(dataset: str):
    fixture_dir, converted_dir = _paths(dataset)
    fixture = fixture_dir / "sample_unconditional.pt"
    if not fixture.exists() or not converted_dir.exists():
        pytest.skip("LayoutDM parity fixtures and converted weights are local-only")
    data = torch.load(fixture, map_location="cpu")
    pipe = LayoutDMPipeline.from_pretrained(converted_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    out = pipe(
        batch_size=int(data["batch_size"]),
        seed=int(data["seed"]),
        num_inference_steps=len(data["trajectory"]),
        sampling="deterministic",
        return_intermediates=True,
    )
    assert torch.equal(out.sequences, data["sequences"])

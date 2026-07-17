from pathlib import Path

import pytest
import torch

from const_layout import ConstLayoutForGeneration

DATASETS = ("rico", "publaynet", "magazine")


def _paths(dataset: str) -> tuple[Path, Path]:
    root = Path(__file__).parents[4]
    fixture = (
        root / ".cache" / "const-layout" / "fixtures" / dataset / "reference_seed0.pt"
    )
    converted = (
        root / ".cache" / "const-layout" / "converted" / f"const-layout-{dataset}"
    )
    return fixture, converted


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", DATASETS)
def test_layoutganpp_bbox_parity(dataset: str):
    fixture, converted_dir = _paths(dataset)
    if not fixture.exists() or not converted_dir.exists():
        pytest.skip("Const-layout parity fixtures and converted weights are local-only")
    data = torch.load(fixture, map_location="cpu")
    model = ConstLayoutForGeneration.from_pretrained(converted_dir).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        out = model.generate(
            labels=data["labels"].to(device),
            attention_mask=data["attention_mask"].to(device),
            latents=data["latents"].to(device),
        )
    torch.testing.assert_close(out.bbox, data["bbox"], atol=1e-6, rtol=1e-5)

from pathlib import Path
import os
from typing import cast

import pytest
import torch

from housegan import HouseGanPipeline


PARITY_ROOT = Path(".cache/housegan/parity/housegan-floorplan-d")
CONVERTED_ROOT = Path(".cache/housegan/converted/housegan-floorplan-d")


@pytest.mark.vendor_parity
def test_housegan_vendor_parity_outputs_match_converted_model():
    required = [
        PARITY_ROOT / "input_graphs.pt",
        PARITY_ROOT / "latents.pt",
        PARITY_ROOT / "forward_masks.pt",
        PARITY_ROOT / "decoded_layouts.pt",
        CONVERTED_ROOT / "config.json",
        CONVERTED_ROOT / "model.safetensors",
        CONVERTED_ROOT / "processor_config.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        if os.environ.get("PARITY_REQUIRE") == "1":
            pytest.fail(f"House-GAN vendor parity artifacts are missing: {missing}")
        pytest.skip(f"House-GAN vendor parity artifacts are missing: {missing}")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    input_graphs = torch.load(PARITY_ROOT / "input_graphs.pt", map_location="cpu")
    latents = torch.load(PARITY_ROOT / "latents.pt", map_location="cpu")
    expected_masks = torch.load(PARITY_ROOT / "forward_masks.pt", map_location="cpu")
    expected_layouts = torch.load(
        PARITY_ROOT / "decoded_layouts.pt", map_location="cpu"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = HouseGanPipeline.from_pretrained(CONVERTED_ROOT, local_files_only=True)
    pipe.to(device)

    with torch.no_grad():
        actual = pipe.model(
            latents=latents.to(device),
            node_features=input_graphs["node_features"].to(device),
            edges=input_graphs["edges"].to(device),
        ).masks.cpu()

    assert torch.equal(actual, expected_masks)

    decoded = pipe.processor.post_process_masks(
        actual,
        labels=input_graphs["labels"],
        edges=input_graphs["edges"],
        node_features=input_graphs["node_features"],
        output_type="dict",
    )
    assert torch.equal(cast(torch.Tensor, decoded["bbox"]), expected_layouts["bbox"])
    assert torch.equal(
        cast(torch.Tensor, decoded["labels"]), expected_layouts["labels"]
    )
    assert torch.equal(cast(torch.Tensor, decoded["mask"]), expected_layouts["mask"])
    assert decoded["id2label"] == expected_layouts["id2label"]

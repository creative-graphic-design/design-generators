from pathlib import Path

import pytest
import torch

from ds_gan import DSGANConfig, DSGANModel
from ds_gan.conversion import convert_vendor_state_dict


@pytest.mark.vendor_parity
def test_ds_gan_vendor_forward_parity():
    fixture = Path(".cache/ds-gan/fixtures/pku/reference_seed0.pt")
    checkpoint = Path(".cache/ds-gan/original/DS-GAN-Epoch300.pth")
    if not fixture.exists() or not checkpoint.exists():
        pytest.skip("DS-GAN vendor parity requires local weights and fixture")

    reference = torch.load(fixture, map_location="cpu")
    model = DSGANModel(DSGANConfig()).eval()
    model.load_state_dict(
        convert_vendor_state_dict(torch.load(checkpoint, map_location="cpu")),
        strict=True,
    )
    with torch.no_grad():
        output = model(
            pixel_values=reference["pixel_values"],
            layout=reference["initial_layout"],
        )

    torch.testing.assert_close(
        output.class_probs, reference["class_probs"], rtol=0, atol=0
    )
    torch.testing.assert_close(output.bbox, reference["bbox"], rtol=0, atol=0)

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
    if not torch.cuda.is_available():
        pytest.skip("DS-GAN vendor parity fixture is generated on CUDA")

    reference = torch.load(fixture, map_location="cpu")
    device = torch.device("cuda")
    model = DSGANModel(DSGANConfig()).eval().to(device)
    model.load_state_dict(
        convert_vendor_state_dict(torch.load(checkpoint, map_location=device)),
        strict=True,
    )
    class_probs = []
    boxes = []
    batch_size = int(reference["batch_size"])
    with torch.no_grad():
        for start in range(0, reference["pixel_values"].shape[0], batch_size):
            output = model(
                pixel_values=reference["pixel_values"][start : start + batch_size].to(
                    device
                ),
                layout=reference["initial_layout"][start : start + batch_size].to(
                    device
                ),
            )
            class_probs.append(output.class_probs.cpu())
            boxes.append(output.bbox.cpu())
    converted_class_probs = torch.cat(class_probs)
    converted_boxes = torch.cat(boxes)

    torch.testing.assert_close(
        converted_class_probs, reference["class_probs"], rtol=0, atol=0
    )
    torch.testing.assert_close(converted_boxes, reference["bbox"], rtol=0, atol=0)

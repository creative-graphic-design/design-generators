from pathlib import Path
from typing import cast

import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from laygen.modeling_outputs import LayoutGenerationOutput

from ds_gan import DSGANConfig, DSGANModel, DSGANProcessor
from ds_gan.conversion import convert_vendor_state_dict


@pytest.mark.vendor_parity
def test_ds_gan_vendor_forward_parity():
    fixture = Path(".cache/ds-gan/fixtures/pku/reference_seed0.pt")
    checkpoint = Path(".cache/ds-gan/original/DS-GAN-Epoch300.pth")
    if not fixture.exists() or not checkpoint.exists():
        skip_or_fail_vendor_parity(
            "DS-GAN vendor parity requires local weights and fixture",
            missing_paths=[fixture, checkpoint],
            regeneration_hint="run models/ds-gan/scripts/generate_reference_outputs.py after downloading DS-GAN weights",
        )
    if not torch.cuda.is_available():
        skip_or_fail_vendor_parity(
            "DS-GAN vendor parity fixture is generated on CUDA",
            missing_paths=["CUDA device"],
            regeneration_hint="rerun on a CUDA-enabled host with DS-GAN parity assets",
        )

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

    processor = DSGANProcessor()
    converted_public = processor.decode(
        class_probs=converted_class_probs,
        bbox=converted_boxes,
    )
    reference_public = processor.decode(
        class_probs=reference["class_probs"],
        bbox=reference["bbox"],
    )
    converted_public = cast(LayoutGenerationOutput, converted_public)
    reference_public = cast(LayoutGenerationOutput, reference_public)
    torch.testing.assert_close(
        converted_public.bbox, reference_public.bbox, rtol=0, atol=0
    )
    torch.testing.assert_close(
        converted_public.labels, reference_public.labels, rtol=0, atol=0
    )
    torch.testing.assert_close(
        converted_public.mask, reference_public.mask, rtol=0, atol=0
    )


@pytest.mark.vendor_parity
def test_ds_gan_processor_matches_vendor_fixture_pixels():
    fixture = Path(".cache/ds-gan/fixtures/pku/reference_seed0.pt")
    dataset_root = Path(".cache/ds-gan/original/Dataset/test")
    if not fixture.exists() or not dataset_root.exists():
        skip_or_fail_vendor_parity(
            "DS-GAN processor parity requires local dataset fixture",
            missing_paths=[fixture, dataset_root],
            regeneration_hint="download the DS-GAN dataset fixture and run models/ds-gan/scripts/generate_reference_outputs.py",
        )

    reference = torch.load(fixture, map_location="cpu")
    processor = DSGANProcessor()
    image_paths = sorted(
        (dataset_root / "image_canvas").glob("*.png"), key=lambda p: p.name
    )
    max_abs = 0.0
    mismatched = 0
    batch_size = int(reference["batch_size"])
    for start in range(0, len(image_paths), batch_size):
        batch = image_paths[start : start + batch_size]
        encoded = processor(
            [str(path) for path in batch],
            saliency_pfpnet=[
                str(
                    dataset_root
                    / "saliencymaps_pfpn"
                    / path.name.replace(".png", "_pred.png")
                )
                for path in batch
            ],
            saliency_basnet=[
                str(dataset_root / "saliencymaps_basnet" / path.name) for path in batch
            ],
        )
        expected = reference["pixel_values"][start : start + len(batch)]
        diff = (encoded["pixel_values"] - expected).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        mismatched += int((diff != 0).sum().item())
    assert max_abs == 0.0
    assert mismatched == 0

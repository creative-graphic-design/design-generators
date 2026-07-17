from pathlib import Path
import sys
from typing import cast

import pytest
import torch

from laygen.common.outputs_diffusers import LayoutGenerationOutput

from lace import build_pipeline_from_vendor_checkpoint, default_model_config
from lace.conversion import convert_state_dict, load_vendor_state_dict
from lace.modeling_lace import LaceTransformerModel


@pytest.mark.vendor_parity
@pytest.mark.parametrize(
    ("dataset", "checkpoint"),
    [
        ("publaynet", "publaynet_best.pt"),
        ("rico13", "rico13_best.pt"),
        ("rico25", "rico25_best.pt"),
    ],
)
def test_checkpoint_conversion_smoke(dataset: str, checkpoint: str) -> None:
    root = Path(__file__).parents[4]
    path = root / ".cache" / "lace" / "original" / "model" / checkpoint
    if not path.exists():
        pytest.skip("LACE vendor checkpoint is local-only")
    pipe = build_pipeline_from_vendor_checkpoint(dataset, path)
    out = cast(
        LayoutGenerationOutput, pipe(batch_size=1, seed=0, num_inference_steps=2)
    )
    assert out.bbox.shape == (1, 25, 4)
    assert out.labels.shape == (1, 25)
    assert torch.all((0 <= out.bbox) & (out.bbox <= 1))


@pytest.mark.vendor_parity
@pytest.mark.parametrize(
    ("dataset", "checkpoint"),
    [
        ("publaynet", "publaynet_best.pt"),
        ("rico25", "rico25_best.pt"),
    ],
)
def test_denoiser_logits_match_vendor(dataset: str, checkpoint: str) -> None:
    root = Path(__file__).parents[4]
    path = root / ".cache" / "lace" / "original" / "model" / checkpoint
    if not path.exists():
        pytest.skip("LACE vendor checkpoint is local-only")
    sys.path.insert(0, str(root / "vendor" / "lace"))
    try:
        from util.backbone import TransformerEncoder
    finally:
        sys.path.pop(0)
    config = default_model_config(dataset)
    device = torch.device("cpu")
    vendor = TransformerEncoder(
        num_layers=config["num_layers"],
        dim_seq=config["seq_dim"],
        dim_transformer=config["dim_transformer"],
        nhead=config["nhead"],
        dim_feedforward=config["dim_feedforward"],
        diffusion_step=config["diffusion_step"],
        device=device,
    ).eval()
    model = LaceTransformerModel(**config).eval()
    state = convert_state_dict(load_vendor_state_dict(path))
    vendor.load_state_dict(state, strict=True)
    model.load_state_dict(state, strict=True)
    generator = torch.Generator(device=device).manual_seed(123)
    sample = torch.randn(
        2,
        config["max_seq_length"],
        config["seq_dim"],
        device=device,
        generator=generator,
    )
    timestep = torch.tensor([1, 201], device=device)
    with torch.no_grad():
        expected = vendor(sample, timestep=timestep)
        actual = model(sample=sample, timestep=timestep).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

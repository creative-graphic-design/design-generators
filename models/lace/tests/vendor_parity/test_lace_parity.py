from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import cast

import pytest
import torch

from laygen.common.vendor import vendor_root
from laygen.common.outputs_diffusers import LayoutGenerationOutput

from lace import build_pipeline_from_vendor_checkpoint, default_model_config
from lace.conversion import convert_state_dict, load_vendor_state_dict
from lace.modeling_lace import LaceTransformerModel


def _load_vendor_transformer_encoder(vendor_dir: Path) -> type[torch.nn.Module]:
    spec = spec_from_file_location(
        "lace_vendor_backbone", vendor_dir / "util" / "backbone.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load vendor LACE backbone")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    transformer = getattr(module, "TransformerEncoder")
    if not isinstance(transformer, type):
        raise TypeError("Vendor TransformerEncoder must be a class")
    return cast(type[torch.nn.Module], transformer)


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
    vendor_dir = vendor_root("lace", marker=Path("util") / "backbone.py")
    TransformerEncoder = _load_vendor_transformer_encoder(vendor_dir)
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

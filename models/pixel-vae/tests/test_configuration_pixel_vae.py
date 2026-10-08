from __future__ import annotations

import pytest

from pixel_vae import PixelVAEConfig


def test_pixel_vae_config_records_production_defaults() -> None:
    config = PixelVAEConfig()
    assert config.model_type == "pixel-vae"
    assert config.latent_dim == 256
    assert config.quantize_factor == 4
    assert config.kl_weight == 100.0
    assert config.l2_weight == 1e-6


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"latent_dim": 0}, "latent_dim"),
        ({"quantize_factor": 9}, "quantize_factor"),
        ({"kl_weight": -1}, "loss weights"),
        ({"l2_weight": -1}, "loss weights"),
        ({"quantize_factor": -1}, "quantize_factor"),
        ({"batch_norm_epsilon": 0}, "batch_norm_epsilon"),
        ({"backbone_batch_norm_momentum": 1}, "backbone_batch_norm_momentum"),
        ({"decoder_batch_norm_momentum": -0.1}, "decoder_batch_norm_momentum"),
    ],
)
def test_pixel_vae_config_rejects_invalid_values(
    kwargs: dict[str, int | float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PixelVAEConfig(**kwargs)

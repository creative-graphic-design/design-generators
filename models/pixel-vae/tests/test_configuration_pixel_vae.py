from __future__ import annotations

from collections.abc import Callable

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
    ("create_config", "message"),
    [
        (lambda: PixelVAEConfig(latent_dim=0), "latent_dim"),
        (lambda: PixelVAEConfig(quantize_factor=9), "quantize_factor"),
        (lambda: PixelVAEConfig(kl_weight=-1), "loss weights"),
        (lambda: PixelVAEConfig(l2_weight=-1), "loss weights"),
        (lambda: PixelVAEConfig(quantize_factor=-1), "quantize_factor"),
        (lambda: PixelVAEConfig(batch_norm_epsilon=0), "batch_norm_epsilon"),
        (
            lambda: PixelVAEConfig(backbone_batch_norm_momentum=1),
            "backbone_batch_norm_momentum",
        ),
        (
            lambda: PixelVAEConfig(decoder_batch_norm_momentum=-0.1),
            "decoder_batch_norm_momentum",
        ),
    ],
)
def test_pixel_vae_config_rejects_invalid_values(
    create_config: Callable[[], PixelVAEConfig], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_config()

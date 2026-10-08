from __future__ import annotations

import torch
import pytest

from pixel_vae import PixelVAEConfig, PixelVAEEncoder, PixelVAEModel
from pixel_vae.modeling_pixel_vae import (
    PixelVAEBatchNorm2d,
    pixelvae_reconstruction_loss,
)


def test_mobile_net_v2_topology_and_posterior_mean_shape() -> None:
    model = PixelVAEModel(PixelVAEConfig(latent_dim=8)).eval()
    assert len(model.encoder.backbone.layers) == 104
    image = torch.zeros(1, 4, 256, 256)

    with torch.inference_mode():
        output = model(image, sample=False)
        mean = model.encode(image)

    assert output.logits is not None
    assert tuple(output.logits.shape) == (1, 256, 256, 4, 16)
    assert tuple(output.posterior_mean.shape) == (1, 8)
    assert tuple(mean.shape) == (1, 8)
    assert mean.dtype is torch.float32
    assert torch.isfinite(output.loss)


def test_encoder_can_be_saved_and_loaded_locally(tmp_path) -> None:
    encoder = PixelVAEEncoder(PixelVAEConfig(latent_dim=8)).eval()
    encoder.save_pretrained(tmp_path)
    restored = PixelVAEEncoder.from_pretrained(tmp_path).eval()
    assert restored.config.latent_dim == 8
    assert all(
        torch.equal(value, restored.state_dict()[key])
        for key, value in encoder.state_dict().items()
    )


def test_model_can_be_saved_and_loaded_locally(tmp_path) -> None:
    model = PixelVAEModel(PixelVAEConfig(latent_dim=8)).eval()
    model.save_pretrained(tmp_path)
    restored = PixelVAEModel.from_pretrained(tmp_path).eval()
    assert restored.config.latent_dim == 8
    assert all(
        torch.equal(value, restored.state_dict()[key])
        for key, value in model.state_dict().items()
    )


def test_model_sampling_uses_fixed_noise_and_validates_shape() -> None:
    model = PixelVAEModel(PixelVAEConfig(latent_dim=8)).eval()
    image = torch.zeros(1, 4, 256, 256)
    epsilon = torch.ones(1, 8)

    with torch.inference_mode():
        output = model(image, sample=True, epsilon=epsilon)
    assert output.latents is not None
    assert not torch.equal(output.latents, output.posterior_mean)
    with pytest.raises(ValueError, match="same shape"):
        model(image, sample=True, epsilon=torch.ones(1, 7))
    with pytest.raises(ValueError, match="only valid"):
        model(image, sample=False, epsilon=epsilon)


def test_model_sampling_uses_shared_randomness_when_noise_is_not_supplied(
    monkeypatch,
) -> None:
    model = PixelVAEModel(PixelVAEConfig(latent_dim=8)).eval()
    image = torch.zeros(1, 4, 256, 256)
    monkeypatch.setattr(
        "pixel_vae.modeling_pixel_vae.randn",
        lambda shape, *, device, dtype: torch.ones(shape, device=device, dtype=dtype),
    )

    with torch.inference_mode():
        output = model(image, sample=True)

    assert output.latents is not None and output.posterior_mean is not None
    assert torch.all(output.latents > output.posterior_mean)


def test_model_rejects_invalid_inputs() -> None:
    model = PixelVAEModel(PixelVAEConfig(latent_dim=8)).eval()
    with pytest.raises(TypeError, match="torch.float32"):
        model(torch.zeros(1, 4, 256, 256, dtype=torch.float64))
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(1, 3, 256, 256))


def test_keras_batch_norm_uses_biased_variance_and_ema() -> None:
    norm = PixelVAEBatchNorm2d(1, eps=1e-3, momentum=0.5).train()
    values = torch.tensor([[[[1.0, 3.0]]]])
    normalized = norm(values)
    torch.testing.assert_close(
        normalized, torch.tensor([[[[-1.0, 1.0]]]]), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(norm.running_mean, torch.tensor([1.0]))
    torch.testing.assert_close(norm.running_var, torch.tensor([1.0]))

    norm.eval()
    torch.testing.assert_close(norm(values), torch.tensor([[[[0.0, 2.0]]]]))


def test_reconstruction_masks_rgb_for_transparent_targets() -> None:
    pixels = torch.zeros(1, 4, 256, 256)
    logits = torch.zeros(1, 256, 256, 4, 16)
    baseline = pixelvae_reconstruction_loss(pixels, logits)
    logits[:, :, :, :3, 1] = 10
    changed_rgb = pixelvae_reconstruction_loss(pixels, logits)
    torch.testing.assert_close(changed_rgb, baseline)
    logits[:, :, :, 3, 0] = 10
    changed_alpha = pixelvae_reconstruction_loss(pixels, logits)
    assert changed_alpha < baseline


def test_reconstruction_rejects_wrong_shapes() -> None:
    pixels = torch.zeros(1, 4, 256, 256)
    with pytest.raises(ValueError, match="logits"):
        pixelvae_reconstruction_loss(pixels, torch.zeros(1, 1))
    with pytest.raises(ValueError, match="pixel_values"):
        pixelvae_reconstruction_loss(torch.zeros(1, 3, 1, 1), torch.zeros(1, 1))

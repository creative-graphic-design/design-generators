from __future__ import annotations

import numpy as np
import pytest
import torch

from pixel_vae.modeling_pixel_vae import PixelVAEBatchNorm2d

pytestmark = pytest.mark.vendor_parity


def test_tf_fused_batch_norm_moving_variance_matches_package() -> None:
    tf = pytest.importorskip("tensorflow")
    values = np.linspace(-2.0, 3.0, num=2 * 4 * 4 * 3, dtype=np.float32).reshape(
        2, 4, 4, 3
    )
    source = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.5)
    source(values, training=True)
    assert source.fused is True

    package = PixelVAEBatchNorm2d(3, eps=1e-3, momentum=0.5).train()
    package_values = torch.from_numpy(values).permute(0, 3, 1, 2).contiguous()
    package(package_values)

    np.testing.assert_allclose(
        package.running_var.numpy(),
        source.moving_variance.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pixel_vae import PixelVAEConfig, PixelVAEModel
from pixel_vae.conversion import (
    convert_tensorflow_variables,
    extract_tensorflow_variable_map,
    extract_tensorflow_weights,
    tensorflow_state_sha256,
    tensorflow_weight_map,
)


class Dense:
    name = "z_mean"
    weights = []

    def get_weights(self) -> list[np.ndarray]:
        return [np.ones((2, 3), dtype=np.float32), np.zeros(3, dtype=np.float32)]


class BatchNormalization:
    name = "bn"
    weights = []

    def get_weights(self) -> list[np.ndarray]:
        return [np.ones(2, dtype=np.float32) * value for value in (1, 0, 0, 1)]


class InputLayer:
    name = "input"

    def get_weights(self) -> list[np.ndarray]:
        return []


class UnsupportedWeightedLayer:
    name = "unsupported"

    def get_weights(self) -> list[np.ndarray]:
        return [np.ones(1, dtype=np.float32)]


def source_model(backbone_layers: list[object]) -> SimpleNamespace:
    return SimpleNamespace(
        encoder=SimpleNamespace(
            cnn=SimpleNamespace(layers=backbone_layers),
            head=SimpleNamespace(z_mean=Dense(), z_log_sigma=Dense()),
        ),
        decoder=SimpleNamespace(
            dense=Dense(), cnn=SimpleNamespace(layers=[BatchNormalization()])
        ),
    )


def test_extract_tensorflow_weights_names_all_weighted_layers() -> None:
    weights = extract_tensorflow_weights(
        source_model([InputLayer(), BatchNormalization()])
    )
    assert "backbone/bn/gamma" in weights
    assert "encoder/head/z_mean/kernel" in weights
    assert "encoder/head/z_log_sigma/bias" in weights
    assert "decoder/dense/kernel" in weights
    assert "decoder/cnn/0/moving_variance" in weights


def test_extract_tensorflow_weights_rejects_unknown_weighted_layers() -> None:
    with pytest.raises(ValueError, match="unsupported weighted"):
        extract_tensorflow_weights(source_model([UnsupportedWeightedLayer()]))


def test_extract_tensorflow_variable_map_uses_source_variable_names() -> None:
    class Variable:
        def __init__(self, value: float) -> None:
            self.value = np.asarray([value], dtype=np.float32)

        def numpy(self) -> np.ndarray:
            return self.value

    dense = Dense()
    dense.weights = [Variable(1), Variable(2)]
    model = source_model([])
    model.encoder.head.z_mean = dense
    model.encoder.head.z_log_sigma.weights = [Variable(3), Variable(4)]
    model.decoder.dense.weights = [Variable(5), Variable(6)]
    model.decoder.cnn.layers[0].weights = [Variable(value) for value in (1, 2, 3, 4)]

    variables = extract_tensorflow_variable_map(model)

    assert variables["encoder/head/z_mean/kernel"].numpy().tolist() == [1]
    assert variables["encoder/head/z_mean/bias"].numpy().tolist() == [2]


def test_tensorflow_state_digest_ignores_mapping_order() -> None:
    first = {
        "b": np.asarray([2], dtype=np.float32),
        "a": np.asarray([1], dtype=np.float32),
    }
    second = dict(reversed(list(first.items())))
    assert tensorflow_state_sha256(first) == tensorflow_state_sha256(second)
    assert len(tensorflow_state_sha256(first)) == 64


def test_weight_map_covers_mobile_net_and_decoder() -> None:
    model = PixelVAEModel(PixelVAEConfig(latent_dim=8))
    mapping = tensorflow_weight_map(model)
    sources = {assignment.source_key for assignment in mapping}
    targets = {assignment.target_key for assignment in mapping}
    assert "backbone/Conv1/kernel" in sources
    assert "backbone/expanded_conv_depthwise/depthwise_kernel" in sources
    assert "encoder/head/z_log_sigma/kernel" in sources
    assert "decoder/cnn/1/kernel" in sources
    assert "decoder/cnn/11/kernel" in sources
    assert "encoder.backbone.layers.block_16_project_BN.running_var" in targets


def test_convert_tensorflow_variables_applies_all_kernel_layouts() -> None:
    model = PixelVAEModel(PixelVAEConfig(latent_dim=8))
    weights = {
        "backbone/Conv1/kernel": np.arange(3 * 3 * 4 * 32, dtype=np.float32).reshape(
            3, 3, 4, 32
        ),
        "backbone/expanded_conv_depthwise/depthwise_kernel": np.arange(
            3 * 3 * 32, dtype=np.float32
        ).reshape(3, 3, 32, 1),
        "backbone/bn_Conv1/gamma": np.full(32, 3, dtype=np.float32),
        "encoder/head/z_mean/kernel": np.arange(1280 * 8, dtype=np.float32).reshape(
            1280, 8
        ),
        "decoder/cnn/1/kernel": np.arange(2 * 2 * 256 * 32, dtype=np.float32).reshape(
            2, 2, 256, 32
        ),
        "decoder/cnn/11/kernel": np.arange(3 * 3 * 64 * 64, dtype=np.float32).reshape(
            3, 3, 64, 64
        ),
    }
    report = convert_tensorflow_variables(weights, model, strict=False)

    assert report.assigned_tensors == 6
    assert len(report.missing_source_keys) > 0
    assert not report.unexpected_source_keys
    torch.testing.assert_close(
        model.encoder.backbone.layers["Conv1"].weight,
        torch.from_numpy(weights["backbone/Conv1/kernel"].transpose(3, 2, 0, 1).copy()),
    )
    torch.testing.assert_close(
        model.encoder.backbone.layers["expanded_conv_depthwise"].weight,
        torch.from_numpy(
            weights["backbone/expanded_conv_depthwise/depthwise_kernel"]
            .transpose(2, 3, 0, 1)
            .copy()
        ),
    )
    torch.testing.assert_close(
        model.encoder.z_mean.weight,
        torch.from_numpy(weights["encoder/head/z_mean/kernel"].T.copy()),
    )
    torch.testing.assert_close(
        model.decoder.transpose_convolutions[0].weight,
        torch.from_numpy(weights["decoder/cnn/1/kernel"].transpose(3, 2, 0, 1).copy()),
    )
    torch.testing.assert_close(
        model.decoder.output_convolution.weight,
        torch.from_numpy(weights["decoder/cnn/11/kernel"].transpose(3, 2, 0, 1).copy()),
    )
    torch.testing.assert_close(
        model.encoder.backbone.layers["bn_Conv1"].weight,
        torch.full((32,), 3.0),
    )


def test_converter_rejects_incomplete_strict_mapping() -> None:
    model = PixelVAEModel(PixelVAEConfig(latent_dim=8))
    before = model.encoder.backbone.layers["Conv1"].weight.clone()
    with pytest.raises(ValueError, match="keys differ"):
        convert_tensorflow_variables({}, model)
    torch.testing.assert_close(model.encoder.backbone.layers["Conv1"].weight, before)


def test_converter_rejects_duplicate_target_keys(monkeypatch) -> None:
    model = PixelVAEModel(PixelVAEConfig(latent_dim=8))
    assignment = tensorflow_weight_map(model)[0]
    monkeypatch.setattr(
        "pixel_vae.conversion.tensorflow_weight_map",
        lambda _: (assignment, assignment),
    )

    with pytest.raises(ValueError, match="more than once"):
        convert_tensorflow_variables({}, model, strict=False)


def test_converter_rejects_shape_mismatch_before_copy() -> None:
    model = PixelVAEModel(PixelVAEConfig(latent_dim=8))
    before = model.encoder.backbone.layers["Conv1"].weight.clone()
    with pytest.raises(ValueError, match="shape mismatch"):
        convert_tensorflow_variables(
            {"backbone/Conv1/kernel": np.zeros((1,), dtype=np.float32)},
            model,
            strict=False,
        )
    torch.testing.assert_close(model.encoder.backbone.layers["Conv1"].weight, before)

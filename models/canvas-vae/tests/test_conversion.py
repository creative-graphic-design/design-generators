import numpy as np
import pytest
import torch

from canvas_vae import CanvasVAEModel
from canvas_vae.conversion import (
    VARIABLE_SUFFIX,
    convert_tensorflow_variables,
    load_lightning_state_dict,
    tensorflow_key_map,
)


def tensorflow_variables(model, config):
    state = model.state_dict()
    variables = {}
    for key, source in tensorflow_key_map(config).items():
        value = state[key].numpy()
        variables[source.key + VARIABLE_SUFFIX] = value.T if source.transpose else value

    variables["optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE"] = np.zeros(())
    variables[
        "encoder/norm/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE"
    ] = np.zeros(16)
    return variables


def test_key_map_covers_state_dict(make_config):
    config = make_config(num_blocks=2)
    model = CanvasVAEModel(config)
    assert set(tensorflow_key_map(config)) == set(model.state_dict())


def test_round_trip_and_rejections(config):
    model = CanvasVAEModel(config)
    variables = tensorflow_variables(model, config)
    state = convert_tensorflow_variables(variables, config)
    restored = CanvasVAEModel(config)
    restored.load_state_dict(state, strict=True)
    for key, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value), key

    with pytest.raises(KeyError, match="unexpected"):
        convert_tensorflow_variables(
            {**variables, "encoder/seq2seq/seq2seq_0/norm3/gamma": np.ones(16)}, config
        )

    missing = dict(variables)
    missing.pop("encoder/norm/moving_mean" + VARIABLE_SUFFIX)
    with pytest.raises(KeyError, match="missing"):
        convert_tensorflow_variables(missing, config)


def test_load_lightning_state_dict(tmp_path, config):
    model = CanvasVAEModel(config)
    payload = {
        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}
        | {"other.x": torch.zeros(1)}
    }
    torch.save(payload, tmp_path / "last.ckpt")
    state = load_lightning_state_dict(tmp_path / "last.ckpt")
    assert set(state) == set(model.state_dict())

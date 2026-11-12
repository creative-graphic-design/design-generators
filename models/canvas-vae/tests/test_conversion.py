import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from canvas_vae import CanvasVAEModel
from canvas_vae import CanvasVAECrelloConfig, CanvasVAECrelloModel
from canvas_vae.conversion import (
    VARIABLE_SUFFIX,
    convert_tensorflow_crello_variables,
    convert_tensorflow_variables,
    load_lightning_state_dict,
    tensorflow_key_map,
    tensorflow_crello_key_map,
)
from canvas_vae.data import load_crello_vocabularies


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


def test_crello_key_map_and_tensorflow_conversion_round_trip():
    config = CanvasVAECrelloConfig(
        vocabularies={
            "length": [1, 2, 3],
            "group": ["", "group"],
            "format": ["", "poster"],
            "canvas_width": [640],
            "canvas_height": [480],
            "category": ["", "food"],
            "type": [
                "",
                "textElement",
                "coloredBackground",
                "svgElement",
                "imageElement",
                "maskElement",
            ],
        },
        max_length=3,
        latent_dim=16,
        num_heads=2,
    )
    model = CanvasVAECrelloModel(config)
    mapping = tensorflow_crello_key_map(config)
    assert set(mapping) == set(model.state_dict())
    variables = {}
    for key, source in mapping.items():
        value = model.state_dict()[key].numpy()
        variables[source.key + VARIABLE_SUFFIX] = value.T if source.transpose else value

    converted = convert_tensorflow_crello_variables(variables, config)
    restored = CanvasVAECrelloModel(config)
    restored.load_state_dict(converted, strict=True)
    for key, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value), key

    with pytest.raises(KeyError, match="missing"):
        convert_tensorflow_crello_variables(
            {
                key: value
                for key, value in variables.items()
                if "encoder/norm/gamma" not in key
            },
            config,
        )
    with pytest.raises(KeyError, match="unexpected"):
        convert_tensorflow_crello_variables(
            {**variables, "unexpected/model/variable": np.zeros(1)}, config
        )


def test_crello_lightning_checkpoint_saves_pretrained_model(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vocabularies = {
        "group": {"group": 1},
        "format": {"poster": 1},
        "canvas_width": {"640": 1},
        "canvas_height": {"480": 1},
        "category": {"food": 1},
        "type": {
            value: 1
            for value in (
                "textElement",
                "coloredBackground",
                "svgElement",
                "imageElement",
                "maskElement",
                "otherElement",
            )
        },
    }
    (data_dir / "vocabulary.json").write_text(json.dumps(vocabularies))
    model = CanvasVAECrelloModel(
        CanvasVAECrelloConfig(vocabularies=load_crello_vocabularies(data_dir))
    )
    checkpoint = tmp_path / "last.ckpt"
    torch.save(
        {
            "state_dict": {
                f"model.{key}": value for key, value in model.state_dict().items()
            }
        },
        checkpoint,
    )
    output_dir = tmp_path / "converted"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "convert_original_checkpoint.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "crello",
            "--checkpoint",
            str(checkpoint),
            "--vocabulary",
            str(data_dir / "vocabulary.json"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    loaded = CanvasVAECrelloModel.from_pretrained(output_dir)
    assert set(loaded.state_dict()) == set(model.state_dict())
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key

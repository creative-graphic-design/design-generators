from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("lightning")

from layoutdiffusion import LayoutDiffusionConfig, LayoutDiffusionTransformer
from layoutdiffusion.training.lightning_module import (
    EMA_CHECKPOINT_KEY,
    LayoutDiffusionTrainingModule,
)

pytestmark = pytest.mark.training

SCRIPT_PATH = Path(
    "models/layoutdiffusion/scripts/export_training_checkpoint_samples.py"
)
CheckpointValue = (
    dict[str, torch.Tensor] | torch.Tensor | int | float | str | bool | None
)


def load_export_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "export_training_checkpoint_samples", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_config() -> LayoutDiffusionConfig:
    return LayoutDiffusionConfig(
        dataset_name="publaynet",
        seq_length=19,
        max_num_elements=3,
        diffusion_steps=10,
        num_channels=8,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=32,
        max_position_embeddings=19,
    )


def test_format_vendor_json_lines_matches_vendor_decoder_shape() -> None:
    exporter = load_export_module()
    format_vendor_json_lines = getattr(exporter, "format_vendor_json_lines")
    assert format_vendor_json_lines(["START text 0 1 2 3 END PAD"]) == [
        '["START text 0 1 2 3 END PAD"]\n'
    ]


def test_export_training_checkpoint_samples_writes_json_lines(tmp_path: Path) -> None:
    exporter = load_export_module()
    config = tiny_config()
    model = LayoutDiffusionTransformer(
        vocab_size=config.vocab_size,
        num_channels=config.num_channels,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        dropout=config.dropout,
        max_position_embeddings=config.max_token_length,
    )
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(
        {
            EMA_CHECKPOINT_KEY: {
                name: param.detach().clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
        },
        checkpoint_path,
    )
    config_path = tmp_path / "layoutdiffusion_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_name": "publaynet",
                "seq_length": 19,
                "max_num_elements": 3,
                "diffusion_steps": 10,
                "num_channels": 8,
                "hidden_size": 16,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "max_position_embeddings": 19,
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "samples.json"

    export_training_checkpoint_samples = getattr(
        exporter, "export_training_checkpoint_samples"
    )
    export_training_checkpoint_samples(
        checkpoint_path=checkpoint_path,
        dataset="publaynet",
        config_path=config_path,
        output_path=output_path,
        num_samples=2,
        batch_size=1,
        seed=3,
        device_name="cpu",
        sampling_name="argmax",
        num_inference_steps=1,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    decoded = [json.loads(line) for line in lines]
    assert all(isinstance(row, list) for row in decoded)
    assert all(len(row) == 1 for row in decoded)
    assert all(isinstance(row[0], str) and row[0] for row in decoded)


def test_training_module_persists_and_loads_ema_checkpoint() -> None:
    module = LayoutDiffusionTrainingModule(config=tiny_config(), scheduler=None)
    checkpoint: dict[str, CheckpointValue] = {}
    module.on_save_checkpoint(checkpoint)
    assert EMA_CHECKPOINT_KEY in checkpoint

    restored = LayoutDiffusionTrainingModule(config=tiny_config(), scheduler=None)
    restored.on_load_checkpoint(checkpoint)
    for key, value in module.ema_state_dict().items():
        assert torch.equal(restored.ema_state_dict()[key], value)

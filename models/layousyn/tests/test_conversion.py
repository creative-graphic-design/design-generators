import json

import torch

from layousyn import LayouSynPipeline
from layousyn.configuration_layousyn import LayouSynConfig
from layousyn.conversion import convert_checkpoint
from layousyn.modeling_layousyn import LayouSynDiTModel


def test_convert_checkpoint_smoke(tmp_path) -> None:
    cfg = LayouSynConfig(
        model_name="DiT-D1-H32-N1",
        concept_in_channels=4,
        y_in_channels=4,
        max_in_len=2,
        max_y_len=3,
        diffusion_steps=2,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg.to_reference_dict()))
    model = LayouSynDiTModel(
        model_name="DiT-D1-H32-N1",
        concept_in_channels=4,
        y_in_channels=4,
        max_in_len=2,
        max_y_len=3,
    )
    checkpoint = tmp_path / "model.pt"
    torch.save({"ema": model.state_dict()}, checkpoint)
    out_dir = tmp_path / "converted"
    convert_checkpoint(
        checkpoint_path=checkpoint,
        config_path=config_path,
        output_dir=out_dir,
        variant_name="toy",
    )
    loaded = LayouSynPipeline.from_pretrained(out_dir)
    assert isinstance(loaded, LayouSynPipeline)
    assert (
        json.loads((out_dir / "conversion_metadata.json").read_text())["variant_name"]
        == "toy"
    )

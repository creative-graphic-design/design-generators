import json

import torch

from basnet import BASNetConfig, BASNetModel, convert_original_checkpoint
from basnet.conversion import strip_module_prefix


def test_strip_module_prefix():
    assert strip_module_prefix({"module.weight": torch.tensor(1)})["weight"].item() == 1


def test_convert_original_checkpoint_smoke(tmp_path):
    checkpoint = tmp_path / "basnet.pth"
    output_dir = tmp_path / "converted"
    torch.save(BASNetModel(BASNetConfig()).state_dict(), checkpoint)

    report = convert_original_checkpoint(
        checkpoint=checkpoint,
        output_dir=output_dir,
        config=BASNetConfig(),
    )

    assert report["missing_keys"] == []
    assert json.loads((output_dir / "conversion_report.json").read_text())["sha256"]
    assert isinstance(
        BASNetModel.from_pretrained(output_dir, local_files_only=True), BASNetModel
    )

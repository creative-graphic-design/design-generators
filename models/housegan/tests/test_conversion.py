import pytest
import torch

from housegan import HouseGanConfig, HouseGanGenerator
from housegan.conversion import convert_original_checkpoint, sha256_file
from housegan.vendor_state_dict import convert_state_dict


def test_convert_state_dict_accepts_model_keys():
    source = HouseGanGenerator(HouseGanConfig()).state_dict()
    converted, report = convert_state_dict(source)
    assert len(converted) == report.key_count
    assert "l1.0.weight" in report.tensor_shapes


def test_convert_state_dict_rejects_unexpected_key():
    with pytest.raises(KeyError):
        convert_state_dict({"bad.weight": torch.zeros(1)})


def test_convert_original_checkpoint_writes_hf_files(tmp_path):
    checkpoint = tmp_path / "gen.pth"
    torch.save(HouseGanGenerator(HouseGanConfig()).state_dict(), checkpoint)
    report = convert_original_checkpoint(
        checkpoint=checkpoint,
        output_dir=tmp_path / "converted",
        target_set="D",
    )
    assert report["source_sha256"] == sha256_file(checkpoint)
    assert (tmp_path / "converted" / "config.json").exists()
    assert (tmp_path / "converted" / "model.safetensors").exists()
    assert (tmp_path / "converted" / "processor_config.json").exists()

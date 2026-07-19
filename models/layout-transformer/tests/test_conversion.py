import json

import torch

from layout_transformer.conversion import convert_original_checkpoint
from layout_transformer.vendor_state_dict import (
    load_original_state_dict,
    load_strict_mapped_state_dict,
)


def test_load_original_state_dict_strips_module_prefix(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"state_dict": {"module.weight": torch.ones(1)}}, checkpoint)

    state_dict = load_original_state_dict(checkpoint)

    assert sorted(state_dict) == ["weight"]


def test_load_strict_mapped_state_dict_detects_mismatch():
    model = torch.nn.Linear(1, 1)

    try:
        load_strict_mapped_state_dict(model, {"other.weight": torch.ones(1, 1)})
    except RuntimeError as exc:
        assert "State dict mismatch" in str(exc)
    else:
        raise AssertionError("expected strict mismatch")


def test_convert_original_checkpoint_writes_metadata(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    vocab = tmp_path / "object_pred_id2name.json"
    cfg = tmp_path / "config.yaml"
    output = tmp_path / "converted"
    torch.save({"state_dict": {}}, checkpoint)
    vocab.write_text(
        json.dumps(
            {"0": "[PAD]", "1": "[CLS]", "2": "[SEP]", "3": "[MASK]", "4": "person"}
        )
    )
    cfg.write_text("MODEL: {}\n")

    convert_original_checkpoint(
        checkpoint_path=checkpoint,
        cfg_path=cfg,
        vocab_path=vocab,
        output_dir=output,
        dataset_name="coco",
        strict=False,
    )

    metadata = json.loads((output / "conversion_metadata.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert metadata["dataset_name"] == "coco"
    assert metadata["strict_vendor_key_mapping"] is False
    assert config["use_vendor_modules"] is False


def test_convert_original_checkpoint_rejects_hub_upload(tmp_path):
    try:
        convert_original_checkpoint(
            checkpoint_path=tmp_path / "missing.pth",
            cfg_path=tmp_path / "missing.yaml",
            vocab_path=tmp_path / "missing.json",
            output_dir=tmp_path / "out",
            dataset_name="coco",
            push_to_hub=True,
        )
    except NotImplementedError as exc:
        assert "Hub upload" in str(exc)
    else:
        raise AssertionError("expected Hub upload rejection")

import json

import pytest
import torch

from layoutdiffusion.conversion import (
    config_from_original,
    find_ema_checkpoint,
    load_original_state_dict,
    remap_transformer_state_dict,
    validate_checkpoint_artifacts,
)


def test_remap_transformer_state_dict_removes_module_prefix() -> None:
    tensor = torch.ones(1)
    assert remap_transformer_state_dict({"module.foo": tensor}) == {"foo": tensor}


def test_validate_checkpoint_artifacts_reports_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_checkpoint_artifacts(tmp_path)


def test_config_from_original_reads_training_args(tmp_path) -> None:
    vocab = {"START": 0, "END": 1, "UNK": 2, "PAD": 3, "|": 4, "text": 5}
    vocab.update({str(i): i + 10 for i in range(128)})
    vocab["MASK"] = 138
    (tmp_path / "training_args.json").write_text(
        json.dumps({"vocab_size": 139, "diffusion_steps": 200}),
        encoding="utf-8",
    )
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tmp_path / "random_emb.torch").write_bytes(b"placeholder")
    torch.save({}, tmp_path / "ema_0.9999_1.pt")
    cfg = config_from_original(tmp_path, dataset_name="publaynet")
    assert cfg.vocab_size == 139


def test_find_ema_checkpoint_explicit_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        find_ema_checkpoint(tmp_path, "missing.pt")


def test_load_original_state_dict_supports_wrapped_state_dict(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {"module.weight": torch.ones(1)}}, path)
    assert "module.weight" in load_original_state_dict(path)


def test_load_original_state_dict_rejects_unknown_format(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(TypeError):
        load_original_state_dict(path)

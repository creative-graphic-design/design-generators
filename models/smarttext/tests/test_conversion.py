from pathlib import Path

import torch
import pytest

from smarttext import (
    SmartTextBASNet,
    SmartTextConfig,
    SmartTextPipeline,
    SmartTextScorer,
)
from smarttext.conversion import (
    convert_original_checkpoints,
    file_sha256,
    strip_module_prefix,
)


def test_strip_module_prefix():
    stripped = strip_module_prefix(
        {"module.weight": torch.tensor([1]), "bias": torch.tensor([2])}
    )

    assert set(stripped) == {"weight", "bias"}
    assert stripped["weight"].item() == 1


def test_file_sha256_and_synthetic_conversion(tmp_path):
    config = SmartTextConfig()
    scorer_path = tmp_path / "SMT.pth"
    basnet_path = tmp_path / "gdi-basnet.pth"
    torch.save(SmartTextScorer(config).state_dict(), scorer_path)
    torch.save(SmartTextBASNet(config).state_dict(), basnet_path)

    report = convert_original_checkpoints(
        smt_checkpoint=scorer_path,
        basnet_checkpoint=basnet_path,
        output_dir=tmp_path / "converted",
        config=config,
    )

    assert file_sha256(scorer_path) == report["smt_sha256"]
    loaded = SmartTextPipeline.from_pretrained(
        tmp_path / "converted", local_files_only=True
    )
    assert isinstance(loaded, SmartTextPipeline)


def test_conversion_reports_key_mismatch(tmp_path):
    scorer_path = tmp_path / "bad-smt.pth"
    basnet_path = tmp_path / "bad-basnet.pth"
    torch.save({"module.unexpected": torch.tensor([1])}, scorer_path)
    torch.save({"unexpected": torch.tensor([1])}, basnet_path)

    try:
        convert_original_checkpoints(
            smt_checkpoint=scorer_path,
            basnet_checkpoint=basnet_path,
            output_dir=tmp_path / "converted",
            config=SmartTextConfig(align_size=3, reduction_dim=4),
        )
    except RuntimeError as exc:
        assert "scorer_unexpected_keys" in str(exc)
    else:
        raise AssertionError("conversion should fail on mismatched keys")


def test_released_checkpoints_load_strictly():
    smt_checkpoint = Path(".cache/smarttext/original/SMT.pth")
    basnet_checkpoint = Path(".cache/smarttext/original/gdi-basnet.pth")
    if not smt_checkpoint.exists() or not basnet_checkpoint.exists():
        pytest.skip("SmartText released checkpoints are not downloaded")
    scorer = SmartTextScorer(SmartTextConfig())
    saliency_model = SmartTextBASNet(SmartTextConfig())

    scorer_missing, scorer_unexpected = scorer.load_state_dict(
        torch.load(smt_checkpoint, map_location="cpu"),
        strict=True,
    )
    basnet_missing, basnet_unexpected = saliency_model.load_state_dict(
        torch.load(basnet_checkpoint, map_location="cpu"),
        strict=True,
    )

    assert scorer_missing == []
    assert scorer_unexpected == []
    assert basnet_missing == []
    assert basnet_unexpected == []

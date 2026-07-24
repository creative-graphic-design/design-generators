import numpy as np
import torch

from layout_fid import LayoutFIDEvaluator, LayoutFIDModel
from layout_fid.conversion import (
    convert_layoutdm_fidnet_v3_checkpoint,
    convert_layoutflow_checkpoint,
    load_musig_statistics,
    strip_module_prefix,
    validate_state_dict_shapes,
)


def test_musig_row_semantics(tmp_path):
    path = tmp_path / "stats.pt"
    torch.save(torch.arange(12, dtype=torch.float64).reshape(3, 4), path)
    stats = load_musig_statistics(path, split="test", dataset_name="x", source="y")
    assert np.array_equal(stats.mu, np.array([0, 1, 2, 3], dtype=np.float64))
    assert stats.sigma.shape == (2, 4)


def test_state_dict_validation_and_save_load(tmp_path):
    cfg = LayoutFIDModel.config_class(
        dataset_name="publaynet",
        architecture="layoutnet",
        source="layoutflow",
        num_public_labels=5,
        num_label_embeddings=6,
        max_length=2,
        d_model=16,
        nhead=4,
        num_layers=1,
    )
    model = LayoutFIDModel(cfg)
    validate_state_dict_shapes(model.state_dict(), cfg)
    assert strip_module_prefix({"module.a": torch.ones(1)}) == {"a": torch.ones(1)}
    model.save_pretrained(tmp_path, safe_serialization=True)
    from layout_fid import LayoutFIDProcessor

    LayoutFIDProcessor(cfg).save_pretrained(tmp_path)
    loaded = LayoutFIDEvaluator.from_pretrained(tmp_path)
    features = loaded.extract_features(
        bbox=torch.zeros(2, 1, 4), labels=torch.zeros(2, 1, dtype=torch.long)
    )
    assert features.shape == (2, 16)


def test_convert_layoutflow_checkpoint_with_stats(tmp_path):
    cfg = LayoutFIDModel.config_class(
        dataset_name="publaynet",
        architecture="layoutnet",
        source="layoutflow",
        num_public_labels=5,
        num_label_embeddings=6,
        max_length=20,
    )
    checkpoint = tmp_path / "fid.pth.tar"
    torch.save(LayoutFIDModel(cfg).state_dict(), checkpoint)
    stats = tmp_path / "stats.pt"
    torch.save(torch.eye(257, 256, dtype=torch.float64), stats)
    out = tmp_path / "converted"
    converted_cfg = convert_layoutflow_checkpoint(
        checkpoint_path=checkpoint,
        output_dir=out,
        dataset_name="publaynet",
        stats_paths={"val": stats, "test": stats},
    )
    assert converted_cfg.label_id_offset == 0
    assert (out / "model.safetensors").exists()
    assert (out / "reference_stats/val.npz").exists()


def test_convert_layoutdm_checkpoint(tmp_path):
    cfg = LayoutFIDModel.config_class(
        dataset_name="publaynet",
        architecture="fidnet_v3",
        source="layoutdm",
        num_public_labels=5,
        num_label_embeddings=5,
        max_length=25,
        bbox_format_for_model="xywh",
    )
    checkpoint = tmp_path / "model_best.pth.tar"
    torch.save({"state_dict": LayoutFIDModel(cfg).state_dict()}, checkpoint)
    out = tmp_path / "converted-layoutdm"
    converted_cfg = convert_layoutdm_fidnet_v3_checkpoint(
        checkpoint_path=checkpoint,
        output_dir=out,
        dataset_name="publaynet",
        num_public_labels=5,
        max_length=25,
    )
    assert converted_cfg.source == "layoutdm"
    assert (out / "model.safetensors").exists()


def test_state_dict_shape_errors(tmp_path):
    cfg = LayoutFIDModel.config_class(
        dataset_name="publaynet",
        architecture="layoutnet",
        source="layoutflow",
        num_public_labels=5,
        num_label_embeddings=6,
        max_length=2,
        d_model=16,
        nhead=4,
        num_layers=1,
    )
    state = LayoutFIDModel(cfg).state_dict()
    del state["fc_bbox.weight"]
    try:
        validate_state_dict_shapes(state, cfg)
    except ValueError as exc:
        assert "missing expected keys" in str(exc)
    else:
        raise AssertionError("expected missing-key validation failure")

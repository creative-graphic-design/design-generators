import torch
import pytest

from cgb_dm.training import CGBDMDataModule, CGBDMTrainingModule
from PIL import Image

from cgb_dm.training.parity import (
    CGBDMStepTraceAdapter,
    build_reference_encoded_dataset,
    capture_source_order,
    load_source_order_manifest,
    write_source_order_manifest,
)
from cgb_dm.training.seed import apply_seed_mode
from traingen.lightning.cli import main


def test_training_step_records_required_trace():
    module = CGBDMTrainingModule(
        config={
            "max_seq_length": 2,
            "image_size": (32, 32),
            "dim_model": 16,
            "n_head": 2,
            "feature_dim": 32,
            "num_layers": 1,
            "num_train_timesteps": 10,
            "ddim_num_steps": 1,
        },
        optimizer=lambda params: torch.optim.Adam(params, lr=1.0e-4),
    )
    batch = {
        "pixel_values": torch.zeros(1, 4, 32, 32),
        "layout": torch.zeros(1, 2, 8),
        "saliency_box": torch.zeros(1, 1, 4),
    }
    batch["layout"][:, :, 0] = 1

    loss = module.training_step(batch, 0)

    assert loss.ndim == 0
    assert CGBDMStepTraceAdapter.trace_points[-1] == "loss"
    assert set(CGBDMStepTraceAdapter.trace_points) <= set(module.latest_step_trace)


def test_datamodule_synthetic_batch():
    datamodule = CGBDMDataModule(
        config={
            "max_seq_length": 2,
            "image_size": (32, 32),
            "dim_model": 16,
            "n_head": 2,
            "feature_dim": 32,
            "num_layers": 1,
        },
        source="synthetic",
        batch_size=2,
    )
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["pixel_values"].shape == (2, 4, 32, 32)
    assert batch["layout"].shape == (2, 2, 8)


def test_datamodule_original_requires_root_and_optimizer_injection():
    config: dict[str, object] = {
        "max_seq_length": 2,
        "image_size": (32, 32),
        "dim_model": 16,
        "n_head": 2,
        "feature_dim": 32,
        "num_layers": 1,
    }
    datamodule = CGBDMDataModule(config=config, source="original_zip")
    try:
        datamodule.setup()
    except ValueError as exc:
        assert "data_root is required" in str(exc)
    else:
        raise AssertionError("expected data_root error")

    module = CGBDMTrainingModule(
        config=config,
        optimizer=lambda params: torch.optim.Adam(params, lr=1.0e-4),
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1
        ),
    )
    optimizers = module.configure_optimizers()
    assert isinstance(optimizers, dict)
    assert "optimizer" in optimizers
    assert "lr_scheduler" in optimizers
    assert apply_seed_mode("default")["mode"] == "default"
    assert apply_seed_mode("deterministic", seed=2)["deterministic_algorithms"] is True


def test_training_helpers_cover_tuple_adapter_and_plain_optimizer():
    config: dict[str, object] = {
        "max_seq_length": 2,
        "image_size": (32, 32),
        "dim_model": 16,
        "n_head": 2,
        "feature_dim": 32,
        "num_layers": 1,
    }
    module = CGBDMTrainingModule(
        config=config,
    )
    assert isinstance(module.configure_optimizers(), torch.optim.Adam)

    image = torch.zeros(1, 4, 32, 32)
    layout = torch.zeros(1, 2, 8)
    saliency_box = torch.zeros(1, 1, 4)
    adapter = CGBDMStepTraceAdapter()
    batch = adapter.comparable_batch((image, layout, saliency_box))
    assert batch["pixel_values"] is image
    assert batch["layout"] is layout
    assert batch["saliency_box"] is saliency_box


def test_shared_lightning_cli_runs_smoke_config(tmp_path):
    main(
        [
            "fit",
            "--config",
            "models/cgb-dm/configs/training/smoke.yaml",
            "--trainer.accelerator",
            "cpu",
            "--trainer.devices",
            "1",
            "--trainer.default_root_dir",
            str(tmp_path),
        ]
    )


def test_shared_lightning_cli_help_is_available():
    with pytest.raises(SystemExit):
        main(["--help"])


def test_vendor_order_manifest_helpers(tmp_path):
    root = tmp_path / "split"
    for rel in [
        "train/inpaint",
        "train/saliency",
        "train/saliency_sub",
        "csv",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20)).save(root / "train/inpaint/sample.png")
    Image.new("L", (20, 20)).save(root / "train/saliency/sample.png")
    Image.new("L", (20, 20)).save(root / "train/saliency_sub/sample.png")
    (root / "csv/train.csv").write_text(
        'poster_path,box_elem,cls_elem\nsample.png,"[0, 0, 10, 10]",1\n',
        encoding="utf-8",
    )
    (root / "csv/train_sal.csv").write_text(
        'poster_path,box_elem\nsample.png,"[0, 0, 20, 20]"\n',
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"

    assert capture_source_order(root) == ["sample.png"]
    assert (
        write_source_order_manifest(
            data_root=root,
            output=manifest,
            dataset="pku_posterlayout",
        )
        == manifest
    )
    assert load_source_order_manifest(manifest) == ["sample.png"]
    dataset = build_reference_encoded_dataset(root, manifest=manifest)
    assert dataset[0]["layout"].shape == (16, 8)


def test_original_zip_datamodule_uses_reference_encoding_and_manifest(tmp_path):
    root = tmp_path / "split"
    for rel in [
        "train/inpaint",
        "train/saliency",
        "train/saliency_sub",
        "val/inpaint",
        "val/saliency",
        "val/saliency_sub",
        "csv",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        for name in ["b.png", "a.png"]:
            Image.new("RGB", (20, 20)).save(root / split / "inpaint" / name)
            Image.new("L", (20, 20)).save(root / split / "saliency" / name)
            Image.new("L", (20, 20)).save(root / split / "saliency_sub" / name)
        (root / "csv" / f"{split}.csv").write_text(
            "poster_path,box_elem,cls_elem\n"
            'a.png,"[0, 0, 10, 10]",1\n'
            'b.png,"[0, 0, 12, 12]",3\n',
            encoding="utf-8",
        )
        (root / "csv" / f"{split}_sal.csv").write_text(
            'poster_path,box_elem\na.png,"[0, 0, 20, 20]"\nb.png,"[0, 0, 20, 20]"\n',
            encoding="utf-8",
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"names": ["b.png", "a.png"]}', encoding="utf-8")

    datamodule = CGBDMDataModule(
        config={
            "dataset_name": "pku_posterlayout",
            "max_seq_length": 16,
            "image_size": (32, 32),
            "dim_model": 16,
            "n_head": 2,
            "feature_dim": 32,
            "num_layers": 1,
        },
        source="original_zip",
        data_root=str(root),
        source_order_manifest=str(manifest),
        batch_size=1,
    )
    datamodule.setup()

    assert datamodule.train_dataset.names == ["b.png", "a.png"]
    row = datamodule.train_dataset[0]
    assert row["layout"][:, :4].argmax(dim=-1).tolist()[:2] == [3, 0]

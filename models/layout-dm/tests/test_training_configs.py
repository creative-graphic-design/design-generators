from pathlib import Path


CONFIG_DIR = Path("models/layout-dm/configs/training")


def test_training_configs_use_lightning_cli_shape_without_hydra_keys() -> None:
    for path in CONFIG_DIR.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "_target_" not in text
        assert "hydra." not in text
        assert "defaults:" not in text
        assert "class_path:" in text
        assert "init_args:" in text
        assert "layout_dm.training.lightning_module." not in text
        assert "layout_dm.training.datamodule." not in text
        assert "class_path: layout_dm.training.LayoutDMTrainingModule" in text
        assert "class_path: layout_dm.training.LayoutDMDataModule" in text


def test_s5_training_configs_pin_layoutdm_experiment_settings() -> None:
    for dataset in ("rico25", "publaynet"):
        text = (CONFIG_DIR / f"layoutdm_{dataset}.yaml").read_text(encoding="utf-8")
        assert "  max_epochs: 50" in text
        assert "  check_val_every_n_epoch: 1" in text
        assert "  gradient_clip_val: 1.0" in text
        assert "    learning_rate: 0.0005" in text
        assert "    weight_decay: 0.1" in text
        assert "    betas: [0.9, 0.98]" in text
        assert "    scheduler: reduce_on_plateau" in text
        assert "    scheduler_factor: 0.5" in text
        assert "    scheduler_patience: 2" in text
        assert "    scheduler_threshold: 0.01" in text
        assert "    batch_size: 64" in text
        assert "bbox_quantization: kmeans" in text
        assert "q_type: constrained" in text
        assert (
            f"cluster_centers_path: .cache/layout-dm/original/download/clustering_weights/{dataset}_max25_kmeans_train_clusters.pkl"
            in text
        )

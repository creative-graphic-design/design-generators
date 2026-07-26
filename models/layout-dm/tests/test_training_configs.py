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
        assert "bbox_quantization: kmeans" in text
        assert "q_type: constrained" in text
        assert (
            f"cluster_centers_path: .cache/layout-dm/original/download/clustering_weights/{dataset}_max25_kmeans_train_clusters.pkl"
            in text
        )

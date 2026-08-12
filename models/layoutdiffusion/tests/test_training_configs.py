import re
from pathlib import Path

CONFIG_DIR = Path("models/layoutdiffusion/configs/training")
REFERENCE_GENERATOR = Path(
    "models/layoutdiffusion/tests/vendor_parity/layoutdiffusion_training_reference.py"
)
ORIGINAL_AUXILIARY_LOSS_WEIGHT = 1e-3


def _auxiliary_loss_weight_from_config(text: str) -> float:
    match = re.search(
        r"^\s+auxiliary_loss_weight:\s+([0-9.eE+-]+)\s*$", text, re.MULTILINE
    )
    assert match is not None
    return float(match.group(1))


def _auxiliary_loss_weight_from_reference_default() -> float:
    text = REFERENCE_GENERATOR.read_text(encoding="utf-8")
    match = re.search(r"auxiliary_loss_weight: float = ([0-9.eE+-]+),", text)
    assert match is not None
    return float(match.group(1))


def test_training_configs_use_lightning_cli_shape_without_hydra_keys() -> None:
    for path in CONFIG_DIR.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "_target_" not in text
        assert "hydra." not in text
        assert "defaults:" not in text
        assert "class_path:" in text
        assert "init_args:" in text
        assert (
            "class_path: layoutdiffusion.training.lightning_module.LayoutDiffusionTrainingModule"
            in text
        )
        assert (
            "class_path: layoutdiffusion.training.datamodule.LayoutDiffusionDataModule"
            in text
        )
        assert (
            "class_path: layoutdiffusion.training.LayoutDiffusionTrainingModule"
            not in text
        )
        assert (
            "class_path: layoutdiffusion.training.LayoutDiffusionDataModule" not in text
        )


def test_s5_training_configs_pin_layoutdiffusion_settings() -> None:
    expected = {
        "rico25": ("0.00004", "175000", "159", "RICO_ltrb_lex"),
        "publaynet": ("0.00005", "400000", "139", "PublayNet_ltrb_lex"),
    }
    for dataset, (lr, lr_anneal_steps, vocab_size, stream_name) in expected.items():
        text = (CONFIG_DIR / f"layoutdiffusion_{dataset}.yaml").read_text(
            encoding="utf-8"
        )
        assert "        seq_length: 121" in text
        assert "        diffusion_steps: 200" in text
        assert "        noise_schedule: gaussian_refine_pow2.5" in text
        assert "        num_channels: 128" in text
        assert "        dropout: 0.1" in text
        assert "        training_mode: discrete1" in text
        assert f"        vocab_size: {vocab_size}" in text
        assert (
            f"    vocab_file: .cache/layoutdiffusion/original-data/{stream_name}/vocab.json"
            in text
        )
        assert f"    learning_rate: {lr}" in text
        assert "    weight_decay: 0.0" in text
        assert (
            _auxiliary_loss_weight_from_config(text) == ORIGINAL_AUXILIARY_LOSS_WEIGHT
        )
        assert "    time_sampler: uniform" in text
        assert "    scheduler: linear_anneal" in text
        assert f"    lr_anneal_steps: {lr_anneal_steps}" in text
        assert "    ema_rate: 0.9999" in text
        assert "    batch_size: 64" in text
        assert "    dataset_source: processed" in text
        assert "    preconsume_train_batches: 1" in text
        assert "    processed_stream_rng_warmup: true" in text
        assert "    train_transforms: [LexicographicOrder]" in text


def test_processed_s5_training_configs_pin_preconsume() -> None:
    for path in CONFIG_DIR.glob("layoutdiffusion_*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "    dataset_source: processed" in text
        assert "    preconsume_train_batches: 1" in text
        assert "    processed_stream_rng_warmup: true" in text
        assert "    auxiliary_loss_weight: 0.001" in text


def test_s5_training_configs_duplicate_same_vocab_file_for_model_and_data() -> None:
    for path in CONFIG_DIR.glob("layoutdiffusion_*.yaml"):
        text = path.read_text(encoding="utf-8")
        matches = re.findall(r"^    vocab_file: (.+)$", text, flags=re.MULTILINE)
        assert len(matches) == 2, path
        assert matches[0] == matches[1]


def test_all_training_configs_match_original_auxiliary_loss_default() -> None:
    reference_default = _auxiliary_loss_weight_from_reference_default()
    assert reference_default == ORIGINAL_AUXILIARY_LOSS_WEIGHT
    for path in CONFIG_DIR.glob("*.yaml"):
        assert (
            _auxiliary_loss_weight_from_config(path.read_text(encoding="utf-8"))
            == reference_default
        )

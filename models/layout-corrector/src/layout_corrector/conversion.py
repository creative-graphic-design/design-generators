"""Conversion helpers for original Layout-Corrector checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from layout_dm.pipeline import LayoutDMPipeline
from laygen.common.labels import normalize_dataset_name

from .configuration_layout_corrector import LayoutCorrectorConfig
from .corrector import LayoutCorrectorModel


def remap_corrector_key(key: str) -> str:
    """Map an original checkpoint key to the converted module key.

    Args:
        key: Original state-dict key.

    Returns:
        Key accepted by `AggregatedCategoricalTransformer`.

    Raises:
        ValueError: If the key does not use an expected original prefix.
    """
    for prefix in ("model.module.model.", "model.module."):
        if key.startswith(prefix):
            return key.removeprefix(prefix)
    raise ValueError(f"Unexpected corrector checkpoint key: {key}")


def split_original_corrector_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strip original wrapper prefixes from a corrector state dict.

    Args:
        state_dict: Original checkpoint state dictionary.

    Returns:
        Converted state dictionary.
    """
    return {remap_corrector_key(key): value for key, value in state_dict.items()}


def load_original_corrector_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load and remap an original corrector checkpoint.

    Args:
        path: Path to `best_model.pt`.

    Returns:
        Converted state dictionary.
    """
    state = torch.load(path, map_location="cpu")
    raw = state.get("state_dict", state)
    return split_original_corrector_state_dict(raw)


def corrector_config_from_original(
    *,
    dataset: str,
    config_path: str | Path,
    state_dict: dict[str, torch.Tensor],
    layout_dm: LayoutDMPipeline,
) -> LayoutCorrectorConfig:
    """Build a Layout-Corrector config from original files.

    Args:
        dataset: Dataset key or alias.
        config_path: Original `config.yaml` path.
        state_dict: Converted corrector state dictionary.
        layout_dm: Nested LayoutDM pipeline used for compatibility checks.

    Returns:
        Converted Layout-Corrector config.
    """
    with Path(config_path).open() as f:
        original_config = yaml.safe_load(f)
    data_cfg = original_config["data"]
    model_cfg = original_config["model"]
    layer_cfg = original_config["backbone"]["encoder_layer"]
    hidden_size = int(state_dict["cat_emb.weight"].shape[1])
    intermediate_size = int(state_dict["backbone.layers.0.linear1.weight"].shape[0])
    normalized_dataset = (
        str(normalize_dataset_name(dataset))
        if dataset not in {"crello", "crello-bbox"}
        else "crello-bbox"
    )
    id2label = (
        layout_dm.tokenizer.config.id2label
        if normalized_dataset == "crello-bbox"
        else None
    )
    return LayoutCorrectorConfig(
        dataset_name=normalized_dataset,
        id2label=id2label,
        vocab_size=int(state_dict["cat_emb.weight"].shape[0]),
        max_seq_length=int(original_config["dataset"].get("max_seq_length", 25)),
        num_attributes_per_element=len(
            data_cfg.get("var_order", "c-x-y-w-h").split("-")
        ),
        hidden_size=hidden_size,
        num_attention_heads=int(layer_cfg.get("nhead", 8)),
        num_hidden_layers=int(original_config["backbone"].get("num_layers", 4)),
        intermediate_size=intermediate_size,
        dropout=float(layer_cfg.get("dropout", 0.0)),
        timestep_type=layer_cfg.get("timestep_type", "adalayernorm"),
        num_timesteps=int(model_cfg.get("num_timesteps", 100)),
        recon_type=model_cfg.get("recon_type", "x_t-1"),
        target=model_cfg.get("target", "recon_acc"),
        attr_loss_weights=tuple(float(v) for v in model_cfg["attr_loss_weights"]),
        use_padding_as_vocab=bool(model_cfg.get("use_padding_as_vocab", True)),
        pos_emb=model_cfg.get("pos_emb", "none"),
        transformer_type=model_cfg.get("transformer_type", "aggregated"),
    )


def build_corrector_from_original(
    *,
    dataset: str,
    checkpoint_dir: str | Path,
    layout_dm: LayoutDMPipeline,
) -> LayoutCorrectorModel:
    """Build a `LayoutCorrectorModel` from an original checkpoint directory.

    Args:
        dataset: Dataset key or alias.
        checkpoint_dir: Directory containing `best_model.pt` and `config.yaml`.
        layout_dm: Nested LayoutDM pipeline paired with the corrector.

    Returns:
        Loaded and eval-mode corrector model.
    """
    checkpoint_dir = Path(checkpoint_dir)
    state_dict = load_original_corrector_state_dict(checkpoint_dir / "best_model.pt")
    config = corrector_config_from_original(
        dataset=dataset,
        config_path=checkpoint_dir / "config.yaml",
        state_dict=state_dict,
        layout_dm=layout_dm,
    )
    validate_layout_dm_compatibility(layout_dm=layout_dm, corrector_config=config)
    corrector = LayoutCorrectorModel(**config.config)
    corrector.model.load_state_dict(state_dict, strict=True)
    corrector.eval()
    return corrector


def discover_seed_dirs(job_dir: str | Path) -> list[Path]:
    """Discover corrector seed directories under an original job directory.

    Args:
        job_dir: Seed directory or parent directory containing seed subdirectories.

    Returns:
        Sorted list of directories containing `config.yaml`.
    """
    path = Path(job_dir)
    if (path / "config.yaml").is_file():
        return [path]
    return sorted(
        child for child in path.iterdir() if (child / "config.yaml").is_file()
    )


def validate_layout_dm_compatibility(
    *,
    layout_dm: LayoutDMPipeline,
    corrector_config: LayoutCorrectorConfig,
) -> None:
    """Validate that a corrector config matches its nested LayoutDM pipeline.

    Args:
        layout_dm: Nested LayoutDM pipeline.
        corrector_config: Corrector config to compare.

    Raises:
        ValueError: If a shared tokenizer or scheduler field differs.
    """
    tokenizer_config = layout_dm.tokenizer.config
    checks: dict[str, Any] = {
        "vocab_size": tokenizer_config.vocab_size,
        "max_seq_length": tokenizer_config.max_seq_length,
        "num_attributes_per_element": tokenizer_config.num_attributes_per_element,
        "num_timesteps": layout_dm.scheduler.config.num_timesteps,
    }
    for key, value in checks.items():
        if getattr(corrector_config, key) != value:
            raise ValueError(
                f"LayoutDM/{key} mismatch: corrector={getattr(corrector_config, key)} "
                f"layout_dm={value}"
            )

from __future__ import annotations

from pathlib import Path
from typing import Any

from layout_dm.pipeline import LayoutDMPipeline

from .configuration_layout_corrector import LayoutCorrectorConfig


def remap_corrector_key(key: str) -> str:
    prefix = "model.module.model."
    if not key.startswith(prefix):
        raise ValueError(f"Unexpected corrector checkpoint key: {key}")
    return key.removeprefix(prefix)


def discover_seed_dirs(job_dir: str | Path) -> list[Path]:
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

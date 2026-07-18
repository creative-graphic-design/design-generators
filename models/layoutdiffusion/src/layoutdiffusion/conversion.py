"""Conversion helpers for original LayoutDiffusion checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import torch

from .configuration_layoutdiffusion import LayoutDiffusionConfig

REQUIRED_ARTIFACTS: Final[tuple[str, ...]] = (
    "training_args.json",
    "vocab.json",
    "random_emb.torch",
)


def find_ema_checkpoint(
    checkpoint_dir: str | Path, checkpoint_name: str | None = None
) -> Path:
    """Find an EMA checkpoint file in an original checkpoint directory.

    Args:
        checkpoint_dir: Original checkpoint directory.
        checkpoint_name: Optional explicit checkpoint filename.

    Returns:
        Path to the selected checkpoint.

    Raises:
        FileNotFoundError: If no checkpoint exists.
    """
    root = Path(checkpoint_dir)
    if checkpoint_name is not None:
        path = root / checkpoint_name
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    matches = sorted(root.glob("ema_0.9999_*.pt"))
    if not matches:
        raise FileNotFoundError(f"No ema_0.9999_*.pt checkpoint under {root}")
    return matches[-1]


def validate_checkpoint_artifacts(checkpoint_dir: str | Path) -> dict[str, Path]:
    """Validate required original checkpoint artifacts.

    Args:
        checkpoint_dir: Original checkpoint directory.

    Returns:
        Mapping from artifact name to path.

    Raises:
        FileNotFoundError: If a required artifact is missing.
    """
    root = Path(checkpoint_dir)
    artifacts = {name: root / name for name in REQUIRED_ARTIFACTS}
    missing = [str(path) for path in artifacts.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    artifacts["checkpoint"] = find_ema_checkpoint(root)
    return artifacts


def config_from_original(
    checkpoint_dir: str | Path,
    *,
    dataset_name: str,
) -> LayoutDiffusionConfig:
    """Build ``LayoutDiffusionConfig`` from original JSON files.

    Args:
        checkpoint_dir: Original checkpoint directory.
        dataset_name: Canonical dataset name.

    Returns:
        Converted configuration.
    """
    artifacts = validate_checkpoint_artifacts(checkpoint_dir)
    args = json.loads(artifacts["training_args.json"].read_text(encoding="utf-8"))
    vocab = json.loads(artifacts["vocab.json"].read_text(encoding="utf-8"))
    vocab = {str(k): int(v) for k, v in vocab.items()}
    return LayoutDiffusionConfig(
        dataset_name=dataset_name,
        vocab=vocab,
        vocab_size=int(args.get("vocab_size", len(vocab))),
        seq_length=int(args.get("seq_length", 121)),
        diffusion_steps=int(args.get("diffusion_steps", 200)),
        noise_schedule=str(args.get("noise_schedule", "gaussian_refine_pow2.5")),
        num_channels=int(args.get("num_channels", 128)),
        training_mode=str(args.get("training_mode", "discrete")),
    )


def remap_transformer_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap original EMA keys to the new transformer module.

    Args:
        state_dict: Original checkpoint state dict.

    Returns:
        Remapped state dict with ``module.`` prefixes removed.
    """
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_original_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Load an original PyTorch checkpoint on CPU."""
    raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        return raw
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

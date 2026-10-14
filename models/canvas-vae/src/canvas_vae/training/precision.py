"""Precision settings required for source-matched Crello training."""

from __future__ import annotations

import torch


def disable_torch_tf32() -> None:
    """Disable TF32 for PyTorch matmul and cuDNN and verify the settings."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("PyTorch CUDA matmul TF32 must be disabled")

    if torch.backends.cudnn.allow_tf32:
        raise RuntimeError("PyTorch cuDNN TF32 must be disabled")

    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("PyTorch float32 matmul precision must be highest")


__all__ = ["disable_torch_tf32"]

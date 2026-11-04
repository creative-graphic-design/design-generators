"""Model-card helpers for LayoutVAE."""

from __future__ import annotations

from pathlib import Path


def write_layoutvae_model_card(output_dir: str | Path) -> Path:
    """Write a minimal Hub README for converted LayoutVAE artifacts.

    Args:
        output_dir: Target checkpoint directory.

    Returns:
        Path to the written README.

    Examples:
        >>> import tempfile
        >>> path = write_layoutvae_model_card(tempfile.mkdtemp())
        >>> path.name
        'README.md'
    """
    path = Path(output_dir) / "README.md"
    text = """---
license: mit
library_name: transformers
pipeline_tag: other
tags:
  - layout-generation
datasets:
  - creative-graphic-design/PubLayNet
---

# Model Card for LayoutVAE PubLayNet

LayoutVAE generates document layouts from PubLayNet label sets.
"""
    path.write_text(text, encoding="utf-8")
    return path

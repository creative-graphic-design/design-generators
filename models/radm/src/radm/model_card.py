"""Model-card helper for local RADM converted artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Final

RADM_BIBTEX: Final[str] = r"""
@inproceedings{zhang2023radm,
  title={Relation-Aware Diffusion Model for Controllable Poster Layout Generation},
  author={Zhang, Yiheng and others},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  year={2023}
}
"""


def write_local_model_card(
    output_dir: str | Path, *, dataset_name: str = "cgl"
) -> Path:
    """Write a short local model-card placeholder for converted artifacts.

    Args:
        output_dir: Directory receiving ``README.md``.
        dataset_name: Dataset variant recorded in the card.

    Returns:
        Path to the written README.
    """
    path = Path(output_dir) / "README.md"
    path.write_text(
        "\n".join(
            [
                "---",
                'library_name: "diffusers"',
                'pipeline_tag: "other"',
                "tags:",
                '  - "radm"',
                '  - "layout-generation"',
                "datasets:",
                '  - "creative-graphic-design/CGL-Dataset"',
                "---",
                "",
                f"# RADM local converted artifact ({dataset_name})",
                "",
                "Converted weights are local-only; the RADM README publishes dataset/test-feature assets but no checkpoint.",
            ]
        ),
        encoding="utf-8",
    )
    return path

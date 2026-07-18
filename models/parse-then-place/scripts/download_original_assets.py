"""Download instructions helper for original Parse-Then-Place assets."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Write the original asset download command to a metadata file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ORIGINAL_ASSETS.md").write_text(
        "\n".join(
            [
                "# Original Parse-Then-Place assets",
                "",
                "Download checkpoints and datasets from the vendor-published",
                "Hugging Face dataset repository:",
                "",
                "```bash",
                "git lfs install",
                "git clone https://huggingface.co/datasets/KyleLin/Parse-Then-Place",
                "```",
                "",
                "This implementation does not push or vendor those artifacts.",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

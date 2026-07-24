"""Save a PosterO prompt configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from postero import PosterOAgent, PosterOConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/postero-prompt-config"),
        help="Directory that will receive postero_config.json.",
    )
    parser.add_argument(
        "--dataset-name",
        default="pku_posterlayout",
        help="Poster dataset name stored in the config.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of exemplar candidates considered by the prompt builder.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the config save utility."""
    args = parse_args()
    agent = PosterOAgent(
        config=PosterOConfig(
            dataset_name=args.dataset_name, sample_size=args.sample_size
        )
    )
    agent.save_pretrained(args.output_dir)
    print(args.output_dir / "postero_config.json")


if __name__ == "__main__":
    main()

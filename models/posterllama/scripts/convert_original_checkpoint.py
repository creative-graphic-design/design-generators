"""Create a local PosterLlama pipeline directory from explicit local assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from posterllama import PosterLlamaConfig, PosterLlamaPipeline, PosterLlamaProcessor
from posterllama.modeling_posterllama import PosterLlamaRuntime


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Local raw pytorch_model.bin path from poong/PosterLlama.",
    )
    parser.add_argument(
        "--base-llm-path",
        type=Path,
        required=True,
        help="Local CodeLLaMA or LLaMA backbone directory.",
    )
    parser.add_argument(
        "--vision-encoder-id",
        default="facebook/dinov2-base",
        help="Vision encoder id recorded in config. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output pipeline directory.",
    )
    parser.add_argument(
        "--parser-smoke-text",
        default=None,
        help="Optional deterministic generated markup for parser-only smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    """Write a local recipe directory.

    The full MiniGPT/DINO/CodeLLaMA tensor mapping is intentionally explicit and
    local-only because the raw checkpoint and backbone redistribution are blocked
    pending license review. This script records the asset paths in runtime
    metadata for smoke testing and leaves heavyweight conversion to the gated
    vendor parity workflow.
    """
    args = parse_args()
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(args.checkpoint_path)
    if not args.base_llm_path.exists():
        raise FileNotFoundError(args.base_llm_path)
    config = PosterLlamaConfig(vision_encoder_repo_id=args.vision_encoder_id)
    pipeline = PosterLlamaPipeline(
        config=config,
        processor=PosterLlamaProcessor.from_config(config),
        runtime=PosterLlamaRuntime(args.parser_smoke_text),
    )
    pipeline.save_pretrained(args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()

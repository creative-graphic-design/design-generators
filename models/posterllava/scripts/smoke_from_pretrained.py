"""Run a local PosterLLaVA from-pretrained smoke check."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer  # ty: ignore[possibly-missing-import]

from posterllava import PosterLlavaConfig, PosterLlavaPipeline, PosterLlavaProcessor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="posterllava/posterllava_v0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--num-elements", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    """Load components and run one image-conditioned generation."""
    args = parse_args()
    config = PosterLlavaConfig(dataset_name="ad_banner", checkpoint_id=args.model_id)
    processor = PosterLlavaProcessor.from_config(
        dataset_name=config.dataset_name,
        id2label=config.id2label,
        prompt_template=config.prompt_template,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )
    pipe = PosterLlavaPipeline.from_pretrained(
        args.model_id,
        config=config,
        components={
            "processor": processor,
            "model": model,
            "tokenizer": tokenizer,
            "image_processor": image_processor,
        },
        local_files_only=args.local_files_only,
    ).to(args.device)
    image = Image.open(args.image).convert("RGB")
    output = pipe(
        images=image,
        num_elements=args.num_elements,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        return_intermediates=True,
    )
    print(output)


if __name__ == "__main__":
    main()

"""Convert an original LayoutDiffusion checkpoint into a Diffusers pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from layoutdiffusion import (
    LayoutDiffusionPipeline,
    LayoutDiffusionScheduler,
    LayoutDiffusionTokenizer,
    LayoutDiffusionTransformer,
)
from layoutdiffusion.conversion import (
    config_from_original,
    find_ema_checkpoint,
    IGNORED_CHECKPOINT_KEYS,
    load_original_state_dict,
    remap_transformer_state_dict,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["rico25", "publaynet"])
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Save model weights with safetensors when supported.",
    )
    return parser.parse_args()


def main() -> None:
    """Run checkpoint conversion."""
    args = parse_args()
    cfg = config_from_original(args.checkpoint_dir, dataset_name=args.dataset)
    tokenizer = LayoutDiffusionTokenizer(cfg)
    transformer = LayoutDiffusionTransformer(
        vocab_size=cfg.vocab_size,
        num_channels=cfg.num_channels,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        dropout=cfg.dropout,
    )
    checkpoint = find_ema_checkpoint(args.checkpoint_dir, args.checkpoint_name)
    original_state = load_original_state_dict(checkpoint)
    remapped_state = remap_transformer_state_dict(original_state)
    transformer.load_state_dict(remapped_state, strict=True)
    scheduler = LayoutDiffusionScheduler(
        num_train_timesteps=cfg.diffusion_steps,
        vocab_size=cfg.vocab_size,
        mask_token_id=cfg.mask_token_id,
        type_classes=cfg.type_classes,
        noise_schedule=cfg.noise_schedule,
        pow_num=cfg.pow_num,
        mul_num=cfg.mul_num,
        type_start_step=cfg.type_start_step,
    )
    pipe = LayoutDiffusionPipeline(transformer, scheduler, tokenizer)
    pipe.save_pretrained(args.output_dir, safe_serialization=args.safe_serialization)
    print(f"saved={args.output_dir}")
    print(f"checkpoint={checkpoint}")
    print(f"original_keys={len(original_state)}")
    print(f"loaded_keys={len(remapped_state)}")
    print(f"ignored_keys={sorted(IGNORED_CHECKPOINT_KEYS)}")


if __name__ == "__main__":
    main()

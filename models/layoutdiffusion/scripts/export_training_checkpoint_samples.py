"""Export package-trained LayoutDiffusion checkpoint samples for vendor metrics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import torch
from jaxtyping import Shaped

from layoutdiffusion import (
    LayoutDiffusionConfig,
    LayoutDiffusionPipeline,
    LayoutDiffusionScheduler,
    LayoutDiffusionTokenizer,
    LayoutDiffusionTransformer,
)
from layoutdiffusion.sampling import LayoutDiffusionSamplingConfig
from layoutdiffusion.training.lightning_module import EMA_CHECKPOINT_KEY

DATASET_SAMPLE_COUNTS = {
    "rico25": 3728,
    "publaynet": 10998,
}
LEGACY_EMA_KEYS = (EMA_CHECKPOINT_KEY, "ema")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=sorted(DATASET_SAMPLE_COUNTS), required=True
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional LayoutDiffusionConfig JSON file for non-default model shapes.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weights",
        choices=["ema", "raw"],
        default="ema",
        help="Use persisted EMA weights by default; raw weights require this explicit opt-in.",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sampling-name",
        choices=["vendor_gumbel", "argmax"],
        default="vendor_gumbel",
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run sample export."""
    args = parse_args()
    export_training_checkpoint_samples(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        output_path=args.output,
        config_path=args.config,
        weights=cast(Literal["ema", "raw"], args.weights),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
        sampling_name=args.sampling_name,
        num_inference_steps=args.num_inference_steps,
    )


def export_training_checkpoint_samples(
    *,
    checkpoint_path: Path,
    dataset: Literal["rico25", "publaynet"] | str,
    output_path: Path,
    config_path: Path | None = None,
    weights: Literal["ema", "raw"] = "ema",
    num_samples: int | None = None,
    batch_size: int = 64,
    seed: int = 101,
    device_name: str = "cuda",
    sampling_name: str = "vendor_gumbel",
    num_inference_steps: int | None = None,
) -> Path:
    """Sample a package training checkpoint and write vendor JSONL tokens."""
    config = load_export_config(dataset=dataset, config_path=config_path)
    tokenizer = LayoutDiffusionTokenizer(config)
    transformer = LayoutDiffusionTransformer(
        vocab_size=config.vocab_size,
        num_channels=config.num_channels,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        dropout=config.dropout,
        max_position_embeddings=config.max_position_embeddings,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")
    state_dict = select_transformer_state_dict(checkpoint, weights=weights)
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint weights do not match LayoutDiffusionTransformer: "
            f"missing={missing}, unexpected={unexpected}"
        )
    device = resolve_device(device_name)
    torch.nn.Module.to(transformer, device)
    scheduler = LayoutDiffusionScheduler.from_layout_config(config)
    pipe = LayoutDiffusionPipeline(transformer, scheduler, tokenizer).to(device)
    sampling = LayoutDiffusionSamplingConfig(
        name=sampling_name,
        num_inference_steps=num_inference_steps,
    )
    sample_count = num_samples or DATASET_SAMPLE_COUNTS[str(dataset)]
    generator = torch.Generator(device=device).manual_seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        while written < sample_count:
            current_batch = min(batch_size, sample_count - written)
            output = pipe(
                batch_size=current_batch,
                generator=generator,
                condition_type="unconditional",
                sampling=sampling,
                return_intermediates=True,
            )
            sequences = output.sequences
            if sequences is None:
                raise RuntimeError("Pipeline did not return token sequences")
            for line in format_vendor_json_lines(
                tokenizer.token_ids_to_text(sequences)
            ):
                handle.write(line)
            written += current_batch
    print(f"wrote={output_path}")
    print(f"samples={written}")
    print(f"weights={weights}")
    return output_path


def load_export_config(
    *, dataset: Literal["rico25", "publaynet"] | str, config_path: Path | None
) -> LayoutDiffusionConfig:
    """Load an explicit LayoutDiffusionConfig or build the dataset default."""
    if config_path is None:
        return LayoutDiffusionConfig(dataset_name=dataset)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a JSON object: {config_path}")
    data.setdefault("dataset_name", dataset)
    return LayoutDiffusionConfig(**data)


def select_transformer_state_dict(
    checkpoint: Mapping[str, object], *, weights: Literal["ema", "raw"]
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    """Select EMA or raw transformer weights from a Lightning checkpoint."""
    if weights == "ema":
        for key in LEGACY_EMA_KEYS:
            maybe_state = checkpoint.get(key)
            if isinstance(maybe_state, Mapping):
                return tensor_state_dict(cast(Mapping[object, object], maybe_state))
        raise RuntimeError(
            "Checkpoint does not contain EMA weights. Re-run training with "
            f"{EMA_CHECKPOINT_KEY} support or pass --weights raw to export raw weights."
        )
    raw_state = checkpoint.get("state_dict", checkpoint.get("model_state", checkpoint))
    if not isinstance(raw_state, Mapping):
        raise TypeError("Raw checkpoint state must be a mapping")
    state_dict = tensor_state_dict(cast(Mapping[object, object], raw_state))
    model_state = strip_prefix(state_dict, "model.")
    return model_state or state_dict


def tensor_state_dict(
    state: Mapping[object, object],
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    """Return only tensor entries from a state mapping."""
    return {
        str(key): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def strip_prefix(
    state_dict: Mapping[str, Shaped[torch.Tensor, "..."]], prefix: str
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    """Strip a module prefix from every matching state-dict key."""
    return {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def format_vendor_json_lines(lines: list[str]) -> list[str]:
    """Format token strings like the vendor text_sample.py JSON output."""
    return [json.dumps((line,)) + "\n" for line in lines]


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested device with a CPU fallback for unavailable CUDA."""
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


if __name__ == "__main__":
    main()

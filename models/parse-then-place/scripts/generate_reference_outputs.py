"""Generate Parse-Then-Place stage-2 reference outputs."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer  # ty: ignore[possibly-missing-import]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Run the stage-2 Transformers generation wrapper and save reference JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=["rico", "web"], required=True)
    parser.add_argument(
        "--stage2-mode", choices=["pretrain", "finetune"], required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-length", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)

    ckpt_dir = (
        args.original_root / "ckpt" / args.dataset_name / "stage2" / args.stage2_mode
    )
    data_path = (
        args.original_root
        / "data"
        / args.dataset_name
        / "stage2"
        / "finetune"
        / "test.json"
    )
    data = json.loads(data_path.read_text(encoding="utf-8"))[: args.num_examples]
    source_texts = [str(item["execution"]) for item in data]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(ckpt_dir)
    model = T5ForConditionalGeneration.from_pretrained(ckpt_dir).to(
        device  # ty: ignore[invalid-argument-type]
    )
    model.eval()
    tokenized = tokenizer(
        source_texts,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            eos_token_id=1,
            max_length=args.max_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=args.num_return_sequences,
        )
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    grouped_decoded = [
        decoded[idx * args.num_return_sequences : (idx + 1) * args.num_return_sequences]
        for idx in range(len(source_texts))
    ]
    metadata = {
        "dataset_name": args.dataset_name,
        "stage2_mode": args.stage2_mode,
        "seed": args.seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "4"),
        "reference_path": (
            "Transformers T5ForConditionalGeneration.generate call with "
            "generation arguments copied from the vendor stage-2 trainer"
        ),
        "vendor_code_executed": False,
        "num_examples": args.num_examples,
        "num_return_sequences": args.num_return_sequences,
        "temperature": args.temperature,
        "max_length": args.max_length,
        "goldens_policy": "Generated tensors/text live outside git.",
    }
    (args.output_dir / "reference_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    reference = {
        "source_texts": source_texts,
        "generated_ids": generated_ids.cpu().tolist(),
        "decoded": grouped_decoded,
    }
    (args.output_dir / "stage2_reference.json").write_text(
        json.dumps(reference, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

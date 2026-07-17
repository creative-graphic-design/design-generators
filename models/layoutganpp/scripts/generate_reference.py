from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from layoutganpp.datasets import labels_for_dataset


def _arg(args: object, key: str) -> object:
    if isinstance(args, dict):
        return args[key]
    return getattr(args, key)


def _synthetic_labels(
    *, batch_size: int, seq_len: int, num_labels: int, device: torch.device
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    labels = torch.arange(seq_len, dtype=torch.long, device=device).remainder(
        num_labels
    )
    labels = labels.unsqueeze(0).repeat(batch_size, 1)
    lengths = torch.arange(seq_len, seq_len - batch_size, -1, device=device).clamp_min(
        1
    )
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    attention_mask = positions < lengths.unsqueeze(1)
    return labels, attention_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()

    sys.path.insert(0, str(args.vendor_dir))
    from model.layoutganpp import Generator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint["args"]
    dataset_name = str(_arg(train_args, "dataset"))
    num_labels = len(labels_for_dataset(dataset_name))
    seq_len = 33 if dataset_name == "magazine" else 9
    model = (
        Generator(
            int(_arg(train_args, "latent_size")),
            num_labels,
            d_model=int(_arg(train_args, "G_d_model")),
            nhead=int(_arg(train_args, "G_nhead")),
            num_layers=int(_arg(train_args, "G_num_layers")),
        )
        .eval()
        .to(device)
    )
    model.load_state_dict(checkpoint["netG"])
    generator = torch.Generator(device=device).manual_seed(args.seed)
    labels, attention_mask = _synthetic_labels(
        batch_size=args.batch_size,
        seq_len=seq_len,
        num_labels=num_labels,
        device=device,
    )
    latents = torch.randn(
        labels.size(0),
        labels.size(1),
        int(_arg(train_args, "latent_size")),
        generator=generator,
        device=device,
    )
    with torch.no_grad():
        bbox = model(latents, labels, ~attention_mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dataset": dataset_name,
            "seed": args.seed,
            "labels": labels.cpu(),
            "attention_mask": attention_mask.cpu(),
            "latents": latents.cpu(),
            "bbox": bbox.cpu(),
        },
        args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()

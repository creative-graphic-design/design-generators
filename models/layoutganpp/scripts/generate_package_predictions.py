from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "models/layoutganpp/src"),
    str(ROOT / "lib/laygen/src"),
]

from layoutganpp import LayoutGANPPPipeline  # noqa: E402
from layoutganpp.configuration_layoutganpp import LayoutGANPPConfig  # noqa: E402
from layoutganpp.modeling_layoutganpp import LayoutGANPPModel  # noqa: E402
from layoutganpp.training.dataset import (  # noqa: E402
    LayoutGANPPDataset,
    collate_layoutganpp,
)
from laygen.modeling_outputs import LayoutGenerationOutput  # noqa: E402

EXPECTED_TEST_COUNTS = {"magazine": 392, "rico13": 3834, "publaynet": 4226}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=EXPECTED_TEST_COUNTS, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    resolved_path = ROOT / manifest["config"]["resolved_config"]
    resolved = json.loads(resolved_path.read_text())["resolved"]["model"]["init_args"]
    checkpoint_path = next((run_dir / "checkpoints").glob("*.ckpt"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    generator_state = {
        key.removeprefix("generator."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("generator.")
    }
    if not generator_state:
        raise RuntimeError(
            f"checkpoint has no package generator weights: {checkpoint_path}"
        )

    model = LayoutGANPPModel(
        LayoutGANPPConfig(
            dataset_name=args.dataset,
            latent_size=resolved["latent_size"],
            d_model=resolved["generator_d_model"],
            nhead=resolved["generator_nhead"],
            num_layers=resolved["generator_num_layers"],
        )
    )
    model.load_state_dict(generator_state, strict=True)
    model.to(torch.device(args.device)).eval()
    pipeline = LayoutGANPPPipeline(model=model)
    dataset = LayoutGANPPDataset(
        dataset_name=args.dataset,
        split="test",
        data_root=ROOT / ".cache/layoutganpp/data" / args.dataset,
        synthetic_size=0,
        seed=0,
    )
    if len(dataset) != EXPECTED_TEST_COUNTS[args.dataset]:
        raise RuntimeError(f"unexpected TEST count for {args.dataset}: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_layoutganpp,
    )
    torch.manual_seed(42005)
    predictions = []
    with torch.no_grad():
        for batch in loader:
            labels = cast(torch.Tensor, batch["labels"]).to(args.device)
            mask = cast(torch.Tensor, batch["mask"]).to(args.device)
            output = cast(
                LayoutGenerationOutput,
                pipeline(labels=labels, mask=mask, output_type="dataclass"),
            )
            for index in range(output.labels.shape[0]):
                valid = output.mask[index].bool()
                predictions.append(
                    (
                        output.bbox[index][valid].cpu().numpy(),
                        output.labels[index][valid].cpu().numpy(),
                    )
                )
    if len(predictions) != EXPECTED_TEST_COUNTS[args.dataset]:
        raise RuntimeError(
            f"generated {len(predictions)} predictions for {args.dataset}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pickle.dumps(predictions))
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "test_layout_count": len(predictions),
                "evaluation_seed": 42005,
                "checkpoint": str(checkpoint_path),
                "checkpoint_global_step": checkpoint["global_step"],
                "predictions": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

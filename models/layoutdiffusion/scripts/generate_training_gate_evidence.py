"""Generate lightweight LayoutDiffusion S3/S4 prelaunch gate evidence."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict, TypeGuard, cast

import torch
from jaxtyping import Float, Shaped
from layoutdiffusion import LayoutDiffusionConfig
from layoutdiffusion.training.config import LayoutDiffusionTrainingDatasetName
from layoutdiffusion.training.datamodule import LayoutDiffusionDataModule
from layoutdiffusion.training.lightning_module import LayoutDiffusionTrainingModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

EVIDENCE_ROOT = Path(".cache/layoutdiffusion/training-evidence")


class GateEvidenceMetadata(TypedDict):
    """Static settings recorded with a gate-evidence artifact."""

    dataset_name: str
    seed: int
    steps: int
    preconsume_train_batches: int
    processed_data_dir: str | None
    auxiliary_loss_weight: float


class GateEvidence(TypedDict):
    """Trajectory, data-order, and state artifacts for the prelaunch gate."""

    metadata: GateEvidenceMetadata
    trajectory: list[dict[str, float | int]]
    batch_heads: list[list[int]]
    lt_history: Float[torch.Tensor, "steps"]
    lt_count: Float[torch.Tensor, "steps"]
    ema: dict[str, Shaped[torch.Tensor, ...]]
    model_state: dict[str, Shaped[torch.Tensor, ...]]
    optimizer_state: dict[
        str,
        int | float | str | list[dict[str, int | float | str | list[int]]],
    ]


class SchedulerConfig(TypedDict):
    """Linear-annealing scheduler config returned by Lightning."""

    scheduler: torch.optim.lr_scheduler.LRScheduler
    interval: str


class OptimizerConfig(TypedDict):
    """Optimizer dictionary returned by Lightning configure_optimizers."""

    optimizer: torch.optim.Optimizer
    lr_scheduler: SchedulerConfig


DATASET_SETTINGS: dict[LayoutDiffusionTrainingDatasetName, dict[str, int | float]] = {
    "rico25": {"vocab_size": 159, "learning_rate": 0.00004, "lr_anneal_steps": 175000},
    "publaynet": {
        "vocab_size": 139,
        "learning_rate": 0.00005,
        "lr_anneal_steps": 400000,
    },
}


def _is_optimizer_config(value: OptimizerLRScheduler) -> TypeGuard[OptimizerConfig]:
    return isinstance(value, dict) and isinstance(value.get("lr_scheduler"), dict)


def _expect_optimizer_config(value: OptimizerLRScheduler) -> OptimizerConfig:
    if not _is_optimizer_config(value):
        raise TypeError("Expected optimizer and scheduler config")
    return value


def training_config(
    dataset_name: LayoutDiffusionTrainingDatasetName, *, tiny: bool
) -> LayoutDiffusionConfig:
    if tiny:
        return LayoutDiffusionConfig(
            dataset_name=dataset_name,
            seq_length=19,
            max_num_elements=3,
            diffusion_steps=10,
            num_channels=8,
            hidden_size=16,
            num_attention_heads=4,
            num_hidden_layers=1,
            intermediate_size=32,
            dropout=0.0,
            max_position_embeddings=19,
        )
    settings = DATASET_SETTINGS[dataset_name]
    return LayoutDiffusionConfig(
        dataset_name=dataset_name,
        seq_length=121,
        diffusion_steps=200,
        noise_schedule="gaussian_refine_pow2.5",
        num_channels=128,
        dropout=0.1,
        training_mode="discrete1",
        vocab_size=int(settings["vocab_size"]),
    )


def collect_package_evidence(
    *,
    dataset_name: LayoutDiffusionTrainingDatasetName,
    output_path: Path,
    processed_data_dir: str | None,
    steps: int,
    seed: int,
    preconsume_train_batches: int,
) -> GateEvidence:
    tiny = processed_data_dir is None
    settings = DATASET_SETTINGS[dataset_name]
    torch.manual_seed(seed)
    config = training_config(dataset_name, tiny=tiny)
    module = LayoutDiffusionTrainingModule(
        config=config,
        learning_rate=float(settings["learning_rate"]),
        scheduler="linear_anneal",
        lr_anneal_steps=int(settings["lr_anneal_steps"]),
        auxiliary_loss_weight=0.001,
        seed_mode="deterministic",
    )
    dm = LayoutDiffusionDataModule(
        dataset_name=dataset_name,
        config=config,
        batch_size=2 if tiny else 64,
        synthetic_size=(steps + preconsume_train_batches) * 2 if tiny else None,
        dataset_source="processed" if processed_data_dir is not None else "hf",
        processed_data_dir=processed_data_dir,
        preconsume_train_batches=preconsume_train_batches,
        processed_stream_rng_warmup=processed_data_dir is not None,
        num_workers=0,
    )
    optimizer_config = _expect_optimizer_config(module.configure_optimizers())
    optimizer = optimizer_config["optimizer"]
    lr_scheduler = optimizer_config["lr_scheduler"]["scheduler"]
    loader = iter(dm.train_dataloader())
    trajectory: list[dict[str, float | int]] = []
    batch_heads: list[list[int]] = []
    for step in range(steps):
        batch = cast(dict[str, Shaped[torch.Tensor, "..."]], next(loader))
        batch_heads.append(batch["input_ids"][:, :8].cpu().tolist())
        optimizer.zero_grad(set_to_none=True)
        loss = module.training_step(batch, step)
        loss.backward()
        grad_norm = _grad_norm(module.model.parameters())
        optimizer.step()
        module.update_ema()
        lr_scheduler.step()
        trajectory.append(
            {
                "step": step,
                "train_loss": float(loss.detach().cpu().item()),
                "kl_loss": float(
                    loss.detach().cpu().item()
                    - module.latest_step_trace["aux_loss"].cpu().item()
                ),
                "aux_loss": float(module.latest_step_trace["aux_loss"].cpu().item()),
                "grad_norm": grad_norm,
                "next_lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
    evidence: GateEvidence = {
        "metadata": {
            "dataset_name": dataset_name,
            "seed": seed,
            "steps": steps,
            "preconsume_train_batches": preconsume_train_batches,
            "processed_data_dir": processed_data_dir,
            "auxiliary_loss_weight": 0.001,
        },
        "trajectory": trajectory,
        "batch_heads": batch_heads,
        "lt_history": module.lt_history.detach().cpu().clone(),
        "lt_count": module.lt_count.detach().cpu().clone(),
        "ema": module.ema_state_dict(),
        "model_state": {
            key: value.detach().cpu()
            for key, value in module.model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(evidence, output_path)
    return evidence


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    sqsum = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            sqsum += float(parameter.grad.detach().pow(2).sum().item())
    return sqsum**0.5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=sorted(DATASET_SETTINGS), default="publaynet"
    )
    parser.add_argument("--processed-data-dir")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=102)
    parser.add_argument("--preconsume-train-batches", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=EVIDENCE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mode = "processed" if args.processed_data_dir is not None else "synthetic"
    path = args.output_root / f"s3-s4-{mode}-{args.dataset}.pt"
    evidence = collect_package_evidence(
        dataset_name=args.dataset,
        output_path=path,
        processed_data_dir=args.processed_data_dir,
        steps=args.steps,
        seed=args.seed,
        preconsume_train_batches=args.preconsume_train_batches,
    )
    print(f"wrote {path}")
    print(f"trajectory steps: {len(evidence['trajectory'])}")


if __name__ == "__main__":
    main()

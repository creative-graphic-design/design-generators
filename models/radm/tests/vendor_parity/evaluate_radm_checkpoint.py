"""Repository-owned, fail-closed entry point for RADM checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import torch

from radm.configuration_radm import RADMConfig
from radm.evaluation import evaluate_checkpoint
from radm.training.config import effective_radm_config
from radm.training.datamodule import RADMDataModule


EXPECTED_STEPS = 250000
DEFAULT_CONFIG = Path("models") / "radm" / "configs" / "training" / "radm_cgl.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_jobs(manifest: dict[str, object]) -> list[dict[str, object]]:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        return [manifest]
    return [cast(dict[str, object], job) for job in jobs if isinstance(job, dict)]


def _manifest_int(job: dict[str, object], *keys: str, default: int = -1) -> int:
    """Read the first integer-valued manifest field from the given keys."""
    for key in keys:
        value = job.get(key)
        if isinstance(value, int):
            return value
    return default


def _manifest_job(manifest: dict[str, object], seed: int) -> dict[str, object]:
    jobs = _manifest_jobs(manifest)
    matches = [
        job for job in jobs if _manifest_int(job, "training_seed", "seed") == seed
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"manifest must identify exactly one job for training seed {seed}"
        )
    return matches[0]


def validate_terminal_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    *,
    seed: int,
    config_path: Path = DEFAULT_CONFIG,
    expected_steps: int = EXPECTED_STEPS,
) -> dict[str, object]:
    """Validate terminal step and manifest recipe/run identity before evaluation."""
    config_sha256 = sha256(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job = _manifest_job(manifest, seed)
    manifest_config = job.get("source_config")
    if manifest_config != config_path.as_posix():
        raise RuntimeError("manifest config path does not match the S5 recipe")
    if job.get("source_config_sha256") != config_sha256:
        raise RuntimeError("manifest config digest does not match the S5 recipe")
    if _manifest_int(job, "max_steps") != expected_steps:
        raise RuntimeError("manifest max_steps does not match the S5 recipe")

    output_root = job.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise RuntimeError("manifest job has no output_root")
    run_dir = Path(output_root).resolve()
    resolved_checkpoint = checkpoint_path.resolve()
    if not resolved_checkpoint.is_relative_to(run_dir):
        raise RuntimeError("checkpoint is outside the manifest job output_root")

    payload = torch.load(checkpoint_path, map_location="meta", weights_only=False)
    global_step = int(payload.get("global_step", -1))
    if global_step != expected_steps:
        raise RuntimeError(
            f"checkpoint global_step={global_step}, expected {expected_steps}"
        )
    return {
        "global_step": global_step,
        "training_seed": seed,
        "config_path": config_path.as_posix(),
        "config_sha256": config_sha256,
    }


def evaluate_package_checkpoint(
    checkpoint_path: Path,
    *,
    data_root: Path,
    output_dir: Path,
    device: str,
    evaluation_seed: int,
) -> dict[str, object]:
    """Evaluate one guarded package checkpoint on the fixed CGL test stream."""
    config = RADMConfig(
        dataset_name="cgl",
        num_classes=4,
        num_proposals=100,
        hidden_dim=256,
        text_feature_dim=768,
        max_text_num=20,
        num_train_timesteps=1000,
        snr_scale=2.0,
        sample_step=1,
        backbone_depth=50,
    )
    effective = effective_radm_config()
    data_module = RADMDataModule(
        train_annotations=data_root / "annotations" / "train.json",
        train_image_root=data_root / "images" / "train",
        train_text_feature_root=data_root / "text_features" / "train",
        test_annotations=data_root / "annotations" / "test.json",
        test_image_root=data_root / "images" / "test",
        test_text_feature_root=data_root / "text_features" / "test",
        batch_size=1,
        num_workers=0,
        allow_missing_text_features=True,
        test_read_text_features=False,
        effective=effective,
    )
    report = evaluate_checkpoint(
        checkpoint_path,
        config=config,
        effective=effective,
        data_module=data_module,
        output_dir=output_dir,
        device=device,
        seed=evaluation_seed,
        num_inference_steps=1,
        class_threshold=0.25,
        nms_threshold=0.15,
    )
    return cast(dict[str, object], report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluation-seed", type=int, default=1)
    args = parser.parse_args()
    validation = validate_terminal_checkpoint(
        args.checkpoint,
        args.manifest,
        seed=args.seed,
    )
    report = evaluate_package_checkpoint(
        args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        evaluation_seed=args.evaluation_seed,
    )
    print(json.dumps({"validation": validation, "report": report}, allow_nan=True))


if __name__ == "__main__":
    main()

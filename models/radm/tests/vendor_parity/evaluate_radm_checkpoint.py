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
from radm.training.data_paths import validate_dot_free_data_root
from radm.training.launch_manifest import CGLS5LaunchManifest


ROOT = Path(__file__).resolve().parents[4]
EXPECTED_STEPS = 250000
DEFAULT_CONFIG = Path("models") / "radm" / "configs" / "training" / "radm_cgl.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_terminal_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    *,
    seed: int,
    config_path: Path = DEFAULT_CONFIG,
    expected_steps: int = EXPECTED_STEPS,
) -> dict[str, object]:
    """Validate the checkpoint against the launch manifest before evaluation."""
    if not checkpoint_path.is_file():
        raise RuntimeError(f"terminal checkpoint is missing: {checkpoint_path}")

    config_sha256 = sha256(config_path)
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise RuntimeError("launch manifest must be a JSON object")
    manifest = cast(CGLS5LaunchManifest, raw_manifest)
    if manifest.get("arm") != "package":
        raise RuntimeError("manifest arm does not match the package evaluator")
    if manifest.get("training_seed") != seed:
        raise RuntimeError("manifest training_seed does not match the evaluator seed")
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, dict):
        raise RuntimeError("manifest has no config identity")
    expected_config_path = config_path.resolve().relative_to(ROOT).as_posix()
    if manifest_config.get("path") != expected_config_path:
        raise RuntimeError("manifest config path does not match the S5 recipe")
    if manifest_config.get("sha256") != config_sha256:
        raise RuntimeError("manifest config digest does not match the S5 recipe")
    if manifest.get("max_steps") != expected_steps:
        raise RuntimeError("manifest max_steps does not match the S5 recipe")

    output_root = manifest.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise RuntimeError("manifest has no output_root")
    resolved_run_dir = Path(output_root).resolve()
    resolved_manifest_dir = manifest_path.resolve().parent
    if resolved_run_dir != resolved_manifest_dir:
        raise RuntimeError("manifest output_root does not match the manifest directory")

    resolved_checkpoint = checkpoint_path.resolve()
    if not resolved_checkpoint.is_relative_to(resolved_run_dir):
        raise RuntimeError("checkpoint is outside the manifest output_root")

    checkpoint_rule = manifest.get("checkpoint_rule")
    if not isinstance(checkpoint_rule, dict):
        raise RuntimeError("manifest has no checkpoint rule")
    manifest_checkpoint = checkpoint_rule.get("checkpoint")
    if not isinstance(manifest_checkpoint, str) or (
        Path(manifest_checkpoint).resolve() != resolved_checkpoint
    ):
        raise RuntimeError(
            "checkpoint path does not match the manifest checkpoint rule"
        )

    payload = torch.load(checkpoint_path, map_location="meta", weights_only=False)
    global_step = int(payload.get("global_step", -1))
    if global_step != expected_steps:
        raise RuntimeError(
            f"checkpoint global_step={global_step}, expected {expected_steps}"
        )
    return {
        "arm": "package",
        "global_step": global_step,
        "training_seed": seed,
        "config_path": expected_config_path,
        "config_sha256": config_sha256,
        "checkpoint": str(resolved_checkpoint),
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
    data_root = Path(validate_dot_free_data_root(data_root))
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
        test_read_text_features=True,
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
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluation-seed", type=int, default=1)
    args = parser.parse_args()
    data_root = validate_dot_free_data_root(args.data_root)
    validation = validate_terminal_checkpoint(
        args.checkpoint,
        args.manifest,
        seed=args.seed,
    )
    report = evaluate_package_checkpoint(
        args.checkpoint,
        data_root=Path(data_root),
        output_dir=args.output_dir,
        device=args.device,
        evaluation_seed=args.evaluation_seed,
    )
    print(json.dumps({"validation": validation, "report": report}, allow_nan=True))


if __name__ == "__main__":
    main()

"""Write the manifest for a full Crello run from its preflight evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def main() -> None:
    root = _repo_root()
    tool_root = Path(__file__).resolve().parents[1]
    run_root = _path(
        "CANVAS_VAE_S5_RUN_ROOT",
        root / ".cache" / "canvas-vae" / "full-run" / "crello" / "path-a",
    )
    inputs = _path(
        "CANVAS_VAE_S5_INPUTS",
        root / ".cache" / "canvas-vae" / "s5-path-a" / "inputs",
    )
    original_data = _path("CANVAS_VAE_ORIGINAL_DATA", inputs / "original-data")
    package_data = _path("CANVAS_VAE_PACKAGE_DATA", inputs / "package-data")
    embedding_cache = _path("CANVAS_VAE_EMBEDDING_CACHE", inputs / "cache")
    lockstep = _path(
        "CANVAS_VAE_LOCKSTEP_RUN_ROOT",
        root
        / ".cache"
        / "canvas-vae"
        / "s5-path-a"
        / "pre-s5"
        / "lockstep-split-20261108-1653"
        / "run-workers8-seeded",
    )
    original_losses = _path(
        "CANVAS_VAE_ORIGINAL_LOSS_ROOT",
        root
        / ".cache"
        / "canvas-vae"
        / "s5-path-a"
        / "pre-s5"
        / "cu126-independent-lanes-20261108-0355"
        / "original"
        / "loss-envelope",
    )
    package_losses = _path(
        "CANVAS_VAE_PACKAGE_LOSS_ROOT",
        root
        / ".cache"
        / "canvas-vae"
        / "s5-path-a"
        / "pre-s5"
        / "cu126-independent-lanes-20261108-0355"
        / "package"
        / "loss-envelope",
    )
    package_seed_runs = _path(
        "CANVAS_VAE_PACKAGE_SEED_ROOT",
        root
        / ".cache"
        / "canvas-vae"
        / "s5-path-a"
        / "pre-s5"
        / "cu126-gpu7-seeds-new-code-20261108-1605",
    )
    evaluation_parity = _path(
        "CANVAS_VAE_EVALUATION_PARITY_RECEIPT",
        root
        / ".cache"
        / "canvas-vae"
        / "full-run"
        / "crello"
        / "evaluation_path_parity.receipt.json",
    )
    gpu = os.environ["CANVAS_VAE_GPU"]
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def rel(path: Path) -> str:
        return path.resolve().relative_to(root).as_posix()

    artifacts_to_hash = [
        root
        / "models"
        / "canvas-vae"
        / "configs"
        / "training"
        / "canvas_vae_crello.yaml",
        root / "models" / "canvas-vae" / "scripts" / "source_tree_guard.py",
        root / "models" / "canvas-vae" / "scripts" / "convert_original_checkpoint.py",
        root / "models" / "canvas-vae" / "scripts" / "evaluate_crello.py",
        root
        / "models"
        / "canvas-vae"
        / "tests"
        / "vendor_parity"
        / "metadata"
        / "crello_s5_fc432"
        / "manifest.json",
        run_root / "stack" / "package-pip-freeze.txt",
        run_root / "stack" / "tensorflow-pip-freeze.txt",
        run_root / "preflight" / "package-stack.json",
        run_root / "preflight" / "original-stack.json",
        run_root / "preflight" / "package-run-metadata.json",
        run_root / "preflight" / "original-run-metadata.json",
        run_root / "logs" / "package-preflight.log",
        run_root / "logs" / "original-preflight.log",
        embedding_cache / "manifest.json",
        embedding_cache / "posterior_means.npy",
        original_data / "count.json",
        original_data / "crello-s5-manifest.json",
        original_data / "train-00000-of-00001.tfrecord",
        original_data / "val-00000-of-00001.tfrecord",
        original_data / "test-00000-of-00001.tfrecord",
        original_data / "vocabulary.json",
        package_data / "count.json",
        package_data / "crello-manifest.json",
        package_data / "train.jsonl",
        package_data / "val.jsonl",
        package_data / "test.jsonl",
        package_data / "vocabulary.json",
        evaluation_parity,
        lockstep / "source-record.json",
        lockstep / "summary.json",
    ]
    artifacts_to_hash.extend(
        path
        for path in sorted(tool_root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    for seed in range(3):
        artifacts_to_hash.extend(
            [
                original_losses / f"seed-{seed}" / "update-trace.jsonl",
                original_losses / f"seed-{seed}" / "training" / "run-metadata.json",
                package_losses / f"seed-{seed}" / "update-trace.jsonl",
                package_seed_runs / f"seed-{seed}" / "update-trace.jsonl",
                package_seed_runs / f"seed-{seed}" / "source-tree-guard.json",
                package_seed_runs / f"seed-{seed}" / "training" / "run-metadata.json",
            ]
        )

    artifacts: dict[str, str] = {}
    missing = []
    for path in artifacts_to_hash:
        if not path.is_file():
            missing.append(rel(path))
            continue
        artifacts[rel(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if missing:
        raise SystemExit(
            "required prelaunch artifacts are missing: " + ", ".join(missing)
        )

    gpu_name = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
            "-i",
            gpu,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "source_commit": source_commit,
        "config": "models/canvas-vae/configs/training/canvas_vae_crello.yaml",
        "evaluator_command": (
            "Per training seed: $CANVAS_VAE_ORIGINAL_ENV/bin/python "
            "models/canvas-vae/tests/vendor_parity/crello_s5/helpers/evaluate_original_best.py "
            "--data-dir $CANVAS_VAE_ORIGINAL_DATA --checkpoint "
            "$CANVAS_VAE_S5_RUN_ROOT/original/seed-$SEED/training/checkpoints/best.ckpt "
            "--output-dir $CANVAS_VAE_S5_RUN_ROOT/original/seed-$SEED/evaluation --seeds 0 1 2; "
            "$CANVAS_VAE_PACKAGE_ENV/bin/python models/canvas-vae/scripts/convert_original_checkpoint.py "
            "--checkpoint $CANVAS_VAE_S5_RUN_ROOT/package/seed-$SEED/training/lightning_logs/version_0/checkpoints/best.ckpt "
            "--vocabulary $CANVAS_VAE_PACKAGE_DATA/vocabulary.json --dataset crello "
            "--output-dir $CANVAS_VAE_S5_RUN_ROOT/package/seed-$SEED/evaluated-model; "
            "$CANVAS_VAE_PACKAGE_ENV/bin/python models/canvas-vae/scripts/evaluate_crello.py "
            "--checkpoint $CANVAS_VAE_S5_RUN_ROOT/package/seed-$SEED/evaluated-model "
            "--data-dir $CANVAS_VAE_PACKAGE_DATA --fixture-dir $CANVAS_VAE_EMBEDDING_CACHE "
            "--seeds 0 1 2 --batch-size 1024 --device cpu --expected-test-documents 2278"
        ),
        "checkpoint_rule": (
            "Original: validation-best top-1 best.ckpt (val_total_score, max); "
            "package: validation-best top-1 best.ckpt (val/total_score, max). "
            "Neither system evaluates last.ckpt."
        ),
        "package_checkpoint_selection": {
            "filename": "best.ckpt",
            "monitor": "val/total_score",
            "mode": "max",
            "save_top_k": 1,
        },
        "launched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpu": {
            "index": int(gpu),
            "name": gpu_name,
            "placement_rule": (
                "One package process and one original process share the selected GPU; "
                "utilization, power, and CPU telemetry are report-only."
            ),
        },
        "training": {
            "seeds": [0, 1, 2],
            "epochs": 500,
            "batch_size": 1024,
            "precision": "float32",
            "package_num_workers": 8,
            "package_log_every_n_steps": 50,
            "package_progress_bar": False,
            "foreach_clipping": True,
            "cache_sha256": "fc432517afc2732c0b1ff218e3b0064800ce3f27349b2d96777cb5b08be7ed1a",
        },
        "prerequisites": {
            "production_path_loss_envelope": "passed for both systems; three matched seeds and at least 1,500 updates per seed",
            "real_scale_lockstep": "300 steps; floating-point noise attribution passed",
            "gpu_bound_check": "waived for both systems by issue comment https://github.com/creative-graphic-design/design-generators/issues/31#issuecomment-6719758806",
            "evaluation_path_parity": "passed for 2,278 documents",
        },
        "artifacts": artifacts,
        "status": "prepared",
    }
    destination = run_root / "manifest.json"
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": rel(destination),
                "source_commit": source_commit,
                "artifact_count": len(artifacts),
                "launched_at": manifest["launched_at"],
            }
        )
    )


if __name__ == "__main__":
    main()

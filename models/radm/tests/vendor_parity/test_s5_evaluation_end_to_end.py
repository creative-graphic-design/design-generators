"""End-to-end CGL evaluation parity for one shared checkpoint and image stream."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from textwrap import dedent
from pathlib import Path
from typing import cast

import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from radm.training.data_paths import validate_dot_free_data_root

import evaluate_radm_checkpoint as evaluation
from evaluate_radm_checkpoint import (
    evaluate_package_checkpoint,
    validate_terminal_checkpoint,
)


ROOT = Path(__file__).resolve().parents[4]
VENDOR_ROOT = ROOT / "vendor" / "radm"
pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]


def _required_env(name: str) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value)
    skip_or_fail_vendor_parity(
        f"set {name} for the shared-checkpoint CGL end-to-end evaluation",
        regeneration_hint=(
            "set the package checkpoint, converted source checkpoint, and launch "
            "manifest paths used by the seed-1 cross-evaluation"
        ),
    )
    raise AssertionError(name)


def _vendor_eval_command(
    *, checkpoint: Path, output_dir: Path, data_root: Path, evaluation_seed: int
) -> list[str]:
    launcher = dedent(
        """
        import json
        import os
        import pdb
        import runpy
        import torch
        from RADM.dataset_mapper import RADMDatasetMapper

        pdb.set_trace = lambda: None
        load_text = RADMDatasetMapper.load_text
        def audit_text(self, *args, **kwargs):
            features, mask = load_text(self, *args, **kwargs)
            valid_count = int(mask.sum())
            nonzero_count = int(torch.count_nonzero(features["feats"]))
            if valid_count or nonzero_count:
                with open(os.environ["RADM_EVAL_TEXT_AUDIT_PATH"], "a", encoding="utf-8") as stream:
                    stream.write(json.dumps({
                        "text_name": str(args[0]),
                        "valid_count": valid_count,
                        "nonzero_count": nonzero_count,
                    }) + "\\n")
            return features, mask

        RADMDatasetMapper.load_text = audit_text
        runpy.run_path("train_net.py", run_name="__main__")
        """
    )
    return [
        sys.executable,
        "-c",
        launcher,
        "--num-gpus",
        "1",
        "--config-file",
        "configs/radm.yaml",
        "--eval-only",
        "--resume",
        "SEED",
        str(evaluation_seed),
        "MODEL.WEIGHTS",
        str(checkpoint),
        "DATASETS.DATASET_PATH",
        str(data_root),
        "DATASETS.TEXT_FEATURE_PATH",
        str(data_root / "text_features"),
        "OUTPUT_DIR",
        str(output_dir),
    ]


@pytest.mark.vendor_parity
def test_s5_package_and_vendor_evaluators_use_the_same_checkpoint_and_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare evaluator coverage and confirm both test loaders read text."""
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.skip("PARITY_REQUIRE=1 is required for end-to-end evaluation parity")
    package_checkpoint = _required_env("RADM_E2E_PACKAGE_CHECKPOINT")
    vendor_checkpoint = _required_env("RADM_E2E_VENDOR_CHECKPOINT")
    manifest = _required_env("RADM_E2E_PACKAGE_MANIFEST")
    data_root = Path(
        validate_dot_free_data_root(os.environ.get("RADM_S4_DATA_ROOT", "radm_cgl"))
    )
    assert data_root.is_absolute()
    test_image_stems = {
        Path(str(image["file_name"])).stem
        for image in json.loads(
            (data_root / "annotations" / "test.json").read_text(encoding="utf-8")
        )["images"]
    }
    from radm.training import dataset as package_dataset
    from radm.training.config import RADMEffectiveConfig

    evaluation_seed = int(os.environ.get("RADM_E2E_EVALUATION_SEED", "1"))
    package_text_reads: list[tuple[str, int, int]] = []
    load_text_features = package_dataset.load_text_features

    def audit_package_text(
        image_name: str,
        *,
        root: Path,
        effective: RADMEffectiveConfig,
        allow_missing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, mask = load_text_features(
            image_name,
            root=root,
            effective=effective,
            allow_missing=allow_missing,
        )
        feature_root = root
        if feature_root == data_root / "text_features" / "test":
            package_text_reads.append(
                (
                    Path(image_name).stem,
                    int(mask.sum().item()),
                    int(torch.count_nonzero(features).item()),
                )
            )
        return features, mask

    monkeypatch.setattr(package_dataset, "load_text_features", audit_package_text)
    validation = validate_terminal_checkpoint(
        package_checkpoint,
        manifest,
        seed=int(os.environ.get("RADM_E2E_TRAINING_SEED", "1")),
        data_root=str(data_root),
    )
    package_output = tmp_path / "package"
    package_report = evaluate_package_checkpoint(
        package_checkpoint,
        data_root=data_root,
        output_dir=package_output,
        device=os.environ.get("RADM_REFERENCE_DEVICE", "cuda"),
        evaluation_seed=evaluation_seed,
    )
    assert any(
        image_stem in test_image_stems and valid_count > 0 and nonzero_count > 0
        for image_stem, valid_count, nonzero_count in package_text_reads
    ), "package evaluator did not consume nonzero text features from the test split"
    vendor_output = tmp_path / "vendor"
    vendor_text_audit = tmp_path / "vendor-text-features.jsonl"
    environment = os.environ.copy()
    environment["RADM_EVAL_TEXT_AUDIT_PATH"] = str(vendor_text_audit)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "models" / "radm" / "scripts" / "vendor_compat"),
            str(VENDOR_ROOT),
            environment.get("PYTHONPATH", ""),
        ]
    )
    completed = subprocess.run(
        _vendor_eval_command(
            checkpoint=vendor_checkpoint,
            output_dir=vendor_output,
            data_root=data_root,
            evaluation_seed=evaluation_seed,
        ),
        cwd=VENDOR_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    vendor_text_reads = (
        [
            json.loads(line)
            for line in vendor_text_audit.read_text(encoding="utf-8").splitlines()
        ]
        if vendor_text_audit.is_file()
        else []
    )
    assert any(
        Path(str(row["text_name"])).stem in test_image_stems
        and row["valid_count"] > 0
        and row["nonzero_count"] > 0
        for row in vendor_text_reads
    ), "vendor evaluator did not consume nonzero text features from the test split"
    package_predictions = json.loads(
        (package_output / "coco_instances_results.json").read_text(encoding="utf-8")
    )
    vendor_predictions = json.loads(
        (vendor_output / "inference" / "coco_instances_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["global_step"] == 250000
    package_results = cast(dict[str, object], package_report["results"])
    assert cast(dict[str, object], package_results["bbox"])
    assert {row["image_id"] for row in package_predictions} == {
        row["image_id"] for row in vendor_predictions
    }
    assert len(package_predictions) == len(vendor_predictions)


def test_package_checkpoint_evaluator_reads_test_text_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    class CapturingDataModule:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

    monkeypatch.setattr(evaluation, "RADMDataModule", CapturingDataModule)
    monkeypatch.setattr(evaluation, "evaluate_checkpoint", lambda *args, **kwargs: {})
    monkeypatch.chdir("/tmp")
    visible_root = Path.cwd() / "radm_cgl"

    evaluate_package_checkpoint(
        tmp_path / "checkpoint.ckpt",
        data_root=Path("radm_cgl"),
        output_dir=tmp_path / "output",
        device="cpu",
        evaluation_seed=1,
    )

    assert observed["test_read_text_features"] is True
    assert observed["test_image_root"] == visible_root / "images" / "test"


def test_vendor_evaluator_command_uses_absolute_visible_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir("/tmp")
    data_root = Path(validate_dot_free_data_root("radm_cgl"))
    command = _vendor_eval_command(
        checkpoint=tmp_path / "model_final.pth",
        output_dir=tmp_path / "evaluation",
        data_root=data_root,
        evaluation_seed=2,
    )

    compile(command[2], "<vendor-eval-launcher>", "exec")
    root_argument = command[command.index("DATASETS.DATASET_PATH") + 1]
    assert root_argument == str(data_root)
    assert Path(root_argument).is_absolute()
    text_root_argument = command[command.index("DATASETS.TEXT_FEATURE_PATH") + 1]
    assert text_root_argument == str(data_root / "text_features")

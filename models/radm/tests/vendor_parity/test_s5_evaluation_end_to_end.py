"""End-to-end CGL evaluation parity for one shared checkpoint and image stream."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from laygen.common.testing import skip_or_fail_vendor_parity

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
    *, checkpoint: Path, output_dir: Path, data_root: Path
) -> list[str]:
    launcher = (
        "import pdb, runpy; pdb.set_trace=lambda: None; "
        "from PIL import Image; Image.LINEAR=Image.BILINEAR; "
        "runpy.run_path('train_net.py', run_name='__main__')"
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
    tmp_path: Path,
) -> None:
    """Run both evaluators and require identical image coverage and count."""
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.skip("PARITY_REQUIRE=1 is required for end-to-end evaluation parity")
    package_checkpoint = _required_env("RADM_E2E_PACKAGE_CHECKPOINT")
    vendor_checkpoint = _required_env("RADM_E2E_VENDOR_CHECKPOINT")
    manifest = _required_env("RADM_E2E_PACKAGE_MANIFEST")
    data_root = Path(os.environ.get("RADM_S4_DATA_ROOT", ".cache/radm/data/cgl"))
    validation = validate_terminal_checkpoint(
        package_checkpoint,
        manifest,
        seed=int(os.environ.get("RADM_E2E_TRAINING_SEED", "1")),
    )
    package_output = tmp_path / "package"
    package_report = evaluate_package_checkpoint(
        package_checkpoint,
        data_root=data_root,
        output_dir=package_output,
        device=os.environ.get("RADM_REFERENCE_DEVICE", "cuda"),
        evaluation_seed=int(os.environ.get("RADM_E2E_EVALUATION_SEED", "1")),
    )
    vendor_output = tmp_path / "vendor"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(VENDOR_ROOT), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        _vendor_eval_command(
            checkpoint=vendor_checkpoint,
            output_dir=vendor_output,
            data_root=data_root,
        ),
        cwd=VENDOR_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
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

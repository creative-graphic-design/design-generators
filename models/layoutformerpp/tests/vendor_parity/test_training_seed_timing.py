from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[2] / "scripts" / "publaynet_s5"
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from launch_track import build_argv, build_environment, read_data_root  # noqa: E402

pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]
PROBE_DIRECTORY = Path(__file__).with_name("seed_timing_probe")


def _production_record(
    arm: str, seed: int, track: Path, *, model_only: bool = False
) -> dict[str, object]:
    track.mkdir(parents=True, exist_ok=True)
    package_environment = Path(os.environ["PACKAGE_ENV"])
    environment_file = Path(os.environ["LAYOUTFORMERPP_DATA_ROOT_ENV_FILE"])
    data_root = read_data_root()
    repository = Path(__file__).resolve().parents[4]
    command = build_argv("label", seed, arm, track, package_environment)[0]
    if arm == "package":
        command.append("--trainer.accelerator=cpu")
    environment = build_environment(
        os.environ,
        seed=seed,
        arm=arm,
        gpu_index=0,
        gpu_uuid="",
        data_root=data_root,
        environment_file=environment_file,
        package_environment=package_environment,
        track_directory=track,
    )
    record_path = track / "seed-timing.json"
    package_source = repository / "models" / "layoutformerpp" / "src"
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PROBE_ARM": arm,
            "PROBE_MODEL_ONLY": str(model_only),
            "PROBE_RECORD_PATH": str(record_path),
            "PYTHONPATH": os.pathsep.join(
                (
                    str(PROBE_DIRECTORY),
                    str(package_source),
                    environment.get("PYTHONPATH", ""),
                )
            ),
        }
    )
    if arm == "original":
        environment["LAYOUTFORMERPP_VENDOR_BOOTSTRAP"] = str(
            PROBE_DIRECTORY / "vendor_bootstrap.py"
        )
    process = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        timeout=900,
    )
    if process.returncode != 0 or not record_path.is_file():
        error = process.stderr.decode("utf-8", errors="replace")
        for secret in (data_root, environment_file, package_environment, repository):
            error = error.replace(str(secret), "<redacted>")
        raise AssertionError(
            f"{arm} production seed probe failed with exit {process.returncode}: "
            f"{error[-1200:]}"
        )

    return json.loads(record_path.read_text(encoding="utf-8"))


@pytest.mark.skipif(
    not os.environ.get("PACKAGE_ENV")
    or not os.environ.get("LAYOUTFORMERPP_DATA_ROOT_ENV_FILE"),
    reason="requires the audited package runtime and configured processed data",
)
def test_s0_real_entries_keep_unseeded_init_and_match_seeded_loader(tmp_path: Path):
    original = _production_record("original", 0, tmp_path / "original")
    package = _production_record("package", 0, tmp_path / "package")
    package_repeat = _production_record(
        "package", 0, tmp_path / "package-repeat", model_only=True
    )

    seed_calls = cast(list[dict[str, object]], package["seed_calls"])
    weight_digest = package["initial_weights_sha256"]
    repeat_weight_digest = package_repeat["initial_weights_sha256"]
    assert all(call.get("stage") != "before_model" for call in seed_calls)
    assert isinstance(weight_digest, str) and len(weight_digest) == 64
    assert isinstance(repeat_weight_digest, str) and len(repeat_weight_digest) == 64
    assert weight_digest != repeat_weight_digest
    assert package["seed_calls"] == original["seed_calls"]
    assert package["pre_loader_rng_sha256"] == original["pre_loader_rng_sha256"]
    assert package["first_loader_sample_ids"] == original["first_loader_sample_ids"]

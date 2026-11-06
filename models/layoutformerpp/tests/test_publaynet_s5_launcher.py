from __future__ import annotations

import io
import hashlib
import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts" / "publaynet_s5"
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from launch_track import (  # noqa: E402
    DATA_ROOT_MARKER,
    FAMILIES,
    JSONValue,
    REPOSITORY,
    VENDOR_SCRIPTS,
    build_argv,
    build_environment,
    config_path,
    pin_pair_gpu_exclusively,
    read_data_root,
    read_data_root_value,
    record_failure,
    stage_original_checkpoint,
    stream_redacted,
    validate_campaign_root,
    write_command_log,
    write_manifest,
)


def test_publaynet_launcher_argv_and_outputs_keep_data_root_private(
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "private" / "processed-data"
    package_environment = tmp_path / "runtime"
    track = tmp_path / "campaign" / "label" / "seed-1" / "package"

    for family in FAMILIES:
        package_argv = build_argv(family, 1, "package", track, package_environment)
        assert package_argv == [
            [
                str(package_environment / "bin" / "traingen"),
                "fit",
                "--config",
                config_path(family).relative_to(REPOSITORY).as_posix(),
                "--seed_everything=1",
                "--trainer.devices=1",
                f"--trainer.default_root_dir={track / 'lightning'}",
            ]
        ]

        original_argv = build_argv(family, 1, "original", track, package_environment)
        assert [command[2] for command in original_argv] == ["train", "test"]
        assert all(command[3] == DATA_ROOT_MARKER for command in original_argv)
        assert all(
            Path(command[1]).name == VENDOR_SCRIPTS[family] for command in original_argv
        )
        assert all(
            command[4] == str(track / "vendor-output") for command in original_argv
        )
        assert all(command[5:] == ["basic", "1", "s5"] for command in original_argv)
        assert all(
            str(fake_root) not in " ".join(command)
            for command in (*package_argv, *original_argv)
        )

    manifest_path = tmp_path / "launch_manifest.json"
    log_path = tmp_path / "launcher.log"
    environment_file = tmp_path / "data-root.env"
    environment_file.write_text(f"LAYOUTFORMERPP_DATA_ROOT={fake_root}\n")
    assert read_data_root_value(environment_file) == str(fake_root)
    secrets = (str(fake_root),)
    manifest = {
        "commands": [["bash", "vendor.sh", "--data_dir", str(fake_root)]],
        "family": "label",
    }
    write_manifest(manifest_path, manifest, secrets)
    with log_path.open("w", encoding="utf-8") as log:
        write_command_log(log, manifest["commands"][0], secrets)
    child_output = io.BytesIO(f"output: {fake_root}\n".encode())
    child_log = io.BytesIO()
    stream_redacted(child_output, child_log, secrets, chunk_size=7)

    assert str(fake_root) not in manifest_path.read_text(encoding="utf-8")
    assert str(fake_root) not in log_path.read_text(encoding="utf-8")
    assert (
        json.loads(manifest_path.read_text(encoding="utf-8"))["commands"][0][-1]
        == "<REDACTED>"
    )
    assert str(fake_root).encode() not in child_log.getvalue()


def test_vendor_best_epoch_is_staged_for_test_and_recorded(tmp_path: Path) -> None:
    vendor_output = tmp_path / "vendor-output"
    vendor_output.mkdir()
    source = vendor_output / "epoch_17_checkpoint.pth.tar"
    source.write_bytes(b"selected vendor checkpoint")
    (vendor_output / "best_epoch").write_text("17\n", encoding="utf-8")

    selection = stage_original_checkpoint(vendor_output)

    staged = vendor_output / "s5_checkpoint.pth.tar"
    assert staged.read_bytes() == source.read_bytes()
    assert selection == {
        "source": source.name,
        "staged_as": staged.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def test_pair_gpu_lock_is_released_after_pinning(tmp_path: Path) -> None:
    pair_path = tmp_path / "label" / "seed-0" / "gpu-type.json"

    pin_pair_gpu_exclusively(pair_path, "label", 0, "A100")
    pin_pair_gpu_exclusively(pair_path, "label", 0, "A100")

    assert json.loads(pair_path.read_text(encoding="utf-8")) == {
        "family": "label",
        "seed": 0,
        "gpu_type": "A100",
    }


def test_cuda_visibility_uses_the_selected_gpu_uuid(tmp_path: Path) -> None:
    environment = build_environment(
        {},
        seed=0,
        arm="package",
        gpu_index=5,
        gpu_uuid="GPU-test-uuid",
        data_root=tmp_path / "data",
        environment_file=tmp_path / "data-root.env",
        package_environment=tmp_path / "runtime",
        track_directory=tmp_path / "campaign" / "label" / "seed-0" / "package",
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-test-uuid"


def test_campaign_root_must_match_its_repository_relative_name() -> None:
    campaign_rel = ".cache/layoutformerpp/s5/example"
    expected = (REPOSITORY / campaign_rel).resolve()

    assert validate_campaign_root(campaign_rel, expected) == expected
    with pytest.raises(RuntimeError, match="CAMPAIGN_ROOT does not match CAMPAIGN_REL"):
        validate_campaign_root(campaign_rel, REPOSITORY / ".cache/other")


def test_pretraining_failure_is_recorded_as_retryable(tmp_path: Path) -> None:
    track_directory = tmp_path / "track"
    track_directory.mkdir()
    manifest_path = track_directory / "launch_manifest.json"
    manifest: dict[str, JSONValue] = {"status": "prepared"}
    write_manifest(manifest_path, manifest, ())

    record_failure(
        track_directory,
        manifest_path,
        manifest,
        (),
        training_started=False,
        phase="training",
        exit_code=1,
    )

    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert recorded["status"] == "failed_before_training"
    assert recorded["exit_code"] == 1
    assert recorded["failure_phase"] == "training"
    assert (track_directory / "exit-code.txt").read_text() == "1\n"


def test_missing_data_root_error_does_not_reveal_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_root = tmp_path / "private-data-root" / "missing"
    environment_file = tmp_path / "data-root.env"
    environment_file.write_text(
        f"LAYOUTFORMERPP_DATA_ROOT={secret_root}\n", encoding="utf-8"
    )
    monkeypatch.setenv("LAYOUTFORMERPP_DATA_ROOT_ENV_FILE", str(environment_file))

    with pytest.raises(
        RuntimeError, match="configured data root is unavailable"
    ) as error:
        read_data_root()

    assert str(secret_root) not in str(error.value)

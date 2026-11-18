from __future__ import annotations

import io
import hashlib
import json
import shlex
import sys
import time
import zipfile
from pathlib import Path

import pytest
import torch


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts" / "publaynet_s5"
sys.path.insert(0, str(SCRIPT_DIRECTORY))

import launch_track  # noqa: E402
from launch_track import (  # noqa: E402
    DATA_ROOT_MARKER,
    FAMILIES,
    JSONValue,
    REPOSITORY,
    VENDOR_SCRIPTS,
    build_argv,
    build_environment,
    config_path,
    parse_args,
    pin_pair_gpu_exclusively,
    prune_vendor_checkpoints,
    read_data_root,
    read_data_root_value,
    record_failure,
    stage_original_checkpoint,
    stream_redacted,
    validate_campaign_root,
    validate_checkpoint_pruning,
    vendor_training_schedule,
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


def test_package_resume_requires_durable_root_and_uses_matching_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    environment_file = tmp_path / "data-root.env"
    environment_file.write_text("LAYOUTFORMERPP_DATA_ROOT=/unused\n", encoding="utf-8")
    campaign_root = tmp_path / "campaign"
    package_environment = tmp_path / "runtime"
    campaign_rel = ".cache/resume-launch-test"
    monkeypatch.setenv("PACKAGE_ENV", str(package_environment))
    monkeypatch.setenv("CAMPAIGN_ROOT", str(campaign_root))
    monkeypatch.setenv("LAYOUTFORMERPP_DATA_ROOT_ENV_FILE", str(environment_file))
    monkeypatch.delenv("LAYOUTFORMERPP_DURABLE_ROOT", raising=False)
    monkeypatch.setattr(
        launch_track.sys, "executable", str(package_environment / "bin" / "python")
    )
    monkeypatch.setattr(
        launch_track, "validate_campaign_root", lambda *_: campaign_root
    )
    monkeypatch.setattr(launch_track, "read_data_root", lambda: data_root)
    monkeypatch.setattr(launch_track, "verify_splits", lambda _: {})
    monkeypatch.setattr(launch_track, "runtime_freeze_sha256", lambda _: "freeze")
    monkeypatch.setattr(launch_track, "require_clean_tree", lambda: None)
    monkeypatch.setattr(launch_track, "gpu_info", lambda _: ("test-device", "GPU-test"))
    commands: list[list[str]] = []
    monkeypatch.setattr(
        launch_track,
        "run",
        lambda command, *_args, **_kwargs: commands.append(command) or 0,
    )

    track = campaign_root / "label" / "seed-0" / "package"
    checkpoint_directory = track / "lightning" / "package-checkpoints"
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    assert (
        launch_track.launch(
            campaign_rel, "label", 0, "package", 0, durable_root=durable_root
        )
        == 0
    )
    assert commands[0] == build_argv("label", 0, "package", track, package_environment)[
        0
    ] + [f"--trainer.callbacks.init_args.dirpath={checkpoint_directory.resolve()}"]
    assert "--ckpt_path" not in commands[0]
    manifest_path = track / "launch_manifest.json"
    identity = json.loads(manifest_path.read_text(encoding="utf-8"))

    with pytest.raises(RuntimeError, match="--durable-root"):
        launch_track.launch(campaign_rel, "label", 0, "package", 0, resume=True)
    assert len(commands) == 1

    checkpoint_directory.mkdir(parents=True)
    checkpoint_identity = json.loads(
        (track / "checkpoint_identity.json").read_text(encoding="utf-8")
    )
    checkpoint_identity["campaign_root"] = str(campaign_root.resolve())
    torch.save(
        {
            "state_dict": {},
            "layoutformerpp_identity": checkpoint_identity,
        },
        checkpoint_directory / "epoch-000099-step-000000396.ckpt",
    )
    torch.save(
        {
            "state_dict": {},
            "layoutformerpp_identity": dict(checkpoint_identity, device="other"),
        },
        checkpoint_directory / "epoch-000098-step-000000392.ckpt",
    )
    (checkpoint_directory / "epoch-000097-step-000000388.ckpt").write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="no complete resume checkpoint"):
        launch_track.launch(
            campaign_rel,
            "label",
            0,
            "package",
            0,
            resume=True,
            durable_root=durable_root,
        )
    assert len(commands) == 1

    durable_track = Path(campaign_rel) / "label" / "seed-0" / "package"

    def write_durable_bundle(
        epoch: int, bundle_identity: dict[str, JSONValue] | None
    ) -> Path:
        bundle = durable_root / durable_track / f"epoch-{epoch:06d}"
        bundle.mkdir(parents=True, exist_ok=True)
        checkpoint = bundle / "checkpoint.ckpt"
        checkpoint_payload: dict[str, JSONValue] = {"state_dict": {}}
        if bundle_identity is not None:
            checkpoint_payload["layoutformerpp_identity"] = bundle_identity
        torch.save(checkpoint_payload, checkpoint)
        bundled_manifest = bundle / "launch_manifest.json"
        bundled_manifest.write_bytes(manifest_path.read_bytes())
        bundle_files = {
            path.name: launch_track.sha256(path)
            for path in (checkpoint, bundled_manifest)
        }
        (bundle / "bundle_manifest.json").write_text(
            json.dumps(
                {
                    "checkpoint": (durable_track / bundle.name).as_posix(),
                    "files": bundle_files,
                }
            ),
            encoding="utf-8",
        )
        return checkpoint

    checkpoint = write_durable_bundle(1, checkpoint_identity)
    write_durable_bundle(2, dict(checkpoint_identity, device="other"))
    write_durable_bundle(3, None)
    assert (
        launch_track.launch(
            campaign_rel,
            "label",
            0,
            "package",
            0,
            resume=True,
            durable_root=durable_root,
        )
        == 0
    )
    resumed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert commands[1][-1] == f"--ckpt_path={checkpoint}"
    assert resumed["commands"][0][-1].endswith(checkpoint.name)
    checkpoint.unlink()
    with pytest.raises(RuntimeError, match="no complete resume checkpoint"):
        launch_track.launch(
            campaign_rel,
            "label",
            0,
            "package",
            0,
            resume=True,
            durable_root=durable_root,
        )
    manifest_path.write_text(
        json.dumps(dict(identity, device="other")), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="identity mismatch: device"):
        launch_track.launch(
            campaign_rel,
            "label",
            0,
            "package",
            0,
            resume=True,
            durable_root=durable_root,
        )


def test_resume_candidates_reject_symlink_escapes(tmp_path: Path) -> None:
    track = tmp_path / "campaign" / "label" / "seed-0" / "package"
    checkpoint_directory = track / "lightning" / "package-checkpoints"
    checkpoint_directory.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    checkpoint_directory.symlink_to(outside, target_is_directory=True)
    durable_track = Path("label") / "seed-0" / "package"
    with pytest.raises(RuntimeError, match="symlink"):
        launch_track.resume_checkpoint_candidates(
            checkpoint_directory, "package", None, durable_track
        )

    checkpoint_directory.unlink()
    checkpoint_directory.mkdir()
    outside_checkpoint = outside / "epoch-000001-step-000000001.ckpt"
    outside_checkpoint.write_bytes(b"checkpoint")
    (checkpoint_directory / outside_checkpoint.name).symlink_to(outside_checkpoint)
    assert not launch_track.resume_checkpoint_candidates(
        checkpoint_directory, "package", None, durable_track
    )

    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    (durable_root / "label").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        launch_track.resume_checkpoint_candidates(
            checkpoint_directory, "package", durable_root, durable_track
        )


def _prepare_original_resume_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    str, Path, Path, dict[str, JSONValue], list[tuple[list[str], dict[str, str]]]
]:
    data_root = tmp_path / "data"
    data_root.mkdir()
    environment_file = tmp_path / "data-root.env"
    environment_file.write_text("LAYOUTFORMERPP_DATA_ROOT=/unused\n", encoding="utf-8")
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    package_environment = tmp_path / "runtime"
    campaign_rel = ".cache/original-resume-test"
    track = campaign_root / "label" / "seed-0" / "original"
    checkpoint_directory = track / "vendor-output"
    resume_directory = checkpoint_directory / "resume"
    resume_directory.mkdir(parents=True)

    monkeypatch.setenv("PACKAGE_ENV", str(package_environment))
    monkeypatch.setenv("CAMPAIGN_ROOT", str(campaign_root))
    monkeypatch.setenv("LAYOUTFORMERPP_DATA_ROOT_ENV_FILE", str(environment_file))
    monkeypatch.delenv("LAYOUTFORMERPP_DURABLE_ROOT", raising=False)
    monkeypatch.setattr(
        launch_track.sys, "executable", str(package_environment / "bin" / "python")
    )
    monkeypatch.setattr(
        launch_track, "validate_campaign_root", lambda *_: campaign_root
    )
    monkeypatch.setattr(launch_track, "read_data_root", lambda: data_root)
    monkeypatch.setattr(launch_track, "verify_splits", lambda _: {})
    monkeypatch.setattr(launch_track, "runtime_freeze_sha256", lambda _: "freeze")
    monkeypatch.setattr(launch_track, "require_clean_tree", lambda: None)
    monkeypatch.setattr(
        launch_track, "vendor_commit", lambda: launch_track.EXPECTED_VENDOR_COMMIT
    )
    monkeypatch.setattr(launch_track, "current_commit", lambda: "test-commit")
    monkeypatch.setattr(launch_track, "gpu_info", lambda _: ("test-device", "GPU-test"))
    monkeypatch.setattr(launch_track, "generate", lambda _: None)
    monkeypatch.setattr(
        launch_track, "stage_original_checkpoint", lambda _: {"source": "test"}
    )

    identity: dict[str, JSONValue] = {
        "config_sha256": launch_track.CONFIG_SHA256["label"],
        "source_commit": "test-commit",
        "vendor_source_commit": launch_track.EXPECTED_VENDOR_COMMIT,
        "device": "test-device",
        "checkpoint_dirpath": str(checkpoint_directory.resolve()),
        "campaign_root": str(campaign_root.resolve()),
    }
    recorded_identity = {
        **identity,
        "checkpoint_dirpath": checkpoint_directory.relative_to(
            campaign_root
        ).as_posix(),
        "campaign_root": campaign_rel,
    }
    (track / "launch_manifest.json").write_text(
        json.dumps(recorded_identity), encoding="utf-8"
    )
    commands: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(
        launch_track,
        "run",
        lambda command, _cwd, environment, *_args, **_kwargs: (
            commands.append((command, dict(environment))) or 0
        ),
    )
    return campaign_rel, track, resume_directory, identity, commands


def test_original_resume_selects_newest_complete_matching_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_rel, track, resume_directory, identity, commands = (
        _prepare_original_resume_launch(tmp_path, monkeypatch)
    )
    matching_paths = [
        resume_directory / "epoch-000002-step-000000008.resume.pt",
        resume_directory / "epoch-000005-step-000000020.resume.pt",
    ]
    for path in matching_paths:
        torch.save({"state_dict": {}, "identity": identity}, path)
    torch.save(
        {
            "state_dict": {},
            "identity": dict(identity, device="other"),
        },
        resume_directory / "epoch-000006-step-000000024.resume.pt",
    )
    torch.save(
        {"state_dict": {}},
        resume_directory / "epoch-000007-step-000000028.resume.pt",
    )
    (resume_directory / "epoch-000008-step-000000032.resume.pt").write_bytes(b"partial")

    assert (
        launch_track.launch(campaign_rel, "label", 0, "original", 0, resume=True) == 0
    )

    selected = matching_paths[-1]
    assert commands[0][1]["LAYOUTFORMERPP_RESUME_PATH"] == str(selected)
    manifest = json.loads((track / "launch_manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["resume_checkpoint"]["path"] == selected.relative_to(track).as_posix()
    )


@pytest.mark.parametrize("identity_state", ["mismatched", "identity-less"])
def test_original_resume_refuses_mismatched_or_identityless_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity_state: str
) -> None:
    campaign_rel, _, resume_directory, identity, commands = (
        _prepare_original_resume_launch(tmp_path, monkeypatch)
    )
    checkpoint_identity = (
        dict(identity, device="other") if identity_state == "mismatched" else None
    )
    checkpoint: dict[str, JSONValue] = {"state_dict": {}}
    if checkpoint_identity is not None:
        checkpoint["identity"] = checkpoint_identity
    torch.save(
        checkpoint,
        resume_directory / "epoch-000001-step-000000004.resume.pt",
    )

    with pytest.raises(RuntimeError, match="no complete resume checkpoint"):
        launch_track.launch(campaign_rel, "label", 0, "original", 0, resume=True)
    assert not commands


@pytest.mark.parametrize("escape_kind", ["checkpoint", "directory"])
def test_original_resume_refuses_symlinked_checkpoint_escapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, escape_kind: str
) -> None:
    campaign_rel, _, resume_directory, identity, commands = (
        _prepare_original_resume_launch(tmp_path, monkeypatch)
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    checkpoint_name = "epoch-000001-step-000000004.resume.pt"
    if escape_kind == "directory":
        resume_directory.rmdir()
        resume_directory.symlink_to(outside, target_is_directory=True)
        with pytest.raises(RuntimeError, match="symlink"):
            launch_track.launch(campaign_rel, "label", 0, "original", 0, resume=True)
    else:
        outside_checkpoint = outside / checkpoint_name
        torch.save({"state_dict": {}, "identity": identity}, outside_checkpoint)
        (resume_directory / checkpoint_name).symlink_to(outside_checkpoint)
        with pytest.raises(RuntimeError, match="no complete resume checkpoint"):
            launch_track.launch(campaign_rel, "label", 0, "original", 0, resume=True)
    assert not commands


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
        validate_campaign_root(campaign_rel, REPOSITORY / ".cache" / "other")


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


def _write_checkpoint(path: Path, *, complete: bool = True) -> None:
    if complete:
        with zipfile.ZipFile(path, "w") as checkpoint:
            checkpoint.writestr("weights", b"checkpoint data")
        return

    path.write_bytes(b"partial checkpoint")


def _prune_twice(
    output: Path,
    *,
    epochs: int,
    eval_interval: int,
    observed: dict[Path, tuple[int, int]] | None = None,
) -> list[str]:
    observations = {} if observed is None else observed
    ledger = output.parent / "pruning.jsonl"
    deleted: list[str] = []
    for _ in range(2):
        deleted.extend(
            prune_vendor_checkpoints(
                output,
                epochs=epochs,
                eval_interval=eval_interval,
                ledger_path=ledger,
                observed=observations,
            )
        )
    return deleted


def test_pruning_keeps_evaluation_final_best_and_newest_checkpoints(
    tmp_path: Path,
) -> None:
    output = tmp_path / "vendor-output"
    output.mkdir()
    for epoch in range(5):
        _write_checkpoint(output / f"epoch_{epoch}_checkpoint.pth.tar")
    best_epoch = output / "best_epoch"
    best_epoch.write_text("2\n", encoding="utf-8")
    protected_files = {
        name: name.encode()
        for name in ("checkpoint.pth.tar", "model_best.pth.tar", "training.log")
    }
    for name, contents in protected_files.items():
        (output / name).write_bytes(contents)
    original_bytes = (output / "epoch_0_checkpoint.pth.tar").read_bytes()

    deleted = _prune_twice(output, epochs=5, eval_interval=2)

    assert deleted == ["epoch_0_checkpoint.pth.tar"]
    assert sorted(path.name for path in output.glob("epoch_*_checkpoint.pth.tar")) == [
        "epoch_1_checkpoint.pth.tar",
        "epoch_2_checkpoint.pth.tar",
        "epoch_3_checkpoint.pth.tar",
        "epoch_4_checkpoint.pth.tar",
    ]
    entry = json.loads((tmp_path / "pruning.jsonl").read_text(encoding="utf-8"))
    assert entry == {
        "filename": "epoch_0_checkpoint.pth.tar",
        "sha256": hashlib.sha256(original_bytes).hexdigest(),
        "size_bytes": len(original_bytes),
    }
    assert best_epoch.read_text(encoding="utf-8") == "2\n"
    assert {
        name: (output / name).read_bytes() for name in protected_files
    } == protected_files


def test_pruning_runs_before_best_epoch_exists(tmp_path: Path) -> None:
    output = tmp_path / "vendor-output"
    output.mkdir()
    for epoch in range(6):
        _write_checkpoint(output / f"epoch_{epoch}_checkpoint.pth.tar")

    deleted = _prune_twice(output, epochs=8, eval_interval=2)

    assert deleted == [
        "epoch_0_checkpoint.pth.tar",
        "epoch_2_checkpoint.pth.tar",
        "epoch_4_checkpoint.pth.tar",
    ]
    assert sorted(path.name for path in output.glob("epoch_*_checkpoint.pth.tar")) == [
        "epoch_1_checkpoint.pth.tar",
        "epoch_3_checkpoint.pth.tar",
        "epoch_5_checkpoint.pth.tar",
    ]


def test_pruning_skips_an_unparseable_best_epoch(tmp_path: Path) -> None:
    output = tmp_path / "vendor-output"
    output.mkdir()
    for epoch in range(3):
        _write_checkpoint(output / f"epoch_{epoch}_checkpoint.pth.tar")
    (output / "best_epoch").write_text("", encoding="utf-8")

    assert _prune_twice(output, epochs=8, eval_interval=8) == []
    assert len(list(output.glob("epoch_*_checkpoint.pth.tar"))) == 3


def test_pruning_tracks_best_epoch_changes_during_training(tmp_path: Path) -> None:
    output = tmp_path / "vendor-output"
    output.mkdir()
    for epoch in range(3):
        _write_checkpoint(output / f"epoch_{epoch}_checkpoint.pth.tar")
    best_epoch = output / "best_epoch"
    best_epoch.write_text("1\n", encoding="utf-8")
    observed: dict[Path, tuple[int, int]] = {}

    deleted = _prune_twice(output, epochs=8, eval_interval=8, observed=observed)
    _write_checkpoint(output / "epoch_3_checkpoint.pth.tar")
    best_epoch.write_text("2\n", encoding="utf-8")

    deleted.extend(_prune_twice(output, epochs=8, eval_interval=8, observed=observed))

    assert deleted == ["epoch_0_checkpoint.pth.tar", "epoch_1_checkpoint.pth.tar"]
    assert (output / "epoch_2_checkpoint.pth.tar").is_file()


def test_pruning_error_stops_process_group_and_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python_executable = sys.executable
    marker = tmp_path / "vendor"
    ready = Path(f"{marker}.ready")
    parent_ready = Path(f"{marker}.parent-ready")
    stopped = Path(f"{marker}.stopped")
    child_code = "\n".join(
        (
            "import signal, sys, time",
            "from pathlib import Path",
            "marker = Path(sys.argv[1])",
            "def stop(signum, frame):",
            "    marker.with_suffix('.stopped').write_text('stopped')",
            "    raise SystemExit(0)",
            "signal.signal(signal.SIGTERM, stop)",
            "marker.with_suffix('.ready').write_text('ready')",
            "time.sleep(120)",
        )
    )
    parent_code = "\n".join(
        (
            "import subprocess, sys, time",
            "from pathlib import Path",
            "marker = Path(sys.argv[1])",
            "subprocess.Popen([sys.executable, '-c', sys.argv[2], str(marker)])",
            "while not marker.with_suffix('.ready').exists(): time.sleep(0.01)",
            "marker.with_suffix('.parent-ready').write_text('ready')",
            "time.sleep(120)",
        )
    )
    vendor_script = tmp_path / "vendor.sh"
    vendor_script.write_text(
        shlex.join([python_executable, "-c", parent_code, str(marker), child_code])
        + "\n",
        encoding="utf-8",
    )
    package_environment = tmp_path / "runtime"
    data_root = tmp_path / "data"
    data_root.mkdir()
    environment_file = tmp_path / "data-root.env"
    environment_file.write_text(
        f"LAYOUTFORMERPP_DATA_ROOT={data_root}\n", encoding="utf-8"
    )
    campaign_root = tmp_path / "campaign"
    monkeypatch.setenv("PACKAGE_ENV", str(package_environment))
    monkeypatch.setenv("CAMPAIGN_ROOT", str(campaign_root))
    monkeypatch.setenv("LAYOUTFORMERPP_DATA_ROOT_ENV_FILE", str(environment_file))
    monkeypatch.setattr(
        launch_track.sys, "executable", str(package_environment / "bin" / "python")
    )
    monkeypatch.setattr(
        launch_track,
        "validate_campaign_root",
        lambda campaign_rel, root: campaign_root,
    )
    monkeypatch.setattr(launch_track, "verify_splits", lambda _: {})
    monkeypatch.setattr(launch_track, "runtime_freeze_sha256", lambda _: "freeze")
    monkeypatch.setattr(
        launch_track, "vendor_commit", lambda: launch_track.EXPECTED_VENDOR_COMMIT
    )
    monkeypatch.setattr(launch_track, "require_clean_tree", lambda: None)
    monkeypatch.setattr(launch_track, "current_commit", lambda: "test-commit")
    monkeypatch.setattr(launch_track, "vendor_training_schedule", lambda _: (200, 50))
    monkeypatch.setattr(
        launch_track,
        "build_argv",
        lambda *args, **kwargs: [["bash", str(vendor_script), "train"]],
    )
    monkeypatch.setattr(launch_track, "gpu_info", lambda _: ("NVIDIA A100", "GPU-test"))
    monkeypatch.setattr(launch_track, "write_command_log", lambda *args: None)
    monkeypatch.setattr(
        launch_track,
        "prune_vendor_checkpoints",
        lambda *args, **kwargs: _raise_after_child_starts(parent_ready),
    )
    monkeypatch.setattr(
        launch_track,
        "stream_redacted",
        lambda source, destination, secrets: source.read(),
    )

    monkeypatch.setattr(
        launch_track,
        "build_environment",
        lambda base, **kwargs: dict(base),
    )
    monkeypatch.setattr(
        launch_track, "sha256", lambda _: launch_track.CONFIG_SHA256["label"]
    )

    with pytest.raises(
        launch_track.CheckpointPruningError, match="simulated pruning failure"
    ):
        launch_track.launch(
            ".cache/pruning-test",
            "label",
            0,
            "original",
            0,
            prune_vendor_checkpoints=True,
        )

    track_directory = campaign_root / "label" / "seed-0" / "original"
    manifest_path = track_directory / "launch_manifest.json"

    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert recorded["failure_phase"] == "checkpoint_pruning"
    assert recorded["failure_reason"] == "OSError: simulated pruning failure"
    assert recorded["exit_code"] == 1
    assert parent_ready.is_file()
    assert ready.is_file()
    assert stopped.is_file()


def _raise_after_child_starts(parent_ready: Path) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not parent_ready.is_file():
        time.sleep(0.01)
    if not parent_ready.is_file():
        raise AssertionError("vendor process did not start its child")
    raise OSError("simulated pruning failure")


def test_partial_newest_checkpoint_does_not_authorize_deletion(
    tmp_path: Path,
) -> None:
    output = tmp_path / "vendor-output"
    output.mkdir()
    _write_checkpoint(output / "epoch_0_checkpoint.pth.tar")
    _write_checkpoint(output / "epoch_1_checkpoint.pth.tar")
    _write_checkpoint(output / "epoch_2_checkpoint.pth.tar", complete=False)
    (output / "best_epoch").write_text("0\n", encoding="utf-8")

    deleted = _prune_twice(output, epochs=4, eval_interval=4)

    assert deleted == []
    assert (output / "epoch_1_checkpoint.pth.tar").is_file()
    assert (output / "epoch_2_checkpoint.pth.tar").is_file()


def test_checkpoint_pruning_is_off_by_default_and_a100_original_only() -> None:
    arguments = parse_args([".cache/campaign", "label", "0", "original", "4"])
    assert not arguments.prune_vendor_checkpoints

    enabled_arguments = parse_args(
        [
            ".cache/campaign",
            "label",
            "0",
            "original",
            "4",
            "--prune-vendor-checkpoints",
        ]
    )
    assert enabled_arguments.prune_vendor_checkpoints
    validate_checkpoint_pruning(True, "original", "NVIDIA A100-SXM4-80GB")
    with pytest.raises(RuntimeError, match="original A100 tracks"):
        validate_checkpoint_pruning(True, "package", "NVIDIA A100-SXM4-80GB")
    with pytest.raises(RuntimeError, match="original A100 tracks"):
        validate_checkpoint_pruning(True, "original", "Tesla V100-SXM2-32GB")


def test_vendor_checkpoint_schedule_comes_from_each_training_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vendor_source = tmp_path / "vendor" / "src"
    scripts = vendor_source / "scripts"
    scripts.mkdir(parents=True)
    for script_name in VENDOR_SCRIPTS.values():
        (scripts / script_name).write_text(
            "python main.py \\\n--epoch 7 \\\n--eval_interval 3\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(launch_track, "VENDOR_SOURCE", vendor_source)

    assert {vendor_training_schedule(family) for family in FAMILIES} == {(7, 3)}

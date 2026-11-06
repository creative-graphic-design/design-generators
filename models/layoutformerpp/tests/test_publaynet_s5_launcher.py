from __future__ import annotations

import io
import json
import sys
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts" / "publaynet_s5"
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from launch_track import (  # noqa: E402
    DATA_ROOT_MARKER,
    FAMILIES,
    REPOSITORY,
    VENDOR_SCRIPTS,
    build_argv,
    config_path,
    read_data_root_value,
    stream_redacted,
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

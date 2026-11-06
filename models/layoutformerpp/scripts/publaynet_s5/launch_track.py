"""Validate inputs and launch one original or package PubLayNet training track."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, TextIO, TypeAlias, cast

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY.parent))

from generate_vendor_bin import generate  # noqa: E402
from run_training_parity_campaign import normalized_freeze, sha256  # noqa: E402


REPOSITORY = Path(__file__).resolve().parents[4]
VENDOR_ROOT = REPOSITORY / "vendor" / "ms-layout-generation"
VENDOR_SOURCE = VENDOR_ROOT / "LayoutFormer++" / "src"
DATA_ROOT_MARKER = "<LAYOUTFORMERPP_DATA_ROOT>"
EXPECTED_VENDOR_COMMIT = "1498ff300710b4fc204aece537582d37ca447fc7"
EXPECTED_RUNTIME_FREEZE_SHA256 = (
    "7d05fc1e0d322bf2c6332b0dab36a0c039edae90ad4507468b107c829047c9fb"
)
EXPECTED_SPLITS = {
    "train": (
        311_397,
        "812abbcbd7298901381fa0191a2ca503d82085f8f163b2185ca6056a86da9297",
    ),
    "val": (16_390, "fe38fa40ebbb5405e3053a50c90c53a8a4ad2270de5346cea4187d98c52c8a66"),
    "test": (
        10_998,
        "c74852e1a931e8ee094eefde3a719e1a183db8de0079835d51d5616e63801303",
    ),
}
FAMILIES = (
    "unconditional",
    "completion",
    "refinement",
    "label",
    "label_size",
    "relation",
)
CONFIG_SHA256 = {
    "unconditional": "9e8dffaf2bd4b9e3646aac189bc50b51c52f8a356970615830cc31015f9b76cb",
    "completion": "4b872c27133b6a677cb1f26643a83e2d9fa63e318acd074c68c0213d61ea27f8",
    "refinement": "5b704ade7ae78923b7ee7727f92f7deed7b0832641f6cbbbc959c10541c232a3",
    "label": "0234c836036f6909ce52c157b2b46e17de23c8975f548063d88fa1bf771c1dd0",
    "label_size": "2e69325b78304507ef405f4188d3801a7e3eaac5d07c3ca2f2fe205f315502f2",
    "relation": "99bc2859c5614778aa165277469e8a22234f7eb63b17c4fa1dff3071234c0670",
}
VENDOR_SCRIPTS = {
    "unconditional": "publaynet_ugen.sh",
    "completion": "publaynet_completion.sh",
    "refinement": "publaynet_refinement.sh",
    "label": "publaynet_gen_t.sh",
    "label_size": "publaynet_gen_ts.sh",
    "relation": "publaynet_gen_r.sh",
}
RUNTIME_FREEZE = (
    REPOSITORY / "models" / "layoutformerpp" / "scripts" / "audit_runtime.freeze"
)
JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)


def read_data_root() -> Path:
    environment_file = Path(os.environ["LAYOUTFORMERPP_DATA_ROOT_ENV_FILE"])
    value = read_data_root_value(environment_file)
    return Path(value).expanduser().resolve(strict=True)


def read_data_root_value(environment_file: Path) -> str:
    values: dict[str, str] = {}
    for line in environment_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in (
            "LAYOUTFORMERPP_DATA_ROOT",
            "LAYOUTFORMERPP_PARITY_DATA_ROOT",
        ):
            if key in values:
                raise RuntimeError("data-root environment key is duplicated")
            values[key] = value.strip().strip("\"'")
    if len(values) != 1 or not next(iter(values.values())):
        raise RuntimeError(
            "data-root environment file must set exactly one supported key"
        )
    return next(iter(values.values()))


def count_rows(path: Path) -> int:
    import torch

    values = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(values, list):
        raise RuntimeError("processed split must contain a list of rows")
    return len(values)


def verify_splits(data_root: Path) -> dict[str, dict[str, int | str]]:
    split_root = data_root.joinpath("publaynet", "pre_processed_20_5")
    records: dict[str, dict[str, int | str]] = {}
    for split, (expected_rows, expected_hash) in EXPECTED_SPLITS.items():
        path = split_root / f"{split}.pt"
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"processed {split} split hash check failed")
        try:
            rows = count_rows(path)
        except Exception as error:
            raise RuntimeError(f"processed {split} split row check failed") from error
        if rows != expected_rows:
            raise RuntimeError(f"processed {split} split row check failed")
        records[split] = {"rows": rows, "sha256": expected_hash}
    return records


def config_path(family: str) -> Path:
    return (
        REPOSITORY
        / "models"
        / "layoutformerpp"
        / "configs"
        / "training"
        / f"publaynet_{family}.yaml"
    )


def build_argv(
    family: str,
    seed: int,
    arm: str,
    track_directory: Path,
    package_environment: Path,
) -> list[list[str]]:
    config = config_path(family).relative_to(REPOSITORY).as_posix()
    if arm == "package":
        return [
            [
                str(package_environment / "bin" / "traingen"),
                "fit",
                "--config",
                config,
                f"--seed_everything={seed}",
                "--trainer.devices=1",
                f"--trainer.default_root_dir={track_directory / 'lightning'}",
            ]
        ]

    vendor_script = VENDOR_SOURCE / "scripts" / VENDOR_SCRIPTS[family]
    output = track_directory / "vendor-output"
    base_command = [
        "bash",
        str(vendor_script),
        DATA_ROOT_MARKER,
        str(output),
        "basic",
        "1",
        "s5",
    ]
    return [[*base_command[:2], mode, *base_command[2:]] for mode in ("train", "test")]


def redact_text(value: str, secrets: tuple[str, ...]) -> str:
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        value = value.replace(secret, "<REDACTED>")
    return value


def redact_record(value: JSONValue, secrets: tuple[str, ...]) -> JSONValue:
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, list):
        return [redact_record(item, secrets) for item in value]
    if isinstance(value, dict):
        return {
            redact_text(key, secrets): redact_record(item, secrets)
            for key, item in value.items()
        }
    return value


def write_manifest(
    path: Path, manifest: Mapping[str, JSONValue], secrets: tuple[str, ...]
) -> None:
    path.write_text(
        json.dumps(redact_record(dict(manifest), secrets), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_command_log(
    stream: TextIO, command: list[str], secrets: tuple[str, ...]
) -> None:
    safe_command = safe_argument(command, secrets)
    stream.write(f"COMMAND {shlex.join(safe_command)}\n")
    stream.flush()


def stream_redacted(
    source: BinaryIO,
    destination: BinaryIO,
    secrets: tuple[str, ...],
    *,
    chunk_size: int = 8192,
) -> None:
    encoded = tuple(
        sorted(
            (secret.encode("utf-8") for secret in secrets if secret),
            key=len,
            reverse=True,
        )
    )
    overlap = max((len(secret) for secret in encoded), default=1) - 1
    buffered = b""
    while chunk := source.read(chunk_size):
        buffered += chunk
        for secret in encoded:
            buffered = buffered.replace(secret, b"<REDACTED>")
        if len(buffered) <= overlap:
            continue
        if overlap:
            output, buffered = buffered[:-overlap], buffered[-overlap:]
        else:
            output, buffered = buffered, b""
        destination.write(output)
        destination.flush()
    for secret in encoded:
        buffered = buffered.replace(secret, b"<REDACTED>")
    destination.write(buffered)
    destination.flush()


def runtime_freeze_sha256(package_environment: Path) -> str:
    freeze_path = RUNTIME_FREEZE
    digest = sha256(freeze_path)
    if digest != EXPECTED_RUNTIME_FREEZE_SHA256:
        raise RuntimeError("committed runtime freeze SHA-256 check failed")
    try:
        actual = subprocess.check_output(
            [
                "uv",
                "pip",
                "freeze",
                "--no-config",
                "--python",
                str(package_environment / "bin" / "python"),
            ],
            cwd=REPOSITORY,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        expected = freeze_path.read_text(encoding="utf-8")
        if normalized_freeze(actual) != expected:
            raise RuntimeError("installed runtime differs from the committed freeze")
    except Exception as error:
        raise RuntimeError("runtime freeze check failed") from error
    return digest


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()


def require_clean_tree() -> None:
    for arguments in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(arguments, cwd=REPOSITORY, check=False).returncode:
            raise RuntimeError("tracked source tree must be clean before launch")


def vendor_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(VENDOR_ROOT), "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        text=True,
    ).strip()


def gpu_info(index: int) -> tuple[str, str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        rows = list(csv.reader(output.splitlines(), skipinitialspace=True))
        selected = next(row for row in rows if int(row[0]) == index)
        device_name, uuid, memory_used = selected[1], selected[2], int(selected[3])
        compute_processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as error:
        raise RuntimeError("cannot verify selected GPU") from error
    if memory_used != 0 or any(
        row and row[0].strip() == uuid
        for row in csv.reader(compute_processes.splitlines(), skipinitialspace=True)
    ):
        raise RuntimeError("selected GPU is not idle")
    return device_name, uuid


def safe_argument(command: list[str], secrets: tuple[str, ...]) -> list[str]:
    return [redact_text(argument, secrets) for argument in command]


def build_environment(
    base: Mapping[str, str],
    *,
    seed: int,
    arm: str,
    gpu_index: int,
    data_root: Path,
    environment_file: Path,
    package_environment: Path,
    track_directory: Path,
) -> dict[str, str]:
    environment = dict(base)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "LAYOUTFORMERPP_DATA_ROOT_ENV_FILE": str(environment_file),
            "LAYOUTFORMERPP_TRAIN_SEED": str(seed),
            "PACKAGE_ENV": str(package_environment),
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": "disabled",
        }
    )
    if arm == "package":
        environment.pop("LAYOUTFORMERPP_PARITY_DATA_ROOT", None)
        environment["LAYOUTFORMERPP_DATA_ROOT"] = str(data_root)
    else:
        environment.pop("LAYOUTFORMERPP_DATA_ROOT", None)
        environment.pop("LAYOUTFORMERPP_PARITY_DATA_ROOT", None)
        shim_directory = track_directory / "vendor-bin"
        generate(shim_directory)
        environment["LAYOUTFORMERPP_VENDOR_BOOTSTRAP"] = str(
            SCRIPT_DIRECTORY / "vendor_main_bootstrap.py"
        )
        environment["PATH"] = (
            f"{shim_directory}{os.pathsep}{environment.get('PATH', '')}"
        )
    return environment


def lock_file(stack: ExitStack, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = stack.enter_context(path.open("a", encoding="utf-8"))
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("campaign already holds a required launch lock") from error


def pin_pair_gpu(pair_path: Path, family: str, seed: int, device_name: str) -> None:
    if pair_path.is_file():
        record = json.loads(pair_path.read_text(encoding="utf-8"))
        if record != {"family": family, "seed": seed, "gpu_type": device_name}:
            raise RuntimeError("paired arms must use the same GPU type")
        return
    temporary = pair_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"family": family, "seed": seed, "gpu_type": device_name},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(pair_path)


def run(
    command: list[str], cwd: Path, environment: dict[str, str], secrets: tuple[str, ...]
) -> int:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    if process.stdout is None:
        raise RuntimeError("training output stream unavailable")
    stream_redacted(
        cast(BinaryIO, process.stdout), cast(BinaryIO, sys.stdout.buffer), secrets
    )
    return process.wait()


def launch(campaign_rel: str, family: str, seed: int, arm: str, gpu_index: int) -> int:
    if family not in FAMILIES or arm not in ("original", "package"):
        raise RuntimeError("unsupported family or arm")
    if seed not in range(3) or gpu_index < 0:
        raise RuntimeError("seed or physical GPU index is out of range")
    campaign_path = Path(campaign_rel)
    if (
        campaign_path.is_absolute()
        or ".." in campaign_path.parts
        or not campaign_path.parts
    ):
        raise RuntimeError("campaign path must be repository-relative")

    package_environment = Path(os.environ["PACKAGE_ENV"]).expanduser().resolve()
    if (
        Path(sys.executable).resolve()
        != (package_environment / "bin" / "python").resolve()
    ):
        raise RuntimeError("launcher must run in the configured package environment")
    campaign_root = Path(os.environ["CAMPAIGN_ROOT"]).expanduser().resolve()
    environment_file = (
        Path(os.environ["LAYOUTFORMERPP_DATA_ROOT_ENV_FILE"])
        .expanduser()
        .resolve(strict=True)
    )
    data_root = read_data_root()
    if campaign_root == data_root or data_root in campaign_root.parents:
        raise RuntimeError("campaign outputs must be outside the processed data root")
    track_directory = campaign_root / family / f"seed-{seed}" / arm
    split_records = verify_splits(data_root)

    config = config_path(family)
    expected_config_hash = CONFIG_SHA256[family]
    config_hash = sha256(config)
    if config_hash != expected_config_hash:
        raise RuntimeError("checked-in family configuration hash mismatch")
    freeze_hash = runtime_freeze_sha256(package_environment)
    if arm == "original" and vendor_commit() != EXPECTED_VENDOR_COMMIT:
        raise RuntimeError("pinned vendor source commit mismatch")
    require_clean_tree()
    source_commit = current_commit()

    campaign_root.mkdir(parents=True, exist_ok=True)
    secrets = (
        str(data_root),
        str(REPOSITORY),
        str(campaign_root.resolve()),
        str(package_environment),
        str(environment_file),
    )
    commands = build_argv(family, seed, arm, track_directory, package_environment)

    with ExitStack() as locks:
        lock_file(locks, campaign_root / f".gpu-{gpu_index}.lock")
        device_name, _ = gpu_info(gpu_index)
        pair_directory = track_directory.parent
        pair_directory.mkdir(parents=True, exist_ok=True)
        if track_directory.exists():
            raise RuntimeError("track output already exists")
        lock_file(locks, pair_directory / ".gpu-type.lock")
        pin_pair_gpu(pair_directory / "gpu-type.json", family, seed, device_name)
        track_directory.mkdir()
        manifest_path = track_directory / "launch_manifest.json"
        manifest = {
            "status": "prepared",
            "created_at": datetime.now().astimezone().isoformat(),
            "source_commit": source_commit,
            "vendor_source_commit": EXPECTED_VENDOR_COMMIT
            if arm == "original"
            else None,
            "family": family,
            "training_seed": seed,
            "arm": arm,
            "physical_gpu_index": gpu_index,
            "gpu_type": device_name,
            "config": config.relative_to(REPOSITORY).as_posix(),
            "config_sha256": config_hash,
            "runtime_freeze": RUNTIME_FREEZE.relative_to(REPOSITORY).as_posix(),
            "runtime_freeze_sha256": freeze_hash,
            "inputs": split_records,
            "output": (campaign_path / family / f"seed-{seed}" / arm).as_posix(),
            "commands": [safe_argument(command, secrets) for command in commands],
        }
        write_manifest(manifest_path, manifest, secrets)
        for command in commands:
            write_command_log(sys.stdout, command, secrets)
            environment = build_environment(
                os.environ,
                seed=seed,
                arm=arm,
                gpu_index=gpu_index,
                data_root=data_root,
                environment_file=environment_file,
                package_environment=package_environment,
                track_directory=track_directory,
            )
            cwd = REPOSITORY if arm == "package" else VENDOR_SOURCE
            checked_device_name, _ = gpu_info(gpu_index)
            if checked_device_name != device_name:
                raise RuntimeError("selected GPU type changed before launch")
            exit_code = run(command, cwd, environment, secrets)
            if exit_code:
                (track_directory / "exit-code.txt").write_text(
                    f"{exit_code}\n", encoding="utf-8"
                )
                return exit_code
        (track_directory / "exit-code.txt").write_text("0\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_rel")
    parser.add_argument("family", choices=tuple(FAMILIES))
    parser.add_argument("seed", type=int)
    parser.add_argument("arm", choices=("original", "package"))
    parser.add_argument("gpu_index", type=int)
    args = parser.parse_args()
    try:
        return launch(
            args.campaign_rel, args.family, args.seed, args.arm, args.gpu_index
        )
    except Exception as error:
        secrets = []
        for name in (
            "LAYOUTFORMERPP_DATA_ROOT_ENV_FILE",
            "PACKAGE_ENV",
            "CAMPAIGN_ROOT",
        ):
            if os.environ.get(name):
                secrets.append(os.environ[name])
        try:
            environment_file = Path(os.environ["LAYOUTFORMERPP_DATA_ROOT_ENV_FILE"])
            secrets.append(read_data_root_value(environment_file))
            secrets.append(str(read_data_root()))
        except Exception:
            pass
        safe_error = redact_text(str(error), tuple(secrets))
        print(f"FAIL {type(error).__name__}: {safe_error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

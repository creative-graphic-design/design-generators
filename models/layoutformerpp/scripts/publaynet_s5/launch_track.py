"""Validate inputs and launch one original or package PubLayNet training track."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, TextIO, TypeAlias, cast

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY.parent))

from generate_vendor_bin import generate  # noqa: E402
from run_training_parity_campaign import normalized_freeze, sha256  # noqa: E402
from traingen.checkpointing import (  # noqa: E402
    load_weights_only_checkpoint,
    resolve_durable_root,
)


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
    "unconditional": "7987c2d3ec2a51cae692b4df952a9fe93e59b49107ca4b5628d294375a5a66ff",
    "completion": "e7fd9ab67d0dd782e222dfe33ea2b839eeace6d736090473ca83d3db6a0ba9ca",
    "refinement": "18e270aac3612f0e05e7aeb8b431d9d44693639637f551df638b9ed375344d3c",
    "label": "eb894893bec4db7360834423526dbc6c2dd36888b5f798cdbe4ad4f715176eb7",
    "label_size": "4e3a404d9e2ff5579c65d272eea8e7de3de34ec9e150cb6b596a7782d13483eb",
    "relation": "bec3d4e340dc8ebf2c3ee585ceb6586e8a29250e5f1a9117a1a301bbd2665edd",
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
    try:
        return Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise RuntimeError("configured data root is unavailable") from None


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


def validate_campaign_root(campaign_rel: str, campaign_root: Path) -> Path:
    campaign_path = Path(campaign_rel)
    if (
        campaign_path.is_absolute()
        or ".." in campaign_path.parts
        or not campaign_path.parts
    ):
        raise RuntimeError("campaign path must be repository-relative")
    resolved_root = campaign_root.expanduser().resolve()
    expected_root = (REPOSITORY / campaign_path).resolve()
    if resolved_root != expected_root:
        raise RuntimeError("CAMPAIGN_ROOT does not match CAMPAIGN_REL")
    return resolved_root


def build_argv(
    family: str,
    seed: int,
    arm: str,
    track_directory: Path,
    package_environment: Path,
    *,
    resume_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> list[list[str]]:
    config = config_path(family).relative_to(REPOSITORY).as_posix()
    if arm == "package":
        command = [
            str(package_environment / "bin" / "traingen"),
            "fit",
            "--config",
            config,
            f"--seed_everything={seed}",
            "--trainer.devices=1",
            f"--trainer.default_root_dir={track_directory / 'lightning'}",
        ]
        if checkpoint_path is not None:
            command.append(
                f"--trainer.callbacks.init_args.dirpath={checkpoint_path.resolve()}"
            )
        if resume_path is not None:
            command.append(f"--ckpt_path={resume_path}")
        return [command]

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


def validate_resume_identity(
    manifest: Mapping[str, JSONValue], expected: Mapping[str, JSONValue]
) -> None:
    keys = (
        "config_sha256",
        "source_commit",
        "vendor_source_commit",
        "device",
        "checkpoint_dirpath",
        "campaign_root",
    )
    mismatch = next(
        (key for key in keys if manifest.get(key) != expected.get(key)), None
    )
    if mismatch is not None:
        raise RuntimeError(f"resume checkpoint identity mismatch: {mismatch}")


def _complete_bundle_checkpoint(
    bundle: Path,
    durable_track: Path,
    checkpoint_name: str,
) -> Path | None:
    if bundle.is_symlink() or not bundle.is_dir():
        return None

    manifest_path = bundle / "bundle_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    files = manifest.get("files") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("checkpoint") != (durable_track / bundle.name).as_posix()
        or not isinstance(files, dict)
        or set(files) != {checkpoint_name, "launch_manifest.json"}
    ):
        return None

    for name in (checkpoint_name, "launch_manifest.json"):
        path = bundle / name
        if path.is_symlink() or not path.is_file():
            return None
        try:
            if sha256(path) != files[name]:
                return None
        except OSError:
            return None

    return bundle / checkpoint_name


def _reject_symlinked_directory(path: Path, boundary: Path) -> None:
    try:
        relative = path.relative_to(boundary)
    except ValueError:
        raise RuntimeError(
            "resume checkpoint directory is outside its expected root"
        ) from None

    current = boundary
    for part in (".", *relative.parts):
        current = current if part == "." else current / part
        if current.is_symlink():
            raise RuntimeError("resume checkpoint directory contains a symlink")


def _is_inside(path: Path, directory: Path) -> bool:
    try:
        return not path.is_symlink() and path.resolve().is_relative_to(directory)
    except (OSError, RuntimeError):
        return False


def newest_complete_checkpoint(
    paths: list[Path], identity: Mapping[str, JSONValue], arm: str
) -> Path:
    def epoch_name(path: Path) -> str:
        if path.name in ("checkpoint.ckpt", "checkpoint.pt"):
            return path.parent.name

        return path.name

    for path in sorted(set(paths), key=epoch_name, reverse=True):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            checkpoint = load_weights_only_checkpoint(path)
        except Exception:
            continue

        identity_key = "identity" if arm == "original" else "layoutformerpp_identity"
        checkpoint_identity = checkpoint.get(identity_key)
        if not isinstance(checkpoint_identity, Mapping):
            continue
        try:
            validate_resume_identity(
                cast(Mapping[str, JSONValue], checkpoint_identity), identity
            )
        except RuntimeError:
            continue

        return path
    raise RuntimeError("no complete resume checkpoint is available")


def resume_checkpoint_candidates(
    checkpoint_path: Path,
    arm: str,
    durable_root: Path | None,
    durable_track: Path,
) -> list[Path]:
    if arm == "package":
        pattern = re.compile(r"epoch-\d{6}-step-\d{9}\.ckpt")
        checkpoint_name = "checkpoint.ckpt"
    else:
        pattern = re.compile(r"epoch-\d{6}-step-\d{9}\.resume\.pt")
        checkpoint_path = checkpoint_path / "resume"
        checkpoint_name = "checkpoint.pt"
    local_track_directory = checkpoint_path.parents[1]
    _reject_symlinked_directory(checkpoint_path, local_track_directory)
    resolved_checkpoint_path = checkpoint_path.resolve()
    candidates = (
        [
            path
            for path in checkpoint_path.glob("epoch-*")
            if pattern.fullmatch(path.name)
            and _is_inside(path, resolved_checkpoint_path)
        ]
        if arm == "original"
        else []
    )
    if durable_root is None:
        return candidates

    durable_track_directory = durable_root / durable_track
    _reject_symlinked_directory(durable_track_directory, durable_root)
    resolved_durable_track = durable_track_directory.resolve()
    if durable_track_directory.is_dir():
        candidates.extend(
            checkpoint
            for bundle in durable_track_directory.iterdir()
            if re.fullmatch(r"epoch-\d{6}", bundle.name)
            if (
                checkpoint := _complete_bundle_checkpoint(
                    bundle, durable_track, checkpoint_name
                )
            )
            is not None
            and _is_inside(checkpoint, resolved_durable_track)
        )

    return candidates


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
    gpu_uuid: str,
    data_root: Path,
    environment_file: Path,
    package_environment: Path,
    track_directory: Path,
) -> dict[str, str]:
    environment = dict(base)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
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
        environment["LAYOUTFORMERPP_VENDOR_CACHE"] = str(
            track_directory / "vendor-cache"
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


def pin_pair_gpu_exclusively(
    pair_path: Path, family: str, seed: int, device_name: str
) -> None:
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    with (pair_path.parent / ".gpu-type.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("campaign is pinning this GPU pair") from error
        try:
            pin_pair_gpu(pair_path, family, seed, device_name)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def stage_original_checkpoint(vendor_output: Path) -> dict[str, JSONValue]:
    try:
        best_epoch = int((vendor_output / "best_epoch").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError(
            "vendor best-epoch checkpoint selection is unavailable"
        ) from None
    source = vendor_output / f"epoch_{best_epoch}_checkpoint.pth.tar"
    staged = vendor_output / "s5_checkpoint.pth.tar"
    if not source.is_file():
        raise RuntimeError("selected vendor checkpoint is unavailable")
    source_hash = sha256(source)
    shutil.copy2(source, staged)
    if sha256(staged) != source_hash:
        raise RuntimeError("staged vendor checkpoint hash mismatch")
    return {
        "source": source.name,
        "staged_as": staged.name,
        "sha256": source_hash,
    }


def vendor_training_schedule(family: str) -> tuple[int, int]:
    script = VENDOR_SOURCE / "scripts" / VENDOR_SCRIPTS[family]
    tokens = shlex.split(
        script.read_text(encoding="utf-8").replace("\\\n", " "), comments=True
    )
    values: list[int] = []
    for option in ("--epoch", "--eval_interval"):
        matches = [
            tokens[index + 1]
            for index, token in enumerate(tokens[:-1])
            if token == option
        ]
        if len(matches) != 1:
            raise RuntimeError("vendor training schedule arguments are unavailable")
        try:
            value = int(matches[0])
        except ValueError:
            raise RuntimeError(
                "vendor training schedule arguments are invalid"
            ) from None
        if value < 1:
            raise RuntimeError("vendor training schedule arguments are invalid")
        values.append(value)
    return values[0], values[1]


def validate_checkpoint_pruning(enabled: bool, arm: str, device_name: str) -> None:
    if not enabled:
        return
    if arm != "original" or "a100" not in device_name.casefold():
        raise RuntimeError("checkpoint pruning is limited to original A100 tracks")


def read_best_epoch(vendor_output: Path) -> tuple[bool, int | None]:
    try:
        value = int((vendor_output / "best_epoch").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, None
    except (OSError, ValueError):
        return True, None
    return True, value


def complete_checkpoint(path: Path, observed: dict[Path, tuple[int, int]]) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    signature = (stat.st_size, stat.st_mtime_ns)
    if stat.st_size == 0 or observed.get(path) != signature:
        observed[path] = signature
        return False
    try:
        with zipfile.ZipFile(path) as checkpoint:
            return bool(checkpoint.infolist())
    except (OSError, EOFError, zipfile.BadZipFile):
        return False


def prune_vendor_checkpoints(
    vendor_output: Path,
    *,
    epochs: int,
    eval_interval: int,
    ledger_path: Path,
    observed: dict[Path, tuple[int, int]],
) -> list[str]:
    if epochs < 1 or eval_interval < 1:
        raise ValueError("vendor epoch count and evaluation interval must be positive")

    checkpoint_epochs: dict[int, Path] = {}
    for path in vendor_output.glob("epoch_*_checkpoint.pth.tar"):
        match = re.fullmatch(r"epoch_(\d+)_checkpoint\.pth\.tar", path.name)
        if match is not None:
            checkpoint_epochs[int(match.group(1))] = path
    if not checkpoint_epochs or any(epoch >= epochs for epoch in checkpoint_epochs):
        return []

    has_best_epoch, selected_best = read_best_epoch(vendor_output)
    if has_best_epoch and (selected_best is None or not 0 <= selected_best < epochs):
        return []
    newest = max(checkpoint_epochs)
    keep = {
        newest,
        epochs - 1,
        *(epoch for epoch in range(epochs) if (epoch + 1) % eval_interval == 0),
    }
    if selected_best is not None:
        keep.add(selected_best)
    deleted: list[str] = []
    for epoch, path in sorted(checkpoint_epochs.items()):
        successor = checkpoint_epochs.get(epoch + 1)
        if epoch in keep or successor is None:
            continue
        candidate_complete = complete_checkpoint(path, observed)
        successor_complete = complete_checkpoint(successor, observed)
        if not candidate_complete or not successor_complete:
            continue
        has_best_epoch, current_best = read_best_epoch(vendor_output)
        if has_best_epoch and (current_best is None or not 0 <= current_best < epochs):
            return deleted
        if current_best == epoch:
            continue
        try:
            before = path.stat()
            digest = sha256(path)
            after = path.stat()
        except OSError:
            continue
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            continue
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8") as ledger:
            ledger.write(
                json.dumps(
                    {
                        "filename": path.name,
                        "size_bytes": before.st_size,
                        "sha256": digest,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            ledger.flush()
            os.fsync(ledger.fileno())
        path.unlink()
        observed.pop(path, None)
        deleted.append(path.name)
    return deleted


def record_failure(
    track_directory: Path,
    manifest_path: Path,
    manifest: dict[str, JSONValue],
    secrets: tuple[str, ...],
    *,
    training_started: bool,
    phase: str,
    exit_code: int,
    reason: str | None = None,
) -> None:
    manifest["status"] = "failed" if training_started else "failed_before_training"
    manifest["failure_phase"] = phase
    manifest["exit_code"] = exit_code
    if reason is not None:
        manifest["failure_reason"] = reason
    write_manifest(manifest_path, manifest, secrets)
    (track_directory / "exit-code.txt").write_text(f"{exit_code}\n", encoding="utf-8")


class CheckpointPruningError(RuntimeError):
    def __init__(self, error: Exception, secrets: tuple[str, ...]) -> None:
        self.reason = redact_text(f"{type(error).__name__}: {error}", secrets)
        super().__init__(f"vendor checkpoint pruning failed: {self.reason}")


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    secrets: tuple[str, ...],
    on_start: Callable[[], None],
    *,
    checkpoint_pruning: tuple[Path, int, int, Path] | None = None,
) -> int:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=checkpoint_pruning is not None,
    )
    on_start()
    if process.stdout is None:
        raise RuntimeError("training output stream unavailable")
    pruning_stop = threading.Event()
    pruning_errors: list[Exception] = []
    observed: dict[Path, tuple[int, int]] = {}

    def prune() -> None:
        assert checkpoint_pruning is not None
        output, epochs, eval_interval, ledger = checkpoint_pruning
        prune_vendor_checkpoints(
            output,
            epochs=epochs,
            eval_interval=eval_interval,
            ledger_path=ledger,
            observed=observed,
        )

    def monitor() -> None:
        while not pruning_stop.is_set():
            try:
                prune()
            except Exception as error:
                pruning_errors.append(error)
                pruning_stop.set()
                stop_process_group(process)
                return
            if pruning_stop.wait(10):
                return

    pruner = None
    if checkpoint_pruning is not None:
        pruner = threading.Thread(
            target=monitor, name="vendor-checkpoint-pruner", daemon=True
        )
        pruner.start()
    try:
        stream_redacted(
            cast(BinaryIO, process.stdout), cast(BinaryIO, sys.stdout.buffer), secrets
        )
        exit_code = process.wait()
    finally:
        pruning_stop.set()
        if pruner is not None:
            pruner.join()
    if pruning_errors:
        error = pruning_errors[0]
        raise CheckpointPruningError(error, secrets) from error
    if checkpoint_pruning is not None and exit_code == 0:
        try:
            prune()
            prune()
        except Exception as error:
            raise CheckpointPruningError(error, secrets) from error
    return exit_code


def launch(
    campaign_rel: str,
    family: str,
    seed: int,
    arm: str,
    gpu_index: int,
    *,
    prune_vendor_checkpoints: bool = False,
    resume: bool = False,
    durable_root: Path | None = None,
) -> int:
    if family not in FAMILIES or arm not in ("original", "package"):
        raise RuntimeError("unsupported family or arm")
    if prune_vendor_checkpoints and arm != "original":
        raise RuntimeError("checkpoint pruning is limited to original A100 tracks")
    if seed not in range(3) or gpu_index < 0:
        raise RuntimeError("seed or physical GPU index is out of range")
    campaign_path = Path(campaign_rel)

    package_environment = Path(os.environ["PACKAGE_ENV"]).expanduser().resolve()
    if (
        Path(sys.executable).resolve()
        != (package_environment / "bin" / "python").resolve()
    ):
        raise RuntimeError("launcher must run in the configured package environment")
    campaign_root = validate_campaign_root(
        campaign_rel, Path(os.environ["CAMPAIGN_ROOT"])
    )
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
    durable_value = (
        str(durable_root)
        if durable_root is not None
        else os.environ.get("LAYOUTFORMERPP_DURABLE_ROOT")
    )
    durable_path = (
        resolve_durable_root(durable_value, env_var="LAYOUTFORMERPP_DURABLE_ROOT")
        if durable_value
        else None
    )
    if resume and arm == "package" and durable_path is None:
        raise RuntimeError("package-arm --resume requires --durable-root")

    campaign_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = track_directory / "vendor-output"
    if arm == "package":
        checkpoint_path = track_directory / "lightning" / "package-checkpoints"
    durable_track = campaign_path / family / f"seed-{seed}" / arm
    resume_path: Path | None = None
    identity: dict[str, JSONValue] = {
        "config_sha256": config_hash,
        "source_commit": source_commit,
        "vendor_source_commit": EXPECTED_VENDOR_COMMIT if arm == "original" else None,
        "device": "",
        "checkpoint_dirpath": str(checkpoint_path.resolve()),
        "campaign_root": str(campaign_root.resolve()),
    }
    secrets = (
        str(data_root),
        str(REPOSITORY),
        str(campaign_root.resolve()),
        str(package_environment),
        str(environment_file),
        *((str(durable_path),) if durable_path is not None else ()),
    )
    schedule = vendor_training_schedule(family) if prune_vendor_checkpoints else None

    with ExitStack() as locks:
        lock_file(locks, campaign_root / f".gpu-{gpu_index}.lock")
        device_name, gpu_uuid = gpu_info(gpu_index)
        identity["device"] = device_name
        recorded_identity = {
            **identity,
            "checkpoint_dirpath": checkpoint_path.relative_to(campaign_root).as_posix(),
            "campaign_root": campaign_rel,
        }
        validate_checkpoint_pruning(prune_vendor_checkpoints, arm, device_name)
        pair_directory = track_directory.parent
        pair_directory.mkdir(parents=True, exist_ok=True)
        if resume:
            manifest_path = track_directory / "launch_manifest.json"
            if not track_directory.is_dir() or not manifest_path.is_file():
                raise RuntimeError("resume launch manifest is unavailable")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise RuntimeError("resume launch manifest is invalid") from None
            if not isinstance(manifest, dict):
                raise RuntimeError("resume launch manifest is invalid")
            validate_resume_identity(manifest, recorded_identity)
            resume_path = newest_complete_checkpoint(
                resume_checkpoint_candidates(
                    checkpoint_path,
                    arm,
                    durable_path,
                    durable_track,
                ),
                identity,
                arm,
            )
            checkpoint_root = track_directory
            if durable_path is not None:
                durable_track_root = durable_path / durable_track
                if resume_path.is_relative_to(durable_track_root):
                    checkpoint_root = durable_track_root
            manifest["resume_checkpoint"] = {
                "path": resume_path.relative_to(checkpoint_root).as_posix(),
                "sha256": sha256(resume_path),
            }
        else:
            if track_directory.exists():
                raise RuntimeError("track output already exists")
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
                "vendor_checkpoint_pruning": {
                    "enabled": prune_vendor_checkpoints,
                    "epoch_count": schedule[0] if schedule is not None else None,
                    "eval_interval": schedule[1] if schedule is not None else None,
                },
                "physical_gpu_index": gpu_index,
                "gpu_type": device_name,
                "config": config.relative_to(REPOSITORY).as_posix(),
                "config_sha256": config_hash,
                "runtime_freeze": RUNTIME_FREEZE.relative_to(REPOSITORY).as_posix(),
                "runtime_freeze_sha256": freeze_hash,
                "inputs": split_records,
                "output": (campaign_path / family / f"seed-{seed}" / arm).as_posix(),
            }
            manifest.update(recorded_identity)

        identity_manifest = track_directory / "checkpoint_identity.json"
        if durable_path is not None:
            identity_manifest.write_text(
                json.dumps(
                    {
                        key: value
                        for key, value in identity.items()
                        if key != "campaign_root"
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        commands = build_argv(
            family,
            seed,
            arm,
            track_directory,
            package_environment,
            resume_path=resume_path if resume and arm == "package" else None,
            checkpoint_path=checkpoint_path if arm == "package" else None,
        )
        manifest["commands"] = [safe_argument(command, secrets) for command in commands]
        pin_pair_gpu_exclusively(
            pair_directory / "gpu-type.json", family, seed, device_name
        )
        write_manifest(manifest_path, manifest, secrets)
        training_started = False
        phase = "training"
        try:
            for command in commands:
                if arm == "original" and command[2] == "test":
                    phase = "checkpoint_selection"
                    manifest["selected_checkpoint"] = stage_original_checkpoint(
                        track_directory / "vendor-output"
                    )
                    write_manifest(manifest_path, manifest, secrets)
                    phase = "test"
                write_command_log(sys.stdout, command, secrets)
                environment = build_environment(
                    os.environ,
                    seed=seed,
                    arm=arm,
                    gpu_index=gpu_index,
                    gpu_uuid=gpu_uuid,
                    data_root=data_root,
                    environment_file=environment_file,
                    package_environment=package_environment,
                    track_directory=track_directory,
                )
                is_training = arm == "package" or command[2] == "train"
                for name in (
                    "LAYOUTFORMERPP_CHECKPOINT_IDENTITY",
                    "LAYOUTFORMERPP_LAUNCH_MANIFEST",
                    "LAYOUTFORMERPP_DURABLE_ROOT",
                    "LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE",
                    "LAYOUTFORMERPP_RESUME_PATH",
                ):
                    environment.pop(name, None)
                if is_training and (arm == "original" or durable_path is not None):
                    environment["LAYOUTFORMERPP_CHECKPOINT_IDENTITY"] = json.dumps(
                        identity, sort_keys=True
                    )
                    if resume_path is not None:
                        environment["LAYOUTFORMERPP_RESUME_PATH"] = str(resume_path)
                if durable_path is not None:
                    environment.update(
                        {
                            "LAYOUTFORMERPP_LAUNCH_MANIFEST": str(identity_manifest),
                            "LAYOUTFORMERPP_DURABLE_ROOT": str(durable_path),
                            "LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE": durable_track.as_posix(),
                        }
                    )
                cwd = REPOSITORY if arm == "package" else VENDOR_SOURCE
                checked_device_name, checked_gpu_uuid = gpu_info(gpu_index)
                if checked_device_name != device_name or checked_gpu_uuid != gpu_uuid:
                    raise RuntimeError("selected GPU changed before launch")
                manifest["status"] = "running"
                write_manifest(manifest_path, manifest, secrets)

                def mark_started() -> None:
                    nonlocal training_started
                    training_started = True

                checkpoint_pruning = None
                if schedule is not None and command[2] == "train":
                    checkpoint_pruning = (
                        track_directory / "vendor-output",
                        schedule[0],
                        schedule[1],
                        track_directory / "vendor-checkpoint-pruning.jsonl",
                    )
                exit_code = run(
                    command,
                    cwd,
                    environment,
                    secrets,
                    mark_started,
                    checkpoint_pruning=checkpoint_pruning,
                )
                if exit_code:
                    record_failure(
                        track_directory,
                        manifest_path,
                        manifest,
                        secrets,
                        training_started=training_started,
                        phase=phase,
                        exit_code=exit_code,
                    )
                    return exit_code
                phase = "checkpoint_selection" if arm == "original" else "training"
            manifest["status"] = "complete"
            manifest["exit_code"] = 0
            write_manifest(manifest_path, manifest, secrets)
            (track_directory / "exit-code.txt").write_text("0\n", encoding="utf-8")
        except Exception as error:
            failure_phase = (
                "checkpoint_pruning"
                if isinstance(error, CheckpointPruningError)
                else phase
            )
            failure_reason = (
                error.reason if isinstance(error, CheckpointPruningError) else None
            )
            record_failure(
                track_directory,
                manifest_path,
                manifest,
                secrets,
                training_started=training_started,
                phase=failure_phase,
                exit_code=1,
                reason=failure_reason,
            )
            raise
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_rel")
    parser.add_argument("family", choices=tuple(FAMILIES))
    parser.add_argument("seed", type=int)
    parser.add_argument("arm", choices=("original", "package"))
    parser.add_argument("gpu_index", type=int)
    parser.add_argument(
        "--prune-vendor-checkpoints",
        action="store_true",
        help="prune old vendor checkpoints on an A100 original track",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a track; package arm requires --durable-root",
    )
    parser.add_argument("--durable-root", type=Path)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        return launch(
            args.campaign_rel,
            args.family,
            args.seed,
            args.arm,
            args.gpu_index,
            prune_vendor_checkpoints=args.prune_vendor_checkpoints,
            resume=args.resume,
            durable_root=args.durable_root,
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

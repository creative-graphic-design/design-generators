"""Run and compare the PubLayNet production loss-envelope pre-check."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

UPDATES = 1500
SEEDS = (0, 1, 2)
LOSS_LINE = re.compile(r"\[\d+/\d+\]\[(\d+)/(\d+)\]\s+Loss:\s+([\d.eE+-]+)")
RUNNER_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = RUNNER_DIRECTORY.parents[3]
OUTPUT_ROOT = REPOSITORY / ".cache" / "layoutformerpp" / "s5" / "prechecks"


def batch_sha256(value: object) -> str:
    """Digest one production batch without writing sample contents."""
    import torch

    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(b"[")
            for child in item:
                update(child)
            digest.update(b"]")
        elif isinstance(item, bytes):
            digest.update(item)
        elif isinstance(item, str):
            digest.update(item.encode("utf-8"))
        else:
            digest.update(repr(item).encode("utf-8"))
        digest.update(b"\0")

    update(value)
    return digest.hexdigest()


def _sitecustomize_source() -> str:
    return """import json
import os
from torch.utils.data import DataLoader
from run_publaynet_loss_envelope import batch_sha256

_original_iter = DataLoader.__iter__
_trace_path = os.environ.get("LAYOUTFORMERPP_BATCH_TRACE")
_limit = int(os.environ.get("LAYOUTFORMERPP_BATCH_LIMIT", "1500"))
_started = False

def _recording_iter(loader):
    global _started
    source = _original_iter(loader)
    if not _trace_path or _started:
        yield from source
        return
    _started = True
    count = 0
    with open(_trace_path, "x", encoding="utf-8") as stream:
        for batch in source:
            if count == _limit:
                raise SystemExit(0)
            count += 1
            stream.write(json.dumps({"update": count, "batch_sha256": batch_sha256(batch)}) + "\\n")
            yield batch

DataLoader.__iter__ = _recording_iter
"""


def _original_losses(lines: Sequence[str]) -> list[float]:
    values: list[float] = []
    for line in lines:
        match = LOSS_LINE.search(line)
        if match is None:
            continue
        value = float(match.group(3))
        values.append(value)
    return values


def _package_losses(path: Path) -> list[float]:
    values: list[float] = []
    for row in csv.DictReader(path.open(encoding="utf-8", newline="")):
        raw_value = row.get("train_loss_step")
        if raw_value is None or raw_value == "":
            continue
        value = float(raw_value)
        values.append(value)
    return values


def _record_native_output(
    *,
    system: str,
    seed: int,
    directory: Path,
    loss_values: Sequence[float],
) -> None:
    batch_rows = [
        json.loads(line)
        for line in (directory / "batches.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    records: list[dict[str, object]] = [
        {"record": "metadata", "system": system, "seed": seed}
    ]
    for update, (loss, row) in enumerate(zip(loss_values, batch_rows, strict=True), 1):
        records.append(
            {
                "record": "update",
                "update": update,
                "objective": loss,
                "batch_sha256": row["batch_sha256"],
            }
        )
    with (directory / "updates.jsonl").open("x", encoding="utf-8") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in records)


def _compare(original: Path, package: Path, result_path: Path) -> dict[str, object]:
    comparator = (
        REPOSITORY
        / "models"
        / "canvas-vae"
        / "scripts"
        / "compare_crello_loss_envelope.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(comparator),
            "--original-dir",
            str(original),
            "--package-dir",
            str(package),
            "--output",
            str(result_path),
            "--seeds",
            *map(str, SEEDS),
            "--updates",
            str(UPDATES),
            "--logger-interval",
            "1",
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result_path.read_text(encoding="utf-8"))


def _run_original(
    campaign: ModuleType,
    *,
    seed: int,
    directory: Path,
    environment: dict[str, str],
) -> None:
    source = campaign.VENDOR_SOURCE / "scripts" / campaign.VENDOR_SCRIPTS["label"]
    script = source.read_text(encoding="utf-8")
    if script.count("--load_vocab") != 1:
        raise RuntimeError(
            "original production script no longer has the expected logging anchor"
        )
    script = script.replace("--load_vocab", "--load_vocab --train_log_step 1", 1)
    script_copy = directory / source.name
    script_copy.write_text(script, encoding="utf-8")
    output = directory / "vendor-output"
    output.mkdir()
    command = campaign.build_argv(
        "label", seed, "original", directory, Path(environment["PACKAGE_ENV"])
    )[0]
    command[1] = str(script_copy)
    child_environment = dict(environment)
    child_environment["LAYOUTFORMERPP_BATCH_TRACE"] = str(directory / "batches.jsonl")
    completed = subprocess.run(
        command,
        cwd=campaign.VENDOR_SOURCE,
        env=child_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    losses = _original_losses(completed.stdout.splitlines())
    if completed.returncode != 0 or len(losses) != UPDATES:
        raise RuntimeError(
            f"original production entry stopped before {UPDATES} updates"
        )
    _record_native_output(
        system="original", seed=seed, directory=directory, loss_values=losses
    )


def _run_package(
    campaign: ModuleType,
    *,
    seed: int,
    directory: Path,
    environment: dict[str, str],
) -> None:
    command = campaign.build_argv(
        "label", seed, "package", directory, Path(environment["PACKAGE_ENV"])
    )[0]
    command.extend(
        [
            f"--trainer.max_steps={UPDATES}",
            "--trainer.log_every_n_steps=1",
            "--trainer.enable_progress_bar=false",
        ]
    )
    child_environment = dict(environment)
    child_environment["LAYOUTFORMERPP_BATCH_TRACE"] = str(directory / "batches.jsonl")
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        env=child_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"package production entry failed for seed {seed}")
    metrics = sorted((directory / "lightning").rglob("metrics.csv"))
    if len(metrics) != 1:
        raise RuntimeError(
            f"package seed {seed}: expected one native Lightning metrics file"
        )
    losses = _package_losses(metrics[0])
    _record_native_output(
        system="package", seed=seed, directory=directory, loss_values=losses
    )


def _run(output: Path, gpu_index: int) -> dict[str, object]:
    sys.path.insert(
        0, str(REPOSITORY / "models" / "layoutformerpp" / "scripts" / "publaynet_s5")
    )
    import launch_track as campaign

    environment_file = Path(os.environ["LAYOUTFORMERPP_DATA_ROOT_ENV_FILE"])
    package_environment = Path(os.environ["PACKAGE_ENV"])
    data_root = campaign.read_data_root()
    device_name, _ = campaign.gpu_info(gpu_index)
    root = output / "runs"
    root.mkdir(parents=True, exist_ok=True)
    sitecustomize = root / "sitecustomize.py"
    sitecustomize.write_text(_sitecustomize_source(), encoding="utf-8")
    for seed in SEEDS:
        for system, run in (("original", _run_original), ("package", _run_package)):
            directory = root / system / f"seed-{seed}"
            directory.mkdir(parents=True)
            # The training record persists the selector, so keep only the index.
            child_environment = campaign.build_environment(
                os.environ,
                seed=seed,
                arm=system,
                gpu_index=gpu_index,
                gpu_uuid=str(gpu_index),
                data_root=data_root,
                environment_file=environment_file,
                package_environment=package_environment,
                track_directory=directory,
            )
            child_environment["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    (
                        str(root),
                        str(RUNNER_DIRECTORY),
                        child_environment.get("PYTHONPATH", ""),
                    ),
                )
            )
            child_environment["LAYOUTFORMERPP_BATCH_LIMIT"] = str(UPDATES)
            run(campaign, seed=seed, directory=directory, environment=child_environment)
    result_path = output / "result.json"
    result = _compare(root / "original", root / "package", result_path)
    result.update(
        gpu_index=gpu_index,
        gpu_type=device_name,
        source_commit=campaign.current_commit(),
        vendor_source_commit=campaign.vendor_commit(),
        config="models/layoutformerpp/configs/training/publaynet_label.yaml",
        config_sha256=campaign.CONFIG_SHA256["label"],
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_entrypoint="vendor PubLayNet label script -> main.py; train_log_step=1",
        package_entrypoint="traingen fit; max_steps=1500; log_every_n_steps=1",
        initialization="independent production initialization and optimizer per matched seed",
        batch_mapping="one loader batch per completed optimizer update; exact digests compared",
        objective="native training loss; original Loss output and package train_loss_step",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT / "loss-envelope"
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (
        not output.is_relative_to(OUTPUT_ROOT.resolve())
        or output.exists()
        or shutil.disk_usage(REPOSITORY).free < 1_500_000_000_000
    ):
        raise SystemExit("output must be a new directory under the S5 pre-check cache")
    output.mkdir(parents=True)
    result = _run(output, args.gpu_index)
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

"""Write the framework and CUDA stack used by one training run."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_identity() -> tuple[str | None, str | None]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible == "" or visible == "-1":
        return None, None

    command = [
        "nvidia-smi",
        "--query-gpu=driver_version,name",
        "--format=csv,noheader",
    ]
    if visible:
        command.extend(("--id", visible.split(",", maxsplit=1)[0]))

    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 2:
        raise RuntimeError("run metadata requires exactly one visible GPU")

    return rows[0][0], rows[0][1]


def _framework_stack(framework: str) -> dict[str, str | bool | None]:
    versions: dict[str, str | bool | None] = {
        "framework": framework,
        "cuda_runtime_package": _package_version("nvidia-cuda-runtime-cu12"),
        "cudnn_package": _package_version("nvidia-cudnn-cu12"),
        "nccl_package": _package_version("nvidia-nccl-cu12"),
    }

    if framework == "torch":
        import torch
        import torch.cuda.nccl

        available = torch.cuda.is_available()
        driver, gpu = _gpu_identity() if available else (None, None)
        nccl = torch.cuda.nccl.version() if available else None
        versions.update(
            {
                "framework_version": torch.__version__,
                "cuda_build": torch.version.cuda,
                "cuda_available": available,
                "driver": driver,
                "cudnn_runtime": str(torch.backends.cudnn.version())
                if available
                else None,
                "nccl_runtime": ".".join(str(value) for value in nccl)
                if isinstance(nccl, tuple)
                else str(nccl)
                if nccl is not None
                else None,
                "gpu": torch.cuda.get_device_name(0) if available else gpu,
            }
        )
    elif framework == "tensorflow":
        import tensorflow as tf

        build = tf.sysconfig.get_build_info()
        available = bool(tf.config.list_physical_devices("GPU"))
        driver, gpu = _gpu_identity() if available else (None, None)
        versions.update(
            {
                "framework_version": tf.__version__,
                "cuda_build": build.get("cuda_version"),
                "cudnn_build": build.get("cudnn_version"),
                "nccl_runtime": versions["nccl_package"],
                "cuda_available": available,
                "driver": driver,
                "gpu": gpu,
            }
        )
    else:
        raise ValueError("framework must be 'torch' or 'tensorflow'")

    return versions


def write_run_metadata(path: str | Path, framework: str) -> None:
    """Create one exclusive JSON record with the active stack and full freeze."""
    freeze_path = os.environ.get("CANVAS_VAE_PIP_FREEZE_FILE")
    if freeze_path:
        pip_freeze = Path(freeze_path).read_text(encoding="utf-8").splitlines()
    else:
        result = subprocess.run(
            ["uv", "pip", "freeze", "--python", sys.executable],
            check=True,
            capture_output=True,
            text=True,
        )
        pip_freeze = result.stdout.splitlines()

    metadata = {
        "schema_version": 1,
        "stack": _framework_stack(framework),
        "pip_freeze": pip_freeze,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as output:
            json.dump(metadata, output, indent=2, sort_keys=True)
            output.write("\n")
    except FileExistsError:
        if json.loads(destination.read_text(encoding="utf-8")) != metadata:
            raise RuntimeError("existing run metadata describes a different stack")


def main() -> None:
    """Write metadata for one command-line training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "tensorflow"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_run_metadata(arguments.output, arguments.framework)


if __name__ == "__main__":
    main()

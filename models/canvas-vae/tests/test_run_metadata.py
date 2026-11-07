import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from canvas_vae import run_metadata


def test_run_metadata_records_stack_and_complete_pip_freeze(tmp_path, monkeypatch):
    freeze = tmp_path / "pip-freeze.txt"
    freeze.write_text("tensorflow==2.15.1\ntraingen==0.1.0\n", encoding="utf-8")
    output = tmp_path / "run" / "run-metadata.json"
    monkeypatch.setenv("CANVAS_VAE_PIP_FREEZE_FILE", str(freeze))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    nccl_module = ModuleType("torch.cuda.nccl")
    setattr(nccl_module, "version", lambda: (2, 29, 3))
    cuda_module = ModuleType("torch.cuda")
    setattr(cuda_module, "__path__", [])
    setattr(cuda_module, "nccl", nccl_module)
    setattr(cuda_module, "is_available", lambda: True)
    setattr(
        cuda_module,
        "get_device_name",
        lambda _index: "Tesla V100-SXM2-32GB",
    )
    monkeypatch.setitem(sys.modules, "torch.cuda", cuda_module)
    monkeypatch.setitem(sys.modules, "torch.cuda.nccl", nccl_module)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.13.0+cu126",
            version=SimpleNamespace(cuda="12.6"),
            cuda=cuda_module,
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 91002)),
        ),
    )

    def run(command, **_kwargs):
        if command[0] == "nvidia-smi":
            return SimpleNamespace(stdout="575.57.08, Tesla V100-SXM2-32GB\n")

        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(
        run_metadata.subprocess,
        "run",
        run,
    )

    run_metadata.write_run_metadata(output, "torch")

    metadata = json.loads(output.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["pip_freeze"] == ["tensorflow==2.15.1", "traingen==0.1.0"]
    assert metadata["stack"]["framework_version"] == "2.13.0+cu126"
    assert metadata["stack"]["cuda_build"] == "12.6"
    assert metadata["stack"]["cudnn_runtime"] == "91002"
    assert metadata["stack"]["nccl_runtime"] == "2.29.3"
    assert metadata["stack"]["driver"] == "575.57.08"
    assert metadata["stack"]["gpu"] == "Tesla V100-SXM2-32GB"


def test_tensorflow_run_metadata_records_build_versions(tmp_path, monkeypatch):
    freeze = tmp_path / "pip-freeze.txt"
    freeze.write_text("tensorflow==2.15.1\n", encoding="utf-8")
    output = tmp_path / "run-metadata.json"
    monkeypatch.setenv("CANVAS_VAE_PIP_FREEZE_FILE", str(freeze))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setattr(
        run_metadata,
        "_package_version",
        lambda package: "2.16.5" if package == "nvidia-nccl-cu12" else None,
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        SimpleNamespace(
            __version__="2.15.1",
            sysconfig=SimpleNamespace(
                get_build_info=lambda: {"cuda_version": "12.2", "cudnn_version": "8"}
            ),
            config=SimpleNamespace(list_physical_devices=lambda _kind: ["GPU:0"]),
        ),
    )
    monkeypatch.setattr(
        run_metadata.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="575.57.08, Tesla V100-SXM2-32GB\n"
        ),
    )

    run_metadata.write_run_metadata(output, "tensorflow")

    metadata = json.loads(output.read_text(encoding="utf-8"))
    assert metadata["pip_freeze"] == ["tensorflow==2.15.1"]
    assert metadata["stack"]["framework_version"] == "2.15.1"
    assert metadata["stack"]["cuda_build"] == "12.2"
    assert metadata["stack"]["cudnn_build"] == "8"
    assert metadata["stack"]["nccl_runtime"] == "2.16.5"
    assert metadata["stack"]["driver"] == "575.57.08"


def test_cpu_run_uses_current_environment_freeze(tmp_path, monkeypatch):
    output = tmp_path / "run-metadata.json"
    freeze_output = ["torch==2.13.0+cpu\n"]
    monkeypatch.delenv("CANVAS_VAE_PIP_FREEZE_FILE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.13.0+cpu",
            version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )

    def run(command, **_kwargs):
        assert command[:3] == ["uv", "pip", "freeze"]
        return SimpleNamespace(stdout=freeze_output[0])

    monkeypatch.setattr(run_metadata.subprocess, "run", run)

    run_metadata.write_run_metadata(output, "torch")
    run_metadata.write_run_metadata(output, "torch")

    metadata = json.loads(output.read_text(encoding="utf-8"))
    assert metadata["pip_freeze"] == ["torch==2.13.0+cpu"]
    assert metadata["stack"]["cuda_available"] is False
    assert metadata["stack"]["driver"] is None
    assert metadata["stack"]["gpu"] is None

    freeze_output[0] = "torch==2.13.0+cu126\n"
    with pytest.raises(RuntimeError, match="different stack"):
        run_metadata.write_run_metadata(output, "torch")


def test_hidden_gpu_has_no_device_identity(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    assert run_metadata._gpu_identity() == (None, None)


def test_metadata_cli_uses_requested_framework_and_path(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_metadata", "--framework", "torch", "--output", "run.json"],
    )
    monkeypatch.setattr(
        run_metadata,
        "write_run_metadata",
        lambda path, framework: recorded.append((path, framework)),
    )

    run_metadata.main()

    assert recorded == [(Path("run.json"), "torch")]

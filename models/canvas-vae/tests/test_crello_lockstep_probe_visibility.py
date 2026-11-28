import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[3]
    / "models"
    / "canvas-vae"
    / "tests"
    / "vendor_parity"
    / "crello_lockstep_probe.py"
)
SPEC = importlib.util.spec_from_file_location("crello_lockstep_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


def test_cuda_probe_preserves_one_selected_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    PROBE._configure_device_visibility("cuda")

    assert PROBE.os.environ["CUDA_VISIBLE_DEVICES"] == "7"


@pytest.mark.parametrize("visible_devices", ["", "0,1"])
def test_cuda_probe_requires_one_selected_device(
    monkeypatch: pytest.MonkeyPatch, visible_devices: str
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)

    with pytest.raises(RuntimeError, match="exactly one selected device"):
        PROBE._configure_device_visibility("cuda")


def test_cpu_probe_hides_cuda_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    PROBE._configure_device_visibility("cpu")

    assert PROBE.os.environ["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize(
    ("cross_step", "noise_step", "expected"),
    [
        (None, None, True),
        (None, 4, True),
        (8, 4, True),
        (3, 4, False),
        (8, None, False),
    ],
)
def test_crello_lockstep_gate_truth_table(
    cross_step: int | None, noise_step: int | None, expected: bool
) -> None:
    assert PROBE._gate_passed(cross_step, noise_step) is expected

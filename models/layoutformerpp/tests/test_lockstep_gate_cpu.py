"""CPU-only checks for the PubLayNet lockstep gate."""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest


RUNNER_PATH = Path(__file__).parent / "vendor_parity" / "run_publaynet_300_lockstep.py"


class ReferenceModuleStub(ModuleType):
    reference_train_loader_stream: Callable[..., list[object]]


class SupportModuleStub(ModuleType):
    capture_provenance: Callable[..., dict[str, object]]


def _load_gate_module() -> ModuleType:
    stub_names = (
        "test_layoutformerpp_training_parity",
        "layoutformerpp_training_reference",
        "publaynet_precheck_support",
    )
    previous_modules = {name: sys.modules.get(name) for name in stub_names}
    parity = ModuleType(stub_names[0])
    reference = ReferenceModuleStub(stub_names[1])
    support = SupportModuleStub(stub_names[2])

    def empty_reference_stream(*args: object, **kwargs: object) -> list[object]:
        return []

    def empty_provenance(*args: object, **kwargs: object) -> dict[str, object]:
        return {}

    reference.reference_train_loader_stream = empty_reference_stream
    support.capture_provenance = empty_provenance

    sys.modules[stub_names[0]] = parity
    sys.modules[stub_names[1]] = reference
    sys.modules[stub_names[2]] = support

    try:
        spec = importlib.util.spec_from_file_location(
            "publaynet_lockstep_gate_cpu", RUNNER_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
                continue

            sys.modules[name] = previous

    return module


LOCKSTEP = _load_gate_module()


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
def test_publaynet_lockstep_gate_truth_table(
    cross_step: int | None, noise_step: int | None, expected: bool
) -> None:
    assert LOCKSTEP._gate_passed(cross_step, noise_step) is expected

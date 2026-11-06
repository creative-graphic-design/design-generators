from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_core_import_does_not_require_training_extra() -> None:
    package_root = Path(__file__).parents[1] / "src"
    script = """
import builtins
import sys

real_import = builtins.__import__

def import_without_lightning(name, *args, **kwargs):
    if name == "lightning" or name.startswith("lightning."):
        raise ModuleNotFoundError("blocked optional training dependency")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_lightning
import ralf
assert "ralf.training" not in sys.modules
assert ralf.RalfConfig is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(package_root)},
    )
    assert result.returncode == 0, result.stderr

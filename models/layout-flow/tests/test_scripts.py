from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "models" / "layout-flow" / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scripts_expose_help_with_defaults() -> None:
    for script in [
        "download_original.py",
        "generate_reference_outputs.py",
        "convert_original_checkpoint.py",
    ]:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in result.stdout
        assert "default:" in result.stdout


def test_convert_script_defaults_are_dataset_specific() -> None:
    convert = _load_script("convert_original_checkpoint")

    assert convert.default_checkpoint("publaynet").name == (
        "checkpoint_PubLayNet_LayoutFlow.ckpt"
    )
    assert convert.default_checkpoint("rico25").name == (
        "checkpoint_RICO_LayoutFlow.ckpt"
    )
    assert convert.default_output_dir("publaynet").name == "publaynet"

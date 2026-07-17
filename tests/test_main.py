"""Tests for the default command-line entrypoint."""

import importlib.util
import runpy
from pathlib import Path
from typing import Protocol, cast

import pytest


class MainModule(Protocol):
    def main(self) -> None: ...


def load_main_module() -> MainModule:
    module_path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("main", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(MainModule, module)


def test_main_prints_default_greeting(capsys: pytest.CaptureFixture[str]) -> None:
    load_main_module().main()

    assert capsys.readouterr().out == "Hello from design-generators!\n"


def test_main_script_entrypoint_prints_default_greeting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module_path = Path(__file__).resolve().parents[1] / "main.py"

    runpy.run_path(str(module_path), run_name="__main__")

    assert capsys.readouterr().out == "Hello from design-generators!\n"

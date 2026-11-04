from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
import sys

import pytest
import torch

SCRIPTS = Path(__file__).parents[1] / "scripts"
SCRIPT = SCRIPTS / "run_package_eval.sh"
VENDOR_SPEC = importlib.util.spec_from_file_location(
    "layoutganpp_run_vendor_full", SCRIPTS / "run_vendor_full.py"
)
assert VENDOR_SPEC is not None and VENDOR_SPEC.loader is not None
VENDOR_RUN = importlib.util.module_from_spec(VENDOR_SPEC)
VENDOR_SPEC.loader.exec_module(VENDOR_RUN)


@pytest.mark.parametrize("absolute", (False, True))
def test_package_eval_harness_rejects_output_outside_repository(
    absolute: bool,
) -> None:
    output_root = (
        str(Path.cwd().parent / "layoutganpp-eval") if absolute else "../outside"
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "magazine", "42004", "0", output_root],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "repository-relative" in result.stdout


def test_seeded_vendor_loader_adds_data_seed_only_for_shuffled_data() -> None:
    calls: list[tuple[bool, int | None]] = []

    def loader_factory(
        _dataset: list[int], *, shuffle: bool, generator: torch.Generator | None = None
    ) -> None:
        calls.append(
            (
                shuffle,
                None if generator is None else generator.initial_seed(),
            )
        )

    loader = VENDOR_RUN.seeded_training_loader(loader_factory, 42003)
    loader([1], shuffle=True)
    loader([1], shuffle=False)

    assert calls == [(True, 42003), (False, None)]


def test_vendor_progress_record_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    path.write_text("old\n")

    VENDOR_RUN.write_progress(path, 50)

    assert json.loads(path.read_text())["updates"] == 50
    assert not path.with_suffix(".tmp").exists()


def test_vendor_harness_refuses_existing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(VENDOR_RUN, "VENDOR_WORK", tmp_path)
    output_dir = tmp_path / "output/magazine/LayoutGAN++/existing"
    output_dir.mkdir(parents=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_vendor_full.py",
            "--dataset",
            "magazine",
            "--seed",
            "42004",
            "--data-seed",
            "42003",
            "--iterations",
            "200000",
            "--run-name",
            "existing",
            "--progress-json",
            str(tmp_path / "progress.json"),
        ],
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        VENDOR_RUN.main()

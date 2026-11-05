from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
import runpy
import sys
from typing import cast

import pytest


def test_reference_metadata_generator_accepts_documented_cache_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[3]
    script = runpy.run_path(
        str(repository / "models/radm/scripts/generate_reference_outputs.py"),
        run_name="radm_reference_outputs_test",
    )
    dataset_root = tmp_path / ".cache" / "radm" / "data" / "cgl"
    text_root = tmp_path / ".cache" / "radm" / "text_features" / "cgl"
    vendor_root = tmp_path / "vendor" / "radm"
    checkpoint = tmp_path / ".cache" / "radm" / "weights" / "radm_cgl.pth"
    output_dir = tmp_path / ".cache" / "radm" / "reference" / "cgl"
    dataset_root.mkdir(parents=True)
    text_root.mkdir(parents=True)
    vendor_root.mkdir(parents=True)
    (vendor_root / "train_net.py").touch()
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    main = cast(Callable[[], None], script["main"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_reference_outputs.py",
            "--vendor-root",
            str(vendor_root),
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(dataset_root),
            "--text-feature-root",
            str(text_root),
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_root"] == str(dataset_root)
    assert metadata["text_feature_root"] == str(text_root)

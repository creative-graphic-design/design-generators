from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping, cast

import numpy as np
import torch

from traingen.checkpointing import (
    capture_rng_state,
    load_weights_only_checkpoint,
    SafeInput,
    publish_checkpoint_bundle,
    retain_checkpoint_bundles,
    restore_rng_state,
    save_weights_only_checkpoint,
    SafeValue,
)


def test_weights_only_checkpoint_restores_rng_and_loads_safely(tmp_path: Path) -> None:
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(2))
    path = tmp_path / "state.pt"
    save_weights_only_checkpoint(
        path,
        cast(Mapping[str, SafeInput], {"rng": state, "step": 2}),
    )

    loaded = load_weights_only_checkpoint(path)
    restore_rng_state(cast(Mapping[str, SafeValue], loaded["rng"]))

    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(2), expected[2])


def test_checkpoint_bundle_hashes_files_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    manifest = tmp_path / "launch_manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text(json.dumps({"identity": "test"}), encoding="utf-8")

    bundle = publish_checkpoint_bundle(
        {"checkpoint.pt": checkpoint},
        manifest,
        tmp_path / "durable",
        "track/epoch-1",
    )

    bundle_manifest = json.loads(
        (bundle / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert bundle.name == "epoch-1"
    assert set(bundle_manifest["files"]) == {"checkpoint.pt", "launch_manifest.json"}
    assert not list((bundle.parent).glob("*.tmp"))


def test_checkpoint_bundle_retention_keeps_two_epochs_and_best(tmp_path: Path) -> None:
    track = tmp_path / "track"
    for name in ("epoch-000001", "epoch-000002", "epoch-000003"):
        bundle = track / name
        bundle.mkdir(parents=True)
        (bundle / "bundle_manifest.json").write_text("{}", encoding="utf-8")
    best = track / "best" / "epoch-000002"
    best.mkdir(parents=True)
    (best / "bundle_manifest.json").write_text("{}", encoding="utf-8")

    retain_checkpoint_bundles(tmp_path, "track")

    assert sorted(path.name for path in track.glob("epoch-*") if path.is_dir()) == [
        "epoch-000002",
        "epoch-000003",
    ]
    assert best.is_dir()

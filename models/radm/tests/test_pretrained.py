from __future__ import annotations

import hashlib
import io
import json
import pickle
from importlib.resources import files
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from radm.modeling_radm import RADMBackbone
from radm.training import pretrained


class _TinyResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 2, kernel_size=1, bias=False)


class _TinyFeaturePyramid(nn.Module):
    placeholder: nn.Parameter

    def __init__(self) -> None:
        super().__init__()

        self.placeholder = nn.Parameter(torch.tensor(7.0))


class _TinyBottomUp(nn.Module):
    body: _TinyResNet
    fpn: _TinyFeaturePyramid

    def __init__(self) -> None:
        super().__init__()

        self.body = _TinyResNet()
        self.fpn = _TinyFeaturePyramid()


class _TinyBackbone(nn.Module):
    body: _TinyBottomUp

    def __init__(self) -> None:
        super().__init__()

        self.body = _TinyBottomUp()


def _checkpoint(weight: torch.Tensor) -> bytes:
    return pickle.dumps(
        {
            "__author__": "torchvision",
            "matching_heuristics": True,
            "model": {
                "stem.conv1.weight": weight,
                "stem.fc.weight": torch.ones(1),
                "stem.fc.bias": torch.zeros(1),
            },
        }
    )


def test_load_pretrained_r50_maps_resnet_and_keeps_fpn_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1)
    content = _checkpoint(expected)
    path = tmp_path / "R-50.pkl"
    path.write_bytes(content)
    monkeypatch.setattr(
        pretrained,
        "_r50_artifact",
        lambda: (
            "https://example.invalid/R-50.pkl",
            hashlib.sha256(content).hexdigest(),
        ),
    )
    tiny_backbone = _TinyBackbone()
    initial_fpn = tiny_backbone.body.fpn.placeholder.clone()

    pretrained.load_pretrained_r50(cast(RADMBackbone, tiny_backbone), path)

    assert torch.equal(tiny_backbone.body.body.conv1.weight, expected)
    assert torch.equal(tiny_backbone.body.fpn.placeholder, initial_fpn)


def test_r50_weights_download_is_hash_checked_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"verified R-50 checkpoint bytes"
    monkeypatch.setattr(
        pretrained,
        "_r50_artifact",
        lambda: (
            "https://example.invalid/R-50.pkl",
            hashlib.sha256(content).hexdigest(),
        ),
    )
    monkeypatch.setattr(pretrained, "urlopen", lambda url, timeout: io.BytesIO(content))
    path = tmp_path / "weights" / "R-50.pkl"

    payload = pretrained._ensure_r50_weights(path)

    assert payload == content
    assert path.read_bytes() == content
    assert list(path.parent.iterdir()) == [path]


def test_r50_weights_reject_bad_download_without_leaving_a_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pretrained,
        "_r50_artifact",
        lambda: ("https://example.invalid/R-50.pkl", "0" * 64),
    )
    monkeypatch.setattr(
        pretrained, "urlopen", lambda url, timeout: io.BytesIO(b"wrong bytes")
    )
    path = tmp_path / "weights" / "R-50.pkl"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pretrained._ensure_r50_weights(path)

    assert not path.exists()
    assert list(path.parent.iterdir()) == []


def test_r50_artifact_matches_the_packaged_source_manifest() -> None:
    manifest = json.loads(
        files("radm.training")
        .joinpath("r50_weights_manifest.json")
        .read_text(encoding="utf-8")
    )

    assert pretrained._r50_artifact() == (manifest["source_url"], manifest["sha256"])

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from typing import cast

import pytest

from radm.training.data_paths import validate_dot_free_data_root


def _launcher() -> dict[str, object]:
    root = Path(__file__).parents[3]
    return runpy.run_path(
        str(root / "models/radm/scripts/train_cgl_s5.py"),
        run_name="radm_cgl_s5_test",
    )


def _write_dataset(root: Path, *, empty_split: str | None = None) -> None:
    for split in ("train", "test"):
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        feature_root = root / "text_features" / split
        feature_root.mkdir(parents=True, exist_ok=True)
        (root / "annotations" / f"{split}.json").write_text(
            json.dumps({"images": [{"file_name": f"{split}/image.jpg"}]}),
            encoding="utf-8",
        )
        if split != empty_split:
            (feature_root / "image_feats.pth").write_bytes(b"features")


def test_dot_free_data_root_makes_relative_alias_absolute_without_resolving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="radm-cgl-", dir="/tmp") as temp_dir:
        root = Path(temp_dir)
        monkeypatch.chdir(root)
        visible_root = Path.cwd()
        target = visible_root / ".cache" / "cgl"
        target.mkdir(parents=True)
        alias = visible_root / "radm_cgl"
        alias.symlink_to(Path(".cache") / "cgl", target_is_directory=True)

        visible_path = validate_dot_free_data_root("radm_cgl")

        assert visible_path == str(alias)
        assert Path(visible_path).is_absolute()
        assert os.readlink(alias) == os.path.join(".cache", "cgl")


@pytest.mark.parametrize("suffix", (".cache/cgl", "cgl.v1"))
def test_dot_free_data_root_rejects_dotted_components(suffix: str) -> None:
    with pytest.raises(ValueError, match="dot-free path components"):
        validate_dot_free_data_root(f"/data/{suffix}")


def test_dot_free_data_root_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_dot_free_data_root("")


@pytest.mark.parametrize("arm", ("original", "package"))
def test_s5_launcher_builds_same_absolute_data_root_for_both_arms(
    arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher()
    build_command = cast(
        Callable[..., tuple[list[str], Path]], launcher["build_command"]
    )
    monkeypatch.chdir("/tmp")
    visible_root = Path.cwd()

    command, _ = build_command(
        arm=arm,
        seed=2,
        data_root="radm_cgl",
        output_dir=tmp_path / arm,
    )

    assert any(str(visible_root / "radm_cgl") in arg for arg in command)
    assert (visible_root / "radm_cgl").is_absolute()


@pytest.mark.parametrize(
    ("arm", "checkpoint_name"),
    (
        ("package", "epoch=0-step=250000.ckpt"),
        ("original", "model_final.pth"),
    ),
)
def test_evaluator_command_names_the_terminal_checkpoint(
    arm: str, checkpoint_name: str, tmp_path: Path
) -> None:
    launcher = _launcher()
    terminal_checkpoint_path = cast(
        Callable[..., Path], launcher["terminal_checkpoint_path"]
    )
    checkpoint_path = terminal_checkpoint_path(arm, tmp_path)
    assert checkpoint_path.name == checkpoint_name
    build_evaluator_command = cast(
        Callable[..., tuple[list[str], Path]], launcher["build_evaluator_command"]
    )

    command, _ = build_evaluator_command(
        arm=arm,
        seed=1,
        data_root="/data/radm_cgl",
        output_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        checkpoint_path=checkpoint_path,
    )

    assert str(checkpoint_path) in command


def test_s5_launcher_writes_manifest_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher()
    main = cast(Callable[[], None], launcher["main"])
    output_dir = tmp_path / "package-run"
    source_commit = "a" * 40
    with tempfile.TemporaryDirectory(prefix="radm-cgl-", dir="/tmp") as data_dir:
        data_parent = Path(data_dir)
        data_target = data_parent / ".cache" / "cgl"
        _write_dataset(data_target)
        data_alias = data_parent / "radm_cgl"
        data_alias.symlink_to(Path(".cache") / "cgl", target_is_directory=True)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_cgl_s5.py",
                "--arm",
                "package",
                "--seed",
                "1",
                "--data-root",
                str(data_alias),
                "--source-commit",
                source_commit,
                "--output-dir",
                str(output_dir),
            ],
        )
        monkeypatch.setattr(
            subprocess, "check_output", lambda *args, **kwargs: source_commit
        )
        launched: list[list[str]] = []

        def capture_training(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            launched.append(command)
            assert manifest["source_commit"] == source_commit
            assert (
                manifest["config"]["path"]
                == "models/radm/configs/training/radm_cgl.yaml"
            )
            assert manifest["config"]["sha256"]
            assert manifest["checkpoint_rule"]["filename"] == "epoch=0-step=250000.ckpt"
            assert any(
                "epoch=0-step=250000.ckpt" in arg
                for arg in manifest["evaluator_command"]
            )
            assert manifest["text_feature_inventory"] == {
                "train": {"annotation_images": 1, "matching_text_features": 1},
                "test": {"annotation_images": 1, "matching_text_features": 1},
            }
            assert manifest["data_root"]["alias"] == str(data_alias)
            assert manifest["data_root"]["target"] == str(data_target.resolve())
            assert (
                manifest["artifacts"][str(data_alias)]
                == manifest["artifacts"][str(data_target.resolve())]
            )
            assert str(data_alias) in manifest["evaluator_command"]
            assert datetime.fromisoformat(manifest["launched_at"]).tzinfo is not None
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(subprocess, "run", capture_training)

        main()

        assert len(launched) == 1


def test_existing_output_root_fails_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher()
    main = cast(Callable[[], None], launcher["main"])
    output_dir = tmp_path / "existing-run"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_cgl_s5.py",
            "--arm",
            "package",
            "--seed",
            "1",
            "--data-root",
            "/tmp/radm_cgl",
            "--source-commit",
            "a" * 40,
            "--output-dir",
            str(output_dir),
        ],
    )

    def fail_if_subprocess_starts(*args: object, **kwargs: object) -> None:
        raise AssertionError("training subprocess must not start")

    monkeypatch.setattr(subprocess, "check_output", fail_if_subprocess_starts)
    monkeypatch.setattr(subprocess, "run", fail_if_subprocess_starts)

    with pytest.raises(SystemExit, match="refusing to reuse training output directory"):
        main()
    assert marker.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize("empty_split", ("train", "test"))
def test_empty_text_feature_splits_fail_before_subprocess(
    empty_split: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher()
    main = cast(Callable[[], None], launcher["main"])
    with tempfile.TemporaryDirectory(prefix="radm-cgl-", dir="/tmp") as data_dir:
        data_root = Path(data_dir) / "radm_cgl"
        _write_dataset(data_root, empty_split=empty_split)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_cgl_s5.py",
                "--arm",
                "package",
                "--seed",
                "1",
                "--data-root",
                str(data_root),
                "--source-commit",
                "a" * 40,
                "--output-dir",
                str(tmp_path / "package-run"),
            ],
        )

        def fail_if_subprocess_starts(*args: object, **kwargs: object) -> None:
            raise AssertionError("training subprocess must not start")

        monkeypatch.setattr(subprocess, "check_output", fail_if_subprocess_starts)
        monkeypatch.setattr(subprocess, "run", fail_if_subprocess_starts)

        with pytest.raises(SystemExit, match="no text-feature files"):
            main()

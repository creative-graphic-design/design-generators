from __future__ import annotations

import json
import inspect
import os
from copy import deepcopy
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import FunctionType, SimpleNamespace
from typing import cast

import pytest
import torch
import yaml
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from radm.training.config import effective_radm_config
from radm.training.datamodule import RADMAspectRatioBatchSampler, RADMDataModule
from radm.training.data_paths import validate_dot_free_data_root
from radm.training.dataset import RADMCOCODataset


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


class _TinyAspectRatioDataset(Dataset[torch.Tensor]):
    def __init__(self, size: int) -> None:
        self.images = [{"width": 10.0, "height": 10.0} for _ in range(size)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([float(index)])


class _TinyCheckpointModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        loss = (self.weight - batch.mean()) ** 2
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> None:
        self.log("val_loss", self.weight.detach().square())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _train_tiny_checkpoint(checkpoint_dir: Path, recipe: dict[str, object]) -> Path:
    trainer_config = cast(dict[str, object], recipe["trainer"])
    data_args = cast(
        dict[str, object], cast(dict[str, object], recipe["data"])["init_args"]
    )
    batch_size = cast(int, data_args["batch_size"])
    effective = replace(effective_radm_config(), max_iter=2)
    dataset = _TinyAspectRatioDataset(batch_size)
    train_loader = DataLoader(
        dataset,
        batch_sampler=RADMAspectRatioBatchSampler(
            cast(RADMCOCODataset, dataset),
            batch_size=batch_size,
            seed=effective.seed,
            max_batches=effective.max_iter,
        ),
    )
    assert len(train_loader) == effective.max_iter
    callback_config = cast(list[dict[str, object]], trainer_config["callbacks"])[0]
    callback_args = cast(dict[str, object], callback_config["init_args"])
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=cast(str, callback_args["monitor"]),
        mode=cast(str, callback_args["mode"]),
        save_top_k=cast(int, callback_args["save_top_k"]),
        save_last=cast(bool, callback_args["save_last"]),
    )
    validation_loader = DataLoader(dataset, batch_size=batch_size)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=effective.max_iter,
        check_val_every_n_epoch=cast(int, trainer_config["check_val_every_n_epoch"]),
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        _TinyCheckpointModule(),
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )
    expected_path = checkpoint_dir / "epoch=0-step=2.ckpt"
    assert checkpoint_callback.best_model_path == str(expected_path)
    return expected_path


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
        Callable[..., tuple[list[str], Path, dict[str, str] | None]],
        launcher["build_command"],
    )
    monkeypatch.chdir("/tmp")
    visible_root = Path.cwd()

    command, _, environment = build_command(
        arm=arm,
        seed=2,
        data_root="radm_cgl",
        output_dir=tmp_path / arm,
    )

    assert any(str(visible_root / "radm_cgl") in arg for arg in command)
    assert (visible_root / "radm_cgl").is_absolute()
    if arm == "original":
        compat_root = Path(cast(str, launcher["VENDOR_COMPAT_ROOT"]))
        assert environment is not None
        assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(compat_root)
        assert "Image.LINEAR" not in " ".join(command)
    else:
        assert environment is None


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
        Callable[..., tuple[list[str], Path, dict[str, str] | None]],
        launcher["build_evaluator_command"],
    )

    command, _, environment = build_evaluator_command(
        arm=arm,
        seed=1,
        data_root="/data/radm_cgl",
        output_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        checkpoint_path=checkpoint_path,
    )

    assert str(checkpoint_path) in command
    if arm == "original":
        compat_root = Path(cast(str, launcher["VENDOR_COMPAT_ROOT"]))
        assert environment is not None
        assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(compat_root)
        assert "Image.LINEAR" not in " ".join(command)
    else:
        assert environment is None


def test_original_arm_manifest_records_the_tracked_compatibility_module() -> None:
    launcher = _launcher()
    metadata_for_arm = cast(
        Callable[[str], dict[str, str] | None],
        launcher["vendor_compatibility_metadata"],
    )
    metadata = metadata_for_arm("original")
    assert metadata is not None
    assert metadata["module"] == "models/radm/scripts/vendor_compat/sitecustomize.py"
    assert metadata["sha256"]
    assert metadata["pythonpath_entry"].endswith("models/radm/scripts/vendor_compat")
    assert metadata_for_arm("package") is None
    compatibility_environment = cast(
        Callable[[], dict[str, str]], launcher["vendor_compatibility_environment"]
    )()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sitecustomize; from PIL import Image; "
            "assert Image.LINEAR == Image.BILINEAR; print(sitecustomize.__file__)",
        ],
        env=compatibility_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == str(launcher["VENDOR_COMPAT_MODULE"])


def test_package_cgl_validation_only_selects_the_terminal_top_k_checkpoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[3]
    recipe = yaml.safe_load(
        (root / "models/radm/configs/training/radm_cgl.yaml").read_text(
            encoding="utf-8"
        )
    )
    trainer = recipe["trainer"]
    trainer_defaults = inspect.signature(Trainer).parameters
    data = recipe["data"]["init_args"]
    callbacks = trainer["callbacks"]
    assert trainer["max_steps"] == 250000
    assert trainer["check_val_every_n_epoch"] == 1
    assert (
        trainer.get(
            "val_check_interval", trainer_defaults["val_check_interval"].default
        )
        is None
    )
    assert (
        trainer.get(
            "limit_train_batches", trainer_defaults["limit_train_batches"].default
        )
        is None
    )
    assert (
        trainer.get(
            "accumulate_grad_batches",
            trainer_defaults["accumulate_grad_batches"].default,
        )
        == 1
    )
    assert data["effective"] == {}
    assert callbacks[0]["class_path"] == "lightning.pytorch.callbacks.ModelCheckpoint"
    assert callbacks[0]["init_args"]["monitor"] == "val_loss"
    assert callbacks[0]["init_args"]["save_top_k"] == 1
    assert "filename" not in callbacks[0]["init_args"]
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    formatted_checkpoint = checkpoint.format_checkpoint_name(
        cast(
            dict[str, torch.Tensor],
            {"epoch": 0, "step": trainer["max_steps"]},
        )
    )
    assert Path(formatted_checkpoint).name == "epoch=0-step=250000.ckpt"

    annotations = tmp_path / "annotations" / "train.json"
    annotations.parent.mkdir()
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "image.jpg", "width": 10, "height": 10}
                ],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    data_module = RADMDataModule(
        train_annotations=annotations,
        train_image_root=tmp_path / "images",
        train_text_feature_root=tmp_path / "text_features",
        batch_size=data["batch_size"],
        num_workers=data["num_workers"],
        effective=effective_radm_config(),
    )
    assert len(data_module.train_dataloader()) >= trainer["max_steps"]


def test_evaluator_reports_a_missing_terminal_checkpoint_clearly(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[3]
    evaluator = runpy.run_path(
        str(root / "models/radm/tests/vendor_parity/evaluate_radm_checkpoint.py"),
        run_name="radm_evaluator_missing_checkpoint_test",
    )
    validate = cast(
        Callable[..., dict[str, object]], evaluator["validate_terminal_checkpoint"]
    )

    with pytest.raises(RuntimeError, match="terminal checkpoint is missing"):
        validate(
            tmp_path / "checkpoints" / "epoch=0-step=250000.ckpt",
            tmp_path / "manifest.json",
            seed=1,
        )


@pytest.mark.parametrize("checkpoint_present", (True, False))
def test_s5_launcher_writes_manifest_before_training(
    checkpoint_present: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher()
    main = cast(Callable[[], None], launcher["main"])
    launcher_globals = cast(FunctionType, main).__globals__
    monkeypatch.setitem(launcher_globals, "STEPS", 2)
    steps = int(launcher_globals["STEPS"])
    checkpoint_name = f"epoch=0-step={steps}.ckpt"
    output_dir = tmp_path / "package-run"
    manifest_path = output_dir / "manifest.json"
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
        gitlink_commit = "b" * 40

        vendor_root = Path(cast(str, launcher["VENDOR_ROOT"]))

        def source_git(command: list[str], *, text: bool) -> str:
            assert text
            repository = Path(command[2])
            args = command[3:]
            if args == ["status", "--porcelain", "--untracked-files=all"]:
                return ""
            if args == ["rev-parse", "HEAD"]:
                return gitlink_commit if repository == vendor_root else source_commit
            if args == ["ls-tree", "HEAD", "vendor/radm"]:
                return f"160000 commit {gitlink_commit}\tvendor/radm"
            raise AssertionError(command)

        monkeypatch.setitem(
            cast(FunctionType, launcher["main"]).__globals__,
            "runtime_metadata",
            lambda arm: {
                "torch_version": "2.8.0+cu128",
                "torchvision_version": "0.23.0+cu128",
                "detectron2_version": "0.6",
                "cuda_version": "12.8",
                "gpu_name": "test GPU",
                "num_workers": 0,
                "omp_num_threads": 1,
            },
        )
        launched: list[list[str]] = []

        def capture_training(
            command: list[str],
            *,
            cwd: Path,
            env: dict[str, str] | None,
            check: bool,
        ) -> subprocess.CompletedProcess[bytes]:
            assert command[0] == "traingen"
            assert cwd == launcher["ROOT"]
            assert env is None
            assert check

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            launched.append(command)
            assert manifest["source_commit"] == source_commit
            assert manifest["vendor_gitlink_commit"] == gitlink_commit
            assert manifest["vendor_checkout_commit"] == gitlink_commit
            assert manifest["runtime"]["torch_version"] == "2.8.0+cu128"
            assert manifest["runtime"]["torchvision_version"] == "0.23.0+cu128"
            assert manifest["runtime"]["detectron2_version"] == "0.6"
            assert manifest["runtime"]["cuda_version"] == "12.8"
            assert manifest["runtime"]["gpu_name"] == "test GPU"
            assert manifest["runtime"]["num_workers"] == 0
            assert manifest["runtime"]["omp_num_threads"] == 1
            assert manifest["vendor_compatibility"] is None
            assert (
                manifest["config"]["path"]
                == "models/radm/configs/training/radm_cgl.yaml"
            )
            assert manifest["config"]["sha256"]
            assert manifest["arm"] == "package"
            assert manifest["max_steps"] == steps
            assert manifest["checkpoint_rule"]["filename"] == checkpoint_name
            assert manifest["checkpoint_rule"]["checkpoint"] == str(
                output_dir / "checkpoints" / checkpoint_name
            )
            assert any(checkpoint_name in arg for arg in manifest["evaluator_command"])
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
            if checkpoint_present:
                recipe = cast(
                    dict[str, object],
                    yaml.safe_load(
                        (
                            Path(__file__).parents[3]
                            / "models/radm/configs/training/radm_cgl.yaml"
                        ).read_text(encoding="utf-8")
                    ),
                )
                checkpoint_path = _train_tiny_checkpoint(
                    output_dir / "checkpoints", recipe
                )
                assert checkpoint_path.name == checkpoint_name
                checkpoint_payload = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                assert checkpoint_payload["global_step"] == steps
                assert "radm_s5_identity" not in checkpoint_payload
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setitem(
            launcher_globals,
            "subprocess",
            SimpleNamespace(check_output=source_git, run=capture_training),
        )

        if checkpoint_present:
            main()
            repo_root = Path(__file__).parents[3]
            evaluator = runpy.run_path(
                str(
                    repo_root
                    / "models/radm/tests/vendor_parity/evaluate_radm_checkpoint.py"
                ),
                run_name="radm_manifest_validation_test",
            )
            validate = cast(
                Callable[..., dict[str, object]],
                evaluator["validate_terminal_checkpoint"],
            )
            checkpoint_path = output_dir / "checkpoints" / checkpoint_name
            monkeypatch.chdir(repo_root)
            validation = validate(
                checkpoint_path, manifest_path, seed=1, expected_steps=steps
            )
            assert validation["global_step"] == steps
            assert validation["training_seed"] == 1

            with pytest.raises(RuntimeError, match="training_seed"):
                validate(checkpoint_path, manifest_path, seed=2, expected_steps=steps)

            outside_checkpoint = tmp_path / "outside.ckpt"
            torch.save({"global_step": steps}, outside_checkpoint)
            with pytest.raises(RuntimeError, match="outside the manifest output_root"):
                validate(
                    outside_checkpoint, manifest_path, seed=1, expected_steps=steps
                )

            checkpoint_bytes = checkpoint_path.read_bytes()
            torch.save({"global_step": steps - 1}, checkpoint_path)
            with pytest.raises(RuntimeError, match=f"global_step={steps - 1}"):
                validate(checkpoint_path, manifest_path, seed=1, expected_steps=steps)
            checkpoint_path.write_bytes(checkpoint_bytes)

            original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for section, field, value, message in (
                (None, "arm", "original", "manifest arm"),
                (
                    None,
                    "output_root",
                    str(output_dir.parent),
                    "manifest output_root",
                ),
                ("config", "path", "other.yaml", "manifest config path"),
                ("config", "sha256", "0" * 64, "manifest config digest"),
                (
                    "checkpoint_rule",
                    "checkpoint",
                    str(output_dir / "checkpoints" / "other.ckpt"),
                    "manifest checkpoint rule",
                ),
                (None, "max_steps", 249999, "manifest max_steps"),
            ):
                mismatched_manifest = deepcopy(original_manifest)
                if section is None:
                    mismatched_manifest[field] = value
                else:
                    mismatched_manifest[section][field] = value
                manifest_path.write_text(
                    json.dumps(mismatched_manifest), encoding="utf-8"
                )
                with pytest.raises(RuntimeError, match=message):
                    validate(
                        checkpoint_path,
                        manifest_path,
                        seed=1,
                        expected_steps=steps,
                    )
            manifest_path.write_text(json.dumps(original_manifest), encoding="utf-8")
        else:
            with pytest.raises(
                SystemExit, match="training completed without the terminal checkpoint"
            ):
                main()

        assert len(launched) == 1


@pytest.mark.parametrize(
    ("root_status", "vendor_status", "vendor_head"),
    (
        (" M tracked.py", "", "b" * 40),
        ("?? untracked.py", "", "b" * 40),
        ("", " M tracked.py", "b" * 40),
        ("", "?? untracked.py", "b" * 40),
        ("", "", "c" * 40),
    ),
)
def test_s5_source_tree_must_be_clean_and_vendor_must_match_gitlink(
    root_status: str,
    vendor_status: str,
    vendor_head: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher()
    verify_source_tree = cast(
        Callable[..., dict[str, str]], launcher["verify_source_tree"]
    )
    root = cast(Path, launcher["ROOT"])
    source_commit = "a" * 40
    vendor_gitlink = "b" * 40

    def source_git(command: list[str], **kwargs: object) -> str:
        repository = Path(command[2])
        args = command[3:]
        if args == ["status", "--porcelain", "--untracked-files=all"]:
            return root_status if repository == root else vendor_status
        if args == ["rev-parse", "HEAD"]:
            return source_commit if repository == root else vendor_head
        if args == ["ls-tree", "HEAD", "vendor/radm"]:
            return f"160000 commit {vendor_gitlink}\tvendor/radm"
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "check_output", source_git)
    if root_status or vendor_status:
        with pytest.raises(SystemExit, match="dirty .* source tree"):
            verify_source_tree(source_commit)
    elif vendor_head != vendor_gitlink:
        with pytest.raises(SystemExit, match="does not match root gitlink"):
            verify_source_tree(source_commit)
    else:
        provenance = verify_source_tree(source_commit)
        assert provenance == {
            "source_commit": source_commit,
            "vendor_gitlink_commit": vendor_gitlink,
            "vendor_checkout_commit": vendor_head,
        }


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

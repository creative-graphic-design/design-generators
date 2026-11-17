from __future__ import annotations

import importlib.util
import json
import random
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import TypeGuard

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from traingen.checkpointing import (
    load_weights_only_checkpoint,
    safe_value_type,
)

pytestmark = pytest.mark.vendor_parity

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[2] / "scripts" / "publaynet_s5"
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from vendor_resume import (  # noqa: E402
    _patch_vendor_sources,
    load_resume_state,
    prepare_vendor_tree,
    save_resume_state,
)


class _Rows(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.generator = torch.Generator().manual_seed(19)
        self.rows = [(float(index), float(index + 1)) for index in range(8)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x, y = self.rows[index]
        return {
            "x": torch.tensor([x + torch.rand((), generator=self.generator)]),
            "y": torch.tensor([y]),
        }


def _load_basic(vendor_source: Path, module_tag: str, compute_maximum_iou=None):
    wandb = types.ModuleType("wandb")
    setattr(wandb, "init", lambda *args, **kwargs: None)
    setattr(wandb, "log", lambda *args, **kwargs: None)
    sys.modules["wandb"] = wandb
    data_package = types.ModuleType("data")
    data_package.__path__ = [str(vendor_source / "data")]
    sys.modules["data"] = data_package
    evaluation = types.ModuleType("evaluation")
    evaluation.__path__ = [str(vendor_source / "evaluation")]
    metrics = types.ModuleType("evaluation.metrics")
    setattr(
        metrics,
        "compute_overlap",
        lambda boxes, mask: boxes.new_zeros(boxes.size(0)),
    )
    setattr(metrics, "compute_maximum_iou", compute_maximum_iou or (lambda *_: 1.0))
    sys.modules["evaluation"] = evaluation
    sys.modules["evaluation.metrics"] = metrics
    for name in (
        "trainer",
        "trainer.basic_trainer",
        "trainer.multitask_trainer",
        "trainer.utils",
        "trainer.ds_trainer",
        "layoutformerpp_resume",
        "model",
    ):
        sys.modules.pop(name, None)
    for name in ("utils", "utils.utils"):
        sys.modules.pop(name, None)
    package = types.ModuleType("trainer")
    package.__path__ = [str(vendor_source / "trainer")]
    sys.modules["trainer"] = package
    sys.path.insert(0, str(vendor_source))
    spec = importlib.util.spec_from_file_location(
        f"trainer.basic_trainer_{module_tag}",
        vendor_source / "trainer" / "basic_trainer.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("vendor basic trainer could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Trainer


def _load_multitask(vendor_source: Path, module_tag: str, compute_maximum_iou):
    _load_basic(vendor_source, module_tag, compute_maximum_iou)
    model = types.ModuleType("model")
    setattr(
        model, "LayoutTransformerTokenizer", type("LayoutTransformerTokenizer", (), {})
    )
    sys.modules["model"] = model
    ds_trainer = types.ModuleType("trainer.ds_trainer")
    setattr(ds_trainer, "DSTrainer", type("DSTrainer", (), {}))
    sys.modules["trainer.ds_trainer"] = ds_trainer
    spec = importlib.util.spec_from_file_location(
        "trainer.multitask_trainer",
        vendor_source / "trainer" / "multitask_trainer.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("vendor multitask trainer could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules["trainer.multitask_trainer"] = module
    spec.loader.exec_module(module)
    return module.MultiTaskTrainer


class _MultiTaskRows(_Rows):
    num_tasks = 1
    seq_processor = None

    def switch_task(self, task: str) -> None:
        assert task == "gen_t"


def _args(output: Path, epochs: int) -> Namespace:
    return Namespace(
        seed=31,
        out_dir=str(output),
        max_num_elements=1,
        epoch=epochs,
        load_train_ckpt=False,
        train_ckpt_path=None,
        gradient_accumulation=1,
        enable_clip_gradient=False,
        clip_gradient=1.0,
        batch_size=2,
        eval_batch_size=2,
        dataset="publaynet",
        lr=0.05,
        train_log_step=100,
        bbox_format="ltwh",
        discrete_x_grid=128,
        discrete_y_grid=128,
    )


def _run_vendor(
    vendor_source: Path,
    output: Path,
    epochs: int,
    monkeypatch: pytest.MonkeyPatch,
    resume_path: Path | None = None,
) -> tuple[dict[str, torch.Tensor], list[Path]]:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    if resume_path is None:
        monkeypatch.delenv("LAYOUTFORMERPP_RESUME_PATH", raising=False)
    else:
        monkeypatch.setenv("LAYOUTFORMERPP_RESUME_PATH", str(resume_path))
    torch.manual_seed(31)
    trainer_class = _load_basic(vendor_source, output.name)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    trainer = trainer_class(
        "test",
        _args(output, epochs),
        None,
        model,
        None,
        _Rows(),
        _Rows(),
        optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9),
        checkpoint_measure="overlap",
        d2c_fn=lambda value: value,
        task_config={},
    )

    def train_step(model, data, tokenizer, device):
        del tokenizer, device
        return ((model(torch.stack(data["x"])) - torch.stack(data["y"])) ** 2).mean()

    def eval_step(model, data, seq_processor, tokenizer, device):
        del model, seq_processor, tokenizer, device
        batch_size = len(data["x"])
        boxes = torch.zeros(batch_size, 1, 4)
        return (
            {
                "num_bbox_correct": 1,
                "num_bbox": 1,
                "num_label_correct": 1,
                "num_examples": 1,
                "violation_num": 0,
                "rel_num": 1,
            },
            {
                "pred_bboxes": boxes,
                "gold_bboxes": boxes,
                "gold_labels": torch.ones(batch_size, 1, dtype=torch.long),
                "pred_labels": torch.ones(batch_size, 1, dtype=torch.long),
                "mask": torch.ones(batch_size, 1, dtype=torch.bool),
                "out_str": "",
            },
        )

    trainer(train_step, eval_step)
    state = {
        name: value.detach().clone()
        for name, value in trainer.model.state_dict().items()
    }
    return state, sorted((output / "resume").glob("*.resume.pt"))


def test_vendor_patch_preserves_uninterrupted_and_resumed_cpu_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        Path(__file__).resolve().parents[4]
        / "vendor"
        / "ms-layout-generation"
        / "LayoutFormer++"
        / "src"
    )
    patched = prepare_vendor_tree(source, tmp_path / "vendor-cache")

    original, _ = _run_vendor(source, tmp_path / "original", 2, monkeypatch)
    uninterrupted, _ = _run_vendor(patched, tmp_path / "uninterrupted", 2, monkeypatch)
    _, split_checkpoints = _run_vendor(patched, tmp_path / "split", 1, monkeypatch)
    resumed, _ = _run_vendor(
        patched, tmp_path / "split", 2, monkeypatch, split_checkpoints[-1]
    )

    assert all(torch.equal(original[name], uninterrupted[name]) for name in original)
    assert all(
        torch.equal(uninterrupted[name], resumed[name]) for name in uninterrupted
    )
    assert (tmp_path / "split" / "best_epoch").read_text(encoding="utf-8") == "0\n"
    payload = load_weights_only_checkpoint(
        sorted((tmp_path / "split" / "resume").glob("*.resume.pt"))[-1]
    )
    assert set(payload) >= {
        "model",
        "optimizer",
        "scheduler",
        "epoch",
        "step",
        "rng",
        "sampler_epoch",
        "dataset_generators",
        "identity",
        "extras",
    }


def test_multitask_resume_matches_uninterrupted_evaluation_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        Path(__file__).resolve().parents[4]
        / "vendor"
        / "ms-layout-generation"
        / "LayoutFormer++"
        / "src"
    )
    patched = prepare_vendor_tree(source, tmp_path / "vendor-cache")

    def check_numpy_layouts(gold_layouts, pred_layouts):
        for layouts in (gold_layouts, pred_layouts):
            for boxes, labels in layouts:
                assert isinstance(boxes, np.ndarray)
                assert isinstance(labels, np.ndarray)
        return float(np.mean([boxes[0, 2] for boxes, _ in pred_layouts]))

    def is_checkpoint_mapping(
        value: safe_value_type,
    ) -> TypeGuard[dict[str | int, safe_value_type]]:
        return isinstance(value, dict) and all(
            isinstance(key, (str, int)) for key in value
        )

    def assert_same(left: safe_value_type, right: safe_value_type) -> None:
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor)
            assert torch.equal(left, right)
            return

        if is_checkpoint_mapping(left):
            assert is_checkpoint_mapping(right)
            assert left.keys() == right.keys()
            for key in left:
                assert_same(left[key], right[key])

            return

        if isinstance(left, list):
            assert isinstance(right, list)
            assert len(left) == len(right)
            for left_value, right_value in zip(left, right, strict=True):
                assert_same(left_value, right_value)

            return

        if isinstance(left, tuple):
            assert isinstance(right, tuple)
            assert len(left) == len(right)
            for left_value, right_value in zip(left, right, strict=True):
                assert_same(left_value, right_value)

            return

        assert type(left) is type(right)
        assert left == right

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setenv("WANDB_MODE", "disabled")

    def run(output: Path, epochs: int, resume_path: Path | None = None) -> list[Path]:
        output.mkdir(exist_ok=True)
        if resume_path is not None:
            monkeypatch.setenv("LAYOUTFORMERPP_RESUME_PATH", str(resume_path))
        else:
            monkeypatch.delenv("LAYOUTFORMERPP_RESUME_PATH", raising=False)
        random.seed(31)
        np.random.seed(31)
        torch.manual_seed(31)
        trainer_class = _load_multitask(patched, output.name, check_numpy_layouts)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        trainer = trainer_class(
            "test",
            _args(output, epochs),
            None,
            model,
            None,
            _MultiTaskRows(),
            _MultiTaskRows(),
            optimizer,
            scheduler=None,
            checkpoint_measure="eval_loss",
            d2c_fn=lambda value: value,
            task_config={},
            save_vocab=False,
        )
        train_batches = 0

        def train_step(model, data, tokenizer, device):
            nonlocal train_batches
            del tokenizer, device
            train_batches += 1
            return (
                (model(torch.stack(data["x"])) - torch.stack(data["y"])) ** 2
            ).mean()

        def eval_step(model, data, seq_processor, tokenizer, device):
            del seq_processor, tokenizer, device
            random.random()
            np.random.random()
            torch.rand(())
            epoch = (
                trainer._resume_start_epoch
                + train_batches // len(trainer.train_dataloader)
                - 1
            )
            score = float(epoch + 1)
            batch_size = len(data["x"])
            pred_boxes = torch.tensor([[[0.0, 0.0, score, 1.0]]] * batch_size)
            gold_boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]] * batch_size)
            labels = torch.ones(batch_size, 1, dtype=torch.long)
            mask = torch.ones(batch_size, 1, dtype=torch.bool)
            return torch.tensor(1.0 / score), {
                "pred_bboxes": pred_boxes,
                "gold_bboxes": gold_boxes,
                "gold_labels": labels,
                "pred_labels": labels,
                "gold_mask": mask,
                "pred_mask": mask,
            }

        trainer(train_step, eval_step, ["gen_t"], eval_interval=1)
        return sorted((output / "resume").glob("*.resume.pt"))

    uninterrupted = run(tmp_path / "uninterrupted", 3)
    interrupted = run(tmp_path / "interrupted", 2)
    assert [path.name.split("-", 2)[1] for path in interrupted] == [
        "000000",
        "000001",
    ]
    assert (tmp_path / "interrupted" / "best_epoch").read_text(encoding="utf-8") == "1"
    assert json.loads(
        (tmp_path / "interrupted" / "best_miou.json").read_text(encoding="utf-8")
    ) == {"gen_t": {"value": 2.0, "epoch": 1}}

    interrupted_eval = load_weights_only_checkpoint(interrupted[-1])
    uninterrupted_eval = load_weights_only_checkpoint(uninterrupted[1])
    assert_same(interrupted_eval, uninterrupted_eval)

    resumed = run(tmp_path / "interrupted", 3, interrupted[-1])
    assert (tmp_path / "interrupted" / "best_epoch").read_text(encoding="utf-8") == (
        tmp_path / "uninterrupted" / "best_epoch"
    ).read_text(encoding="utf-8")
    assert (tmp_path / "interrupted" / "best_miou.json").read_text(
        encoding="utf-8"
    ) == (tmp_path / "uninterrupted" / "best_miou.json").read_text(encoding="utf-8")
    assert_same(
        load_weights_only_checkpoint(resumed[-1]),
        load_weights_only_checkpoint(uninterrupted[-1]),
    )


def test_resume_requires_matching_identity_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "LAYOUTFORMERPP_LAUNCH_MANIFEST",
        "LAYOUTFORMERPP_DURABLE_ROOT",
        "LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LAYOUTFORMERPP_CHECKPOINT_IDENTITY", json.dumps({"seed": 1}))
    model = torch.nn.Linear(1, 1)
    trainer = types.SimpleNamespace(
        args=Namespace(out_dir=str(tmp_path)),
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_dataset=_Rows(),
        val_dataset=_Rows(),
    )
    path = save_resume_state(trainer, epoch=0, global_step=0, extras={})
    monkeypatch.setenv("LAYOUTFORMERPP_RESUME_PATH", str(path))
    monkeypatch.setenv("LAYOUTFORMERPP_CHECKPOINT_IDENTITY", json.dumps({"seed": 2}))

    with pytest.raises(RuntimeError, match="identity mismatch"):
        load_resume_state(trainer)


def test_patch_rejects_missing_or_duplicate_source_anchors(tmp_path: Path) -> None:
    vendor = tmp_path / "vendor"
    (vendor / "trainer").mkdir(parents=True)
    basic = vendor / "trainer" / "basic_trainer.py"
    anchor = "from .utils import CheckpointMeasurement\n"
    basic.write_text(anchor + anchor, encoding="utf-8")

    with pytest.raises(RuntimeError, match="exactly once; found 2"):
        _patch_vendor_sources(vendor)


def test_prepare_vendor_tree_rebuilds_a_stale_cached_copy(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).resolve().parents[4]
        / "vendor"
        / "ms-layout-generation"
        / "LayoutFormer++"
        / "src"
    )
    patched = prepare_vendor_tree(source, tmp_path / "vendor-cache")
    target = patched / "trainer" / "multitask_trainer.py"
    target.write_text("stale cache", encoding="utf-8")

    rebuilt = prepare_vendor_tree(source, tmp_path / "vendor-cache")

    assert rebuilt == patched
    assert "stale cache" not in target.read_text(encoding="utf-8")

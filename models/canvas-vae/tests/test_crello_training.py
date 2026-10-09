import json
import runpy
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from canvas_vae.data import (
    CrelloDocument,
    CrelloElement,
    CrelloSplit,
    document_id,
    fixture_image_ids,
    write_embedding_fixture,
    load_crello_vocabularies,
)
from canvas_vae.modeling_canvas_vae import CanvasVAECrelloModel
from canvas_vae.configuration_canvas_vae import CanvasVAECrelloConfig
from canvas_vae.training.datamodule import CanvasVAECrelloDataModule
from canvas_vae.training.lightning_module import CanvasVAECrelloTrainingModule


TYPE_VALUES = [
    "textElement",
    "coloredBackground",
    "imageElement",
    "svgElement",
    "maskElement",
    "otherElement",
]


def test_crello_calibration_threshold_rounds_up_to_two_significant_figures():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    ceil_two_significant_digits = cast(
        Callable[[float], float], namespace["_ceil_two_significant_digits"]
    )

    assert ceil_two_significant_digits(0.001845) == 0.0019
    assert ceil_two_significant_digits(0.0) == 0.0


def make_document(split: CrelloSplit, index: int) -> CrelloDocument:
    kinds = ("textElement", "imageElement", "otherElement")
    elements: list[CrelloElement] = [
        {
            "type": kind,
            "left": 0.1,
            "top": 0.2,
            "width": 0.3,
            "height": 0.4,
            "opacity": 0.8,
            "color": [40, 120, 200],
            "image_id": f"crello-v1/{split}/image-{index}-{element_index}",
        }
        for element_index, kind in enumerate(kinds)
    ]
    return {
        "split": split,
        "document_id": document_id(split, index),
        "context": {
            "id": index,
            "length": len(elements),
            "group": "group",
            "format": "poster",
            "canvas_width": 640,
            "canvas_height": 480,
            "category": "food",
        },
        "elements": elements,
    }


def write_prepared_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    documents = [make_document(split, index) for index, split in enumerate(CrelloSplit)]
    for split, row in zip(CrelloSplit, documents, strict=True):
        (data_dir / f"{split}.jsonl").write_text(json.dumps(row) + "\n")

    vocabulary = {
        "group": {"group": 3},
        "format": {"poster": 3},
        "category": {"food": 3},
        "canvas_width": {"640": 3},
        "canvas_height": {"480": 3},
        "type": {kind: 1 for kind in TYPE_VALUES},
    }
    (data_dir / "vocabulary.json").write_text(json.dumps(vocabulary))

    fixture_dir = tmp_path / "fixture"
    ordered_ids = fixture_image_ids(documents)
    vectors = np.arange(len(ordered_ids) * 256, dtype=np.float32).reshape(-1, 256)
    write_embedding_fixture(
        fixture_dir,
        ordered_ids,
        vectors,
        encoder_state_sha256="a" * 64,
    )
    return data_dir, fixture_dir


def test_crello_data_module_loads_fixture_and_serves_canonical_splits(tmp_path):
    data_dir, fixture_dir = write_prepared_data(tmp_path)
    data = CanvasVAECrelloDataModule(
        data_dir=str(data_dir), fixture_dir=str(fixture_dir), batch_size=2, seed=0
    )
    with pytest.raises(RuntimeError, match="call setup"):
        data.test_dataloader()

    data.setup()
    train = next(iter(data.train_dataloader()))
    validation = next(iter(data.val_dataloader()))
    test = list(data.test_dataloader())
    assert train["num_elements"].tolist() == [3, 3]
    assert validation["document_id"] == [document_id("val", 1)] * 2
    assert [key for batch in test for key in batch["document_id"]] == [
        document_id("test", 2)
    ]


def test_crello_training_step_uses_keras_adam_and_per_tensor_clipping(tmp_path):
    data_dir, fixture_dir = write_prepared_data(tmp_path)
    data = CanvasVAECrelloDataModule(
        data_dir=str(data_dir), fixture_dir=str(fixture_dir), batch_size=2, seed=0
    )
    data.setup()
    batch = next(iter(data.train_dataloader()))
    module = CanvasVAECrelloTrainingModule(
        data_dir=str(data_dir),
        fixture_dir=str(fixture_dir),
        latent_dim=16,
        num_heads=2,
        dropout=0.0,
    )
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    optimizer = module.configure_optimizers()
    assert optimizer.__class__.__name__ == "KerasAdam"
    with pytest.raises(ValueError, match="clip_norm"):
        module.configure_gradient_clipping(optimizer, gradient_clip_val=1.0)


def test_crello_validation_logs_channel_weighted_scores(tmp_path, monkeypatch):
    data_dir, fixture_dir = write_prepared_data(tmp_path)
    data = CanvasVAECrelloDataModule(
        data_dir=str(data_dir), fixture_dir=str(fixture_dir), batch_size=2, seed=0
    )
    data.setup()
    batch = next(iter(data.train_dataloader()))
    module = CanvasVAECrelloTrainingModule(
        data_dir=str(data_dir),
        fixture_dir=str(fixture_dir),
        latent_dim=16,
        num_heads=2,
        dropout=0.0,
    ).eval()
    logged = {}
    monkeypatch.setattr(
        module, "log", lambda name, value, **kwargs: logged.__setitem__(name, value)
    )

    module.on_validation_epoch_end()
    assert not logged
    module.on_validation_epoch_start()
    module.validation_step(batch, 0)

    assert module.validation_counts["color"] == 3 * batch["num_elements"].shape[0]
    module.on_validation_epoch_end()
    assert {
        "val/total_score",
        "val/color_score",
        "val/image_embedding_score",
        "val/layout_acc",
        "val/layout_miou",
        "val/kl_divergence",
    } <= logged.keys()


def test_crello_evaluation_script_reads_saved_model_and_writes_metrics(tmp_path):
    data_dir, fixture_dir = write_prepared_data(tmp_path)
    checkpoint_dir = tmp_path / "model"
    model = CanvasVAECrelloModel(
        CanvasVAECrelloConfig(
            vocabularies=load_crello_vocabularies(data_dir),
            latent_dim=16,
            num_heads=2,
            dropout=0.0,
        )
    )
    model.save_pretrained(checkpoint_dir)
    report_path = tmp_path / "evaluation.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_crello.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint_dir),
            "--data-dir",
            str(data_dir),
            "--fixture-dir",
            str(fixture_dir),
            "--seeds",
            "0",
            "--batch-size",
            "2",
            "--output",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    report = json.loads(report_path.read_text())
    assert "reconst_color" in report
    assert "random_seed0_image_embedding" in report
    assert all(value >= 0 for value in report.values())

import json
import hashlib
import io
import runpy
import subprocess
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
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


def test_crello_parity_report_serializes_numpy_scalars(tmp_path):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    write_json = cast(Callable[..., bytes], namespace["_write_json"])

    path = tmp_path / "report.json"
    write_json(path, {"vendor_value": np.float32(0.5)})

    assert json.loads(path.read_text(encoding="utf-8")) == {"vendor_value": 0.5}


def test_crello_calibration_uses_three_fresh_processes_before_freezing_limits(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    calibrate = cast(Callable[..., dict[str, object]], namespace["_calibrate"])
    original_run = subprocess.run
    commands: list[list[str]] = []
    values = (0.1, 0.3, 0.2)
    plan_visible_at_launch: list[bool] = []

    def run(command, **kwargs):
        if len(command) > 1 and command[1] == str(script):
            commands.append(command)
            batch_index = int(command[command.index("--batch-index") + 1])
            report_dir = Path(command[command.index("--report-dir") + 1])
            plan_path = report_dir / "calibration-plan.json"
            plan_visible_at_launch.append(plan_path.is_file())
            child = {
                "static_errors": [],
                "batch_digests": [
                    hashlib.sha256(f"batch-{batch_index}".encode()).hexdigest()
                ],
                "initial_state_sha256": "same-initial-state",
                "initial_state_mismatch_keys": [],
                "measurements": [
                    {
                        "group": "s1_eval_outputs",
                        "name": "output",
                        "batch": batch_index,
                        "metric": "max_rel_to_max",
                        "value": values[batch_index],
                        "actual_shape": [1],
                        "expected_shape": [1],
                        "shape_match": True,
                    }
                ],
            }
            (report_dir / f"calibration-batch-{batch_index}.json").write_text(
                json.dumps(child), encoding="utf-8"
            )
            return SimpleNamespace(returncode=0)

        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    report = calibrate(report_dir=tmp_path)

    assert [
        int(command[command.index("--batch-index") + 1]) for command in commands
    ] == [0, 1, 2]
    assert plan_visible_at_launch == [True, True, True]
    assert all(command[1] == str(script) for command in commands)
    plan_bytes = (tmp_path / "calibration-plan.json").read_bytes()
    assert report["calibration_plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
    assert report["distinct_input_batches"] is True
    limits = cast(dict[str, dict[str, object]], report["limits"])
    limit = limits["s1_eval_outputs"]
    assert cast(list[float], limit["calibration_batch_maxima"]) == list(values)
    assert limit["max_calibration_error"] == 0.3
    assert limit["limit"] == 0.45
    assert (tmp_path / "limits.json").is_file()


def test_crello_metric_comparison_records_color_total_and_layout_scores():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    compare_metrics = cast(Callable[..., None], namespace["_compare_metric_mappings"])
    records = []

    compare_metrics(
        records,
        "s1_reconstruction_metrics",
        4,
        {
            "color": torch.tensor([[0.2, 0.5, 0.8]]),
            "total": torch.tensor([[0.6]]),
        },
        {
            "color": np.array([[0.2, 0.5, 0.8]], dtype=np.float32),
            "total": np.array([[0.6]], dtype=np.float32),
        },
    )
    compare_metrics(
        records,
        "s1_layout_metrics",
        4,
        {"layout_acc": torch.tensor([0.7]), "layout_miou": torch.tensor([0.4])},
        {
            "layout_acc": np.array([0.7], dtype=np.float32),
            "layout_miou": np.array([0.4], dtype=np.float32),
        },
    )

    assert {(row.group, row.name) for row in records} == {
        ("s1_reconstruction_metrics/color", "color"),
        ("s1_reconstruction_metrics/total", "total"),
        ("s1_layout_metrics/layout_acc", "layout_acc"),
        ("s1_layout_metrics/layout_miou", "layout_miou"),
    }
    assert next(row for row in records if row.name == "color").actual_shape == [1, 3]
    assert all(row.shape_match and row.value == 0.0 for row in records)


def test_crello_hub_location_parser_accepts_model_and_dataset_resolve_urls():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    parse_hub_location = cast(
        Callable[[str], tuple[str, str | None, str, str]],
        namespace["_parse_hub_location"],
    )

    assert parse_hub_location(
        "https://huggingface.co/owner/model/resolve/main/path/fixture.tar"
    ) == ("owner/model", None, "main", "path/fixture.tar")
    assert parse_hub_location(
        "https://huggingface.co/datasets/owner/data/resolve/main/fixture.tar"
    ) == ("owner/data", "dataset", "main", "fixture.tar")
    with pytest.raises(ValueError, match="HTTPS Hugging Face resolve URL"):
        parse_hub_location("http://huggingface.co/owner/model/resolve/main/fixture.tar")


def test_crello_fixture_archive_extracts_only_expected_root_files(tmp_path):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    extract_fixture_archive = cast(
        Callable[[Path, Path], None], namespace["_extract_fixture_archive"]
    )
    archive = tmp_path / "fixture.tar"
    entries = {
        "manifest.json": b"{}\n",
        "manifest.sha256": b"fixture-manifest-hash\n",
        "posterior_means.npy": b"fixture-array",
    }
    with tarfile.open(archive, mode="w") as bundle:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))

    fixture_dir = tmp_path / "fixture"
    extract_fixture_archive(archive, fixture_dir)
    assert {path.name: path.read_bytes() for path in fixture_dir.iterdir()} == entries

    unsafe_archive = tmp_path / "unsafe.tar"
    with tarfile.open(unsafe_archive, mode="w") as bundle:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        bundle.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe path"):
        extract_fixture_archive(unsafe_archive, tmp_path / "unsafe")


def test_crello_fixture_download_checks_hash_and_deletes_token_file(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    download_fixture_archive = cast(
        Callable[[str, str, Path, Path | None], None],
        namespace["_download_fixture_archive"],
    )
    cached_file = tmp_path / "cached.tar"
    cached_file.write_bytes(b"fixture archive")
    token_file = tmp_path / "hf-token"
    token_file.write_text("hf_private_token\n")
    token_file.chmod(0o600)

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda **_: str(cached_file)
    )
    destination = tmp_path / "download.tar"
    digest = hashlib.sha256(cached_file.read_bytes()).hexdigest()
    download_fixture_archive(
        "https://huggingface.co/owner/model/resolve/main/fixture.tar",
        digest,
        destination,
        token_file,
    )

    assert destination.read_bytes() == cached_file.read_bytes()
    assert not token_file.exists()

    bad_token_file = tmp_path / "bad-token"
    bad_token_file.write_text("hf_private_token\n")
    bad_token_file.chmod(0o644)
    with pytest.raises(PermissionError, match="mode 0600"):
        download_fixture_archive(
            "https://huggingface.co/owner/model/resolve/main/fixture.tar",
            digest,
            destination,
            bad_token_file,
        )


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

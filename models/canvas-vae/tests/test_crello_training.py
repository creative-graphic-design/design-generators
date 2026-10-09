import json
import hashlib
import runpy
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FunctionType, SimpleNamespace
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
    load_embedding_fixture,
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


def test_crello_heldout_batches_follow_sequential_index_batches():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    heldout_batches = cast(
        Callable[[Sequence[CrelloDocument]], list[tuple[int, list[CrelloDocument]]]],
        namespace["_heldout_test_batches"],
    )
    documents: list[CrelloDocument] = [
        {
            "split": CrelloSplit.test.value,
            "document_id": f"crello-v1/test/{index}",
            "context": {},
            "elements": [],
        }
        for index in range(1025)
    ]

    batches = heldout_batches(documents)

    assert [index for index, _ in batches] == [0, 1]
    assert [len(rows) for _, rows in batches] == [1024, 1]
    assert batches[0][1] == documents[:-1]
    assert batches[1][1] == documents[-1:]


def test_crello_heldout_batches_start_in_fresh_processes_from_frozen_limits(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    heldout_function = cast(FunctionType, namespace["_heldout"])
    heldout = cast(Callable[..., dict[str, object]], heldout_function)
    parity_globals = heldout_function.__globals__
    documents: list[CrelloDocument] = [
        {
            "split": CrelloSplit.test.value,
            "document_id": f"crello-v1/test/{index}",
            "context": {},
            "elements": [],
        }
        for index in range(1025)
    ]
    monkeypatch.setitem(parity_globals, "load_crello_split", lambda *_: documents)
    monkeypatch.setitem(
        parity_globals,
        "_cpu_report_context",
        lambda: {"commit": "test-commit", "device": "cpu"},
    )
    calibration_bytes = json.dumps(
        {"initial_state_sha256": "same-initial-state"}
    ).encode()
    (tmp_path / "calibrate.json").write_bytes(calibration_bytes)
    limits = {
        "commit": "test-commit",
        "calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "selection": "three distinct batches",
        "formula": "test formula",
        "limits": {
            "s1_eval_outputs": {
                "metric": "max_rel_to_max",
                "L": 0.0,
                "calibration_batch_maxima": [0.1, 0.1, 0.1],
                "max_calibration_error": 0.1,
                "limit": 1.0,
            }
        },
    }
    limits_bytes = json.dumps(limits).encode()
    (tmp_path / "limits.json").write_bytes(limits_bytes)
    (tmp_path / "limits.sha256").write_text(
        hashlib.sha256(limits_bytes).hexdigest(), encoding="ascii"
    )

    commands: list[list[str]] = []
    plan_visible_at_launch: list[bool] = []

    def run(command, **kwargs):
        assert command[1] == str(script)
        assert command[2] == "heldout-batch"
        commands.append(command)
        batch_index = int(command[command.index("--batch-index") + 1])
        report_dir = Path(command[command.index("--report-dir") + 1])
        plan_visible_at_launch.append((report_dir / "heldout-plan.json").is_file())
        child = {
            "static_errors": [],
            "batch_digests": [
                hashlib.sha256(f"test-{batch_index}".encode()).hexdigest()
            ],
            "initial_state_sha256": "same-initial-state",
            "initial_state_mismatch_keys": [],
            "measurements": [
                {
                    "group": "s1_eval_outputs",
                    "name": "output",
                    "batch": batch_index,
                    "metric": "max_rel_to_max",
                    "value": 0.1,
                    "actual_shape": [1],
                    "expected_shape": [1],
                    "shape_match": True,
                }
            ],
        }
        (report_dir / f"heldout-batch-{batch_index}.json").write_text(
            json.dumps(child), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    report = heldout(report_dir=tmp_path)

    assert [
        int(command[command.index("--batch-index") + 1]) for command in commands
    ] == [
        0,
        1,
    ]
    assert plan_visible_at_launch == [True, True]
    assert report["initial_state_sha256"] == "same-initial-state"
    assert report["heldout_errors"] == []
    assert (tmp_path / "heldout.json").is_file()


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


def test_crello_metric_diagnostic_records_bleu_steps_and_float64_scores():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    diagnose_bleu = cast(
        Callable[..., dict[str, object]], namespace["_bleu_diagnostic"]
    )
    target_ids = np.array([[[1], [2], [2], [0]]], dtype=np.int64)
    target_mask = np.array([[True, True, True, False]])
    predicted_ids = np.array([[[2], [3], [0], [0], [0], [0]]], dtype=np.int64)
    predicted_mask = np.array([[True, True, False, False, False, False]])

    report = diagnose_bleu(target_ids, target_mask, predicted_ids, predicted_mask, 4)
    float32 = cast(dict[str, list[list[float]]], report["float32_intermediates"])

    assert float32["target_token_count"] == [[3.0]]
    assert float32["predicted_token_count"] == [[2.0]]
    assert float32["matching_token_count"] == [[1.0]]
    assert float32["precision"] == [[0.5]]
    assert float32["clipped_score"][0][0] < 1.0
    assert cast(list[list[float]], report["float64_scores"])[0][0] < 1.0

    color_report = diagnose_bleu(
        np.repeat(target_ids, 3, axis=2),
        target_mask,
        np.repeat(predicted_ids, 3, axis=2),
        predicted_mask,
        4,
    )
    color_intermediates = cast(
        dict[str, list[list[float]]], color_report["float32_intermediates"]
    )
    assert (
        color_intermediates["clipped_score"][0] == [float32["clipped_score"][0][0]] * 3
    )


def test_crello_metric_diagnostic_recomputes_scaled_cosine_in_float64():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    diagnose_cosine = cast(
        Callable[..., dict[str, object]], namespace["_scaled_cosine_diagnostic"]
    )
    report = diagnose_cosine(
        np.array([[[1.0, 0.0], [99.0, 99.0]]], dtype=np.float32),
        np.array([[True, False]]),
        np.array([[[-1.0, 0.0], [99.0, 99.0], [99.0, 99.0]]], dtype=np.float32),
        np.array([[True, False, False]]),
    )

    assert report["target_count"] == [1.0]
    assert report["prediction_count"] == [1.0]
    assert report["target_mean_l2_norm"] == pytest.approx([1.0])
    assert report["prediction_mean_l2_norm"] == pytest.approx([1.0])
    assert report["tensorflow_cosine_similarity"] == pytest.approx([-1.0])
    assert report["torch_cosine_similarity"] == pytest.approx([-1.0])
    assert report["length_penalty"] == [1.0]
    assert report["tensorflow_score"] == [0.0]
    assert report["torch_score"] == [0.0]


def test_crello_metric_diagnostic_reports_signed_float32_ulp_delta():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    ulp_delta = cast(Callable[..., np.ndarray], namespace["_float32_ulp_delta"])
    expected = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)
    actual = np.array(
        [
            [
                np.nextafter(np.float32(1.0), np.float32(2.0)),
                np.nextafter(np.float32(0.5), np.float32(1.0)),
                0.0,
            ]
        ],
        dtype=np.float32,
    )

    assert ulp_delta(actual, expected).tolist() == [[1, 1, 0]]
    assert ulp_delta(expected, actual).tolist() == [[-1, -1, 0]]


def test_crello_hub_location_parser_accepts_model_and_dataset_tree_urls():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    parse_hub_location = cast(
        Callable[[str], tuple[str, str | None, str, str]],
        namespace["_parse_hub_location"],
    )

    assert parse_hub_location(
        "https://huggingface.co/owner/model/tree/main/path/fixture"
    ) == ("owner/model", None, "main", "path/fixture")
    assert parse_hub_location(
        "https://huggingface.co/datasets/owner/data/tree/main/fixture"
    ) == ("owner/data", "dataset", "main", "fixture")
    with pytest.raises(ValueError, match="HTTPS Hugging Face tree URL"):
        parse_hub_location("http://huggingface.co/owner/model/tree/main/fixture")


def test_crello_fixture_download_verifies_hashes_and_loads_manifest_ids(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    download_fixture = cast(
        Callable[[str, str, str, Path, Path | None], None],
        namespace["_download_embedding_fixture"],
    )
    source_dir = tmp_path / "source"
    image_ids = ["crello-v1/train/first", "crello-v1/val/second"]
    values = np.arange(512, dtype=np.float32).reshape(2, 256)
    write_embedding_fixture(
        source_dir,
        image_ids,
        values,
        encoder_state_sha256="a" * 64,
    )
    cached_files = {
        name: source_dir / name
        for name in ("manifest.json", "manifest.sha256", "posterior_means.npy")
    }
    token_file = tmp_path / "hf-token"
    token_file.write_text("hf_private_token\n")
    token_file.chmod(0o600)

    import huggingface_hub

    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        filename = cast(str, kwargs["filename"])
        return str(cached_files[Path(filename).name])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    manifest_sha256 = hashlib.sha256(
        cached_files["manifest.json"].read_bytes()
    ).hexdigest()
    array_sha256 = hashlib.sha256(
        cached_files["posterior_means.npy"].read_bytes()
    ).hexdigest()
    fixture_dir = tmp_path / "fixture"
    download_fixture(
        "https://huggingface.co/owner/model/tree/main/path/fixture",
        array_sha256,
        manifest_sha256,
        fixture_dir,
        token_file,
    )

    assert not token_file.exists()
    assert [Path(cast(str, call["filename"])).name for call in calls] == [
        "manifest.json",
        "manifest.sha256",
        "posterior_means.npy",
    ]
    loaded = load_embedding_fixture(fixture_dir)
    assert list(loaded) == image_ids
    assert all(
        np.array_equal(loaded[key], values[index])
        for index, key in enumerate(image_ids)
    )

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        download_fixture(
            "https://huggingface.co/owner/model/tree/main/path/fixture",
            array_sha256,
            "0" * 64,
            tmp_path / "bad-manifest",
            None,
        )

    with pytest.raises(ValueError, match="array SHA-256 mismatch"):
        download_fixture(
            "https://huggingface.co/owner/model/tree/main/path/fixture",
            "0" * 64,
            manifest_sha256,
            tmp_path / "bad-array",
            None,
        )

    bad_token_file = tmp_path / "bad-token"
    bad_token_file.write_text("hf_private_token\n")
    bad_token_file.chmod(0o644)
    with pytest.raises(PermissionError, match="mode 0600"):
        download_fixture(
            "https://huggingface.co/owner/model/tree/main/path/fixture",
            array_sha256,
            manifest_sha256,
            tmp_path / "bad-token",
            bad_token_file,
        )
    assert not bad_token_file.exists()


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

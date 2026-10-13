from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def load_runner() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "tests"
        / "vendor_parity"
        / "run_condition_parity.py"
    )
    spec = importlib.util.spec_from_file_location("ralf_condition_parity_runner", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_freeze_rejects_interpreter_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner_path = Path(__file__).parents[1] / "tests" / "vendor_parity"
    monkeypatch.syspath_prepend(str(runner_path))
    runner = load_runner()
    expected_path = tmp_path / "runtime-freeze.txt"
    expected_path.write_bytes(b"package==expected\n")
    runtime_python = tmp_path / "venv" / "bin" / "python"

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        assert args[0] == [
            "uv",
            "pip",
            "freeze",
            "--python",
            str(runtime_python),
        ]
        assert kwargs == {"check": True, "capture_output": True}
        return SimpleNamespace(stdout=b"package==wrong\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="refusing to launch"):
        runner.runtime_freeze(tmp_path / "output", runtime_python, expected_path)


def test_runtime_freeze_accepts_normalized_stored_freeze(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner_path = tmp_path / "a" / "b" / "c" / "d" / "runner.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner := load_runner(), "__file__", str(runner_path))
    expected_path = tmp_path / "runtime-freeze.txt"
    expected_path.write_bytes(b"-e <repo root>/lib/laygen\npackage==same\n")
    editable = tmp_path / "lib" / "laygen"
    editable.mkdir(parents=True)
    runtime_python = tmp_path / "venv" / "bin" / "python"

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        assert args[0] == [
            "uv",
            "pip",
            "freeze",
            "--python",
            str(runtime_python),
        ]
        assert kwargs == {"check": True, "capture_output": True}
        return SimpleNamespace(
            stdout=f"-e {editable.as_uri()}\npackage==same\n".encode()
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.runtime_freeze(tmp_path / "output", runtime_python, expected_path)

    assert result["sha256"] == runner.sha256(expected_path)


def test_runtime_freeze_normalizes_local_urls(tmp_path: Path) -> None:
    runner = load_runner()
    repo_root = tmp_path / "repo"
    editable = (repo_root / "lib" / "laygen").as_uri()
    freeze = (
        f"-e {editable}\n"
        "torch @ file:///tmp/torch-2.8.0%2Bcu128-cp311-cp311.whl\n"
        "torchvision==0.23.0+cu128\n"
    ).encode()

    normalized = runner.normalize_runtime_freeze(freeze, repo_root)

    assert normalized == (
        b"-e <repo root>/lib/laygen\ntorch==2.8.0+cu128\ntorchvision==0.23.0+cu128\n"
    )
    assert runner.normalize_runtime_freeze(normalized, repo_root) == normalized


def test_runtime_preflight_rejects_mismatched_cuda_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner()
    distributions = [
        SimpleNamespace(metadata={"Name": "nvidia-cudnn-cu13"}, version="9.20"),
        SimpleNamespace(metadata={"Name": "nvidia-cuda-runtime"}, version="13.0"),
    ]
    monkeypatch.setattr(
        runner.importlib.metadata, "distributions", lambda: distributions
    )
    monkeypatch.setattr(runner.torch.version, "cuda", "12.8")

    with pytest.raises(RuntimeError, match="does not match"):
        runner.installed_cuda_package_family()


def test_full_test_counts_are_dataset_specific() -> None:
    runner = load_runner()

    assert runner.TEST_COUNTS == {"cgl": 6002, "pku": 1000}


def test_vendor_job_mapping_uses_vendor_config_and_original_checkpoint(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    campaign_root = tmp_path / "campaign"
    package_job = campaign_root / "s5" / "evals" / "package-seed-1"
    vendor_job = campaign_root / "s5" / "evals" / "vendor-seed-1"
    package_job.mkdir(parents=True)
    vendor_job.mkdir(parents=True)
    (package_job / "config.yaml").write_text(
        "generator:\n  _target_: ralf.train.models.PackageModel\n",
        encoding="utf-8",
    )
    (vendor_job / "config.yaml").write_text(
        "generator:\n  _target_: image2layout.train.models.VendorModel\n",
        encoding="utf-8",
    )
    checkpoint = package_job / "gen_final_model.pt"
    checkpoint.write_bytes(b"unchanged checkpoint")
    args = SimpleNamespace(
        campaign_root=campaign_root,
        checkpoint_path=package_job,
        checkpoint_artifact_path="campaign/s5/evals/package-seed-1/gen_final_model.pt",
    )

    assert runner.campaign_vendor_job(args) == vendor_job
    record = runner.materialize_vendor_job(args, tmp_path / "output")

    assert record["source_job_dir"] == "s5/evals/vendor-seed-1"
    assert record["used_job_dir"] == "vendor-job"
    assert (
        (tmp_path / "output" / "vendor-job" / "config.yaml")
        .read_text(encoding="utf-8")
        .startswith("generator:")
    )
    assert (tmp_path / "output" / "vendor-job" / "gen_final_model.pt").read_bytes() == (
        checkpoint.read_bytes()
    )


def test_vendor_job_mapping_rejects_package_namespace(tmp_path: Path) -> None:
    runner = load_runner()
    vendor_job = tmp_path / "campaign" / "s5" / "evals" / "vendor-seed-1"
    vendor_job.mkdir(parents=True)
    (vendor_job / "config.yaml").write_text(
        "generator:\n  _target_: ralf.train.models.PackageModel\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="vendor namespace"):
        runner.vendor_model_target(vendor_job)


def test_vendor_dataset_overrides_keep_configured_path(tmp_path: Path) -> None:
    runner = load_runner()
    input_root = tmp_path / "input"
    configured_path = tmp_path / "recorded" / "input"
    input_root.mkdir()

    assert runner.vendor_dataset_overrides(
        config={"dataset": {"path": str(configured_path)}},
        input_root=input_root,
    ) == (f"dataset_path={input_root}",)


def test_vendor_dataset_overrides_supply_nested_path_for_missing_config_key(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    input_root = tmp_path / "input"
    input_root.mkdir()

    assert runner.vendor_dataset_overrides(
        config={"dataset": {"name": "cgl"}},
        input_root=input_root,
    ) == (f"dataset_path={input_root}", f"dataset.path={input_root}")


def test_vendor_dataset_overrides_fail_without_config_or_input(tmp_path: Path) -> None:
    runner = load_runner()

    with pytest.raises(RuntimeError, match="prepared input directory"):
        runner.vendor_dataset_overrides(
            config={},
            input_root=tmp_path / "missing",
        )


def test_evaluator_source_prefers_campaign_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner()
    source = tmp_path / "campaign" / "s5" / "eval-vendor-source"
    source.mkdir(parents=True)
    (source / "eval.py").write_text("print('campaign')\n", encoding="utf-8")
    args = SimpleNamespace(campaign_root=tmp_path / "campaign", vendor_source_path=None)
    monkeypatch.setattr(runner, "evaluator_source_commit", lambda _: "a" * 40)

    assert runner.vendor_source(args) == source
    record = runner.evaluator_source_record(args)
    assert record["kind"] == "campaign eval-vendor-source"
    assert record["path"] == "campaign/s5/eval-vendor-source"
    assert record["sha256"] == runner.sha256(source / "eval.py")


def test_evaluator_source_falls_back_to_pinned_vendor(tmp_path: Path) -> None:
    runner = load_runner()
    args = SimpleNamespace(campaign_root=tmp_path / "campaign", vendor_source_path=None)

    result = runner.evaluator_source_record(args)

    assert result["kind"] == "tracked vendor/ralf"
    assert result["path"] == "vendor/ralf"
    assert result["sha256"] == runner.sha256(
        Path(__file__).parents[3] / "vendor" / "ralf" / "eval.py"
    )
    assert len(result["vendor_revision"]) == 40


def test_evaluator_source_fails_closed_without_either_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner()
    monkeypatch.setattr(
        runner,
        "tracked_vendor_source",
        lambda: (_ for _ in ()).throw(
            FileNotFoundError("missing tracked vendor evaluator")
        ),
    )
    args = SimpleNamespace(campaign_root=tmp_path / "campaign", vendor_source_path=None)

    with pytest.raises(FileNotFoundError, match="tracked vendor evaluator"):
        runner.vendor_source(args)


def test_condition_preflight_accepts_matching_job_and_campaign_metadata(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    campaign_root = tmp_path / "campaign"
    vendor_job = campaign_root / "s5" / "vendor-seed-1"
    vendor_job.mkdir(parents=True)
    (vendor_job / "config.yaml").write_text(
        "generator:\n"
        "  _target_: image2layout.train.models.VendorModel\n"
        "  auxilary_task: uncond\n",
        encoding="utf-8",
    )
    (campaign_root / "s5" / "evaluation-ledger.txt").write_text(
        "cond_type=uncond\n", encoding="utf-8"
    )
    args = SimpleNamespace(
        campaign_root=campaign_root,
        condition="unconditional",
    )

    result = runner.validate_condition_job(args=args, vendor_job=vendor_job)

    assert result["expected_vendor_condition"] == "uncond"
    assert result["job_config"] == {"auxilary_task": "uncond"}
    assert result["campaign_evaluation_metadata"] == {"campaign_metadata": "uncond"}


def test_condition_preflight_rejects_mismatched_job_and_metadata(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    campaign_root = tmp_path / "campaign"
    vendor_job = campaign_root / "s5" / "vendor-seed-1"
    vendor_job.mkdir(parents=True)
    (vendor_job / "config.yaml").write_text(
        "generator:\n"
        "  _target_: image2layout.train.models.VendorModel\n"
        "  auxilary_task: partial\n",
        encoding="utf-8",
    )
    (campaign_root / "s5" / "evaluation-ledger.txt").write_text(
        "cond_type=partial\n", encoding="utf-8"
    )
    args = SimpleNamespace(
        campaign_root=campaign_root,
        condition="unconditional",
    )

    with pytest.raises(RuntimeError, match="expected auxilary_task/cond_type='uncond'"):
        runner.validate_condition_job(args=args, vendor_job=vendor_job)


def test_localized_replay_restores_the_recorded_pre_batch_state() -> None:
    runner = load_runner()
    generator = torch.Generator(device="cpu").manual_seed(17)
    torch.rand(11, generator=generator)
    pre_batch_state = runner.capture_pre_batch_generator_state(generator)
    expected = torch.rand(8, generator=generator)
    expected_after = generator.get_state().clone()

    generator.manual_seed(999)
    replayed = runner.replay_batch_from_state(
        generator,
        pre_batch_state,
        lambda active_generator: torch.rand(8, generator=active_generator),
    )

    assert torch.equal(replayed, expected)
    assert torch.equal(generator.get_state(), expected_after)


def test_relation_comparison_records_unavoidable_difference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = load_runner()
    monkeypatch.setattr(runner, "INFERENCE_SEEDS", (0,))
    package_scores = tmp_path / "package.yaml"
    vendor_scores = tmp_path / "vendor.yaml"
    score = "average:\n  test:\n    metric: 1.0\n"
    package_scores.write_text(score, encoding="utf-8")
    vendor_scores.write_text(score, encoding="utf-8")

    result = runner.compare_predictions(
        {0: [{"label": [1]}]},
        {0: [{"label": [2]}]},
        package_scores,
        vendor_scores,
        allow_relation_difference=True,
    )

    assert result["status"] == "measured_relation_difference"
    assert result["prediction_mismatch_count"] == 1

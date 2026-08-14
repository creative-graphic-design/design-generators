"""Unit tests for the fail-closed S2 evidence comparisons."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from run_training_stages import (
    _compare_learning_rates,
    _compare_named_gradients,
    _compare_state_dicts,
    _fresh_s3_run_root,
    _natural_run_envelope,
    _run_s3_fit,
    _s3,
    _s4,
    RalfS3TraceCallback,
)


def _linear_pair() -> tuple[torch.nn.Linear, torch.nn.Linear]:
    package = torch.nn.Linear(2, 2)
    vendor = torch.nn.Linear(2, 2)
    vendor.load_state_dict(package.state_dict())
    return package, vendor


def test_named_gradient_comparison_requires_per_parameter_coverage() -> None:
    package, vendor = _linear_pair()
    inputs = torch.tensor([[1.0, 2.0]])
    package(inputs).sum().backward()
    vendor(inputs).sum().backward()

    result = _compare_named_gradients(package, vendor, "raw_gradients")

    assert result["first_divergence"] is None
    assert result["max_abs_diff"] == 0.0
    assert result["package"]["named_parameter_count"] == 2
    assert result["package"]["present_count"] == 2
    assert result["package"]["presence_digest"] == result["vendor"]["presence_digest"]


def test_named_gradient_comparison_reports_first_parameter_and_max_abs() -> None:
    package, vendor = _linear_pair()
    package.weight.grad = torch.ones_like(package.weight)
    package.bias.grad = torch.ones_like(package.bias)
    vendor.weight.grad = torch.zeros_like(vendor.weight)
    vendor.bias.grad = torch.ones_like(vendor.bias)

    with pytest.raises(
        RuntimeError, match=r"first divergence at raw_gradients\.weight"
    ) as exc_info:
        _compare_named_gradients(package, vendor, "raw_gradients")

    assert "aggregate max_abs_diff=1" in str(exc_info.value)


def test_learning_rate_comparison_reports_both_optimizer_values() -> None:
    package, vendor = _linear_pair()
    package_optimizer = torch.optim.AdamW(package.parameters(), lr=1e-4)
    vendor_optimizer = torch.optim.AdamW(vendor.parameters(), lr=1e-4)

    result = _compare_learning_rates(package_optimizer, vendor_optimizer)

    assert result["first_divergence"] is None
    assert result["max_abs_diff"] == 0.0
    assert result["groups"] == [
        {"index": 0, "package": 1e-4, "vendor": 1e-4, "abs_diff": 0.0}
    ]


def test_state_comparison_can_record_natural_drift_without_hiding_it() -> None:
    package_state = {"weight": torch.ones(2)}
    vendor_state = {"weight": torch.zeros(2)}

    result = _compare_state_dicts(
        package_state,
        vendor_state,
        "natural.parameters",
        enforce=False,
    )

    assert result == {
        "first_divergence": "weight",
        "max_abs_diff": 1.0,
    }

    with pytest.raises(RuntimeError, match=r"first divergence at strict\.weight"):
        _compare_state_dicts(package_state, vendor_state, "strict")


def test_natural_run_envelope_reports_run_to_run_drift() -> None:
    first = {
        "trajectory": [
            {
                "global_step": 1,
                "loss": {"package": 2.0, "vendor": 2.1},
                "raw_gradient_norm": {"package": 3.0, "vendor": 3.1},
                "clipped_gradient_norm": {"package": 0.5, "vendor": 0.6},
                "package_state_sha256": "a",
            }
        ]
    }
    second = {
        "trajectory": [
            {
                "global_step": 1,
                "loss": {"package": 2.25, "vendor": 2.2},
                "raw_gradient_norm": {"package": 3.5, "vendor": 3.3},
                "clipped_gradient_norm": {"package": 0.75, "vendor": 0.8},
                "package_state_sha256": "b",
            }
        ]
    }

    result = _natural_run_envelope(first, second)

    assert result["run_count"] == 2
    assert result["step_count"] == 1
    max_abs_diff = result["max_abs_diff"]
    assert max_abs_diff["loss"]["package"] == pytest.approx(0.25)
    assert max_abs_diff["loss"]["vendor"] == pytest.approx(0.1)
    assert max_abs_diff["raw_gradient_norm"]["package"] == pytest.approx(0.5)
    assert max_abs_diff["raw_gradient_norm"]["vendor"] == pytest.approx(0.2)
    assert max_abs_diff["clipped_gradient_norm"]["package"] == pytest.approx(0.25)
    assert max_abs_diff["clipped_gradient_norm"]["vendor"] == pytest.approx(0.2)
    assert result["first_package_state_hash_divergence_step"] == 1
    assert result["package_state_hashes_equal"] is False


def test_s3_uses_production_trainer_and_has_no_manual_scheduler_sentinel() -> None:
    source = inspect.getsource(_s3) + inspect.getsource(_run_s3_fit)

    assert "traingen fit" in source
    assert "RalfTrainingModule" in source
    assert "RalfDataModule" in source
    assert '"scheduler_last_epoch": 0' not in source
    assert "natural_run_to_run_envelope" in source
    assert "state_synchronized_lockstep" in source
    assert "--trainer.limit_train_batches=" in source
    assert "--trainer.limit_val_batches=" in source
    assert "train_limit" in source
    assert "validation_limit" in source


def test_s3_does_not_disable_deterministic_training() -> None:
    source = inspect.getsource(_s3) + inspect.getsource(_run_s3_fit)

    assert "--trainer.deterministic=warn" in source
    assert "--trainer.deterministic=true" not in source
    assert "--trainer.deterministic=false" not in source


def test_training_configs_use_lightning_warning_mode() -> None:
    root = Path(__file__).parents[4]
    for dataset in ("cgl", "pku"):
        source = (
            root / "models" / "ralf" / "configs" / "training" / f"{dataset}.yaml"
        ).read_text()
        assert "deterministic: warn" in source
        assert "deterministic: true" not in source


def test_s3_requires_effective_deterministic_warning_state() -> None:
    source = inspect.getsource(RalfS3TraceCallback.on_fit_start)

    assert "torch.are_deterministic_algorithms_enabled()" in source
    assert "torch.is_deterministic_algorithms_warn_only_enabled()" in source
    assert "lightning_trainer.deterministic" not in source


def test_s3_reference_backward_restores_package_rng_state() -> None:
    source = inspect.getsource(RalfS3TraceCallback.on_after_backward)

    assert "package_rng_after_backward" in source
    assert "restore_rng_state(package_rng_after_backward)" in source


def test_s3_optimizer_state_sync_separates_tensor_storage(tmp_path: Path) -> None:
    sync_source = inspect.getsource(
        RalfS3TraceCallback.on_train_batch_start
    ) + inspect.getsource(RalfS3TraceCallback.on_train_batch_end)
    assert sync_source.count("copy.deepcopy(package_optimizer.state_dict())") == 2

    package, vendor = _linear_pair()
    package_optimizer = torch.optim.AdamW(package.parameters(), lr=1e-4)
    vendor_optimizer = torch.optim.AdamW(vendor.parameters(), lr=1e-4)
    package(torch.ones(1, 2)).sum().backward()
    package_optimizer.step()

    callback = RalfS3TraceCallback(
        cache_dir=str(tmp_path), output_dir=str(tmp_path), seed=1
    )
    callback.synchronized = True
    callback.vendor_model = vendor
    callback.vendor_optimizer = vendor_optimizer
    callback.package_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        package_optimizer, milestones=[49]
    )
    callback.vendor_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        vendor_optimizer, milestones=[49]
    )
    callback.optimizer_step_count = 1

    callback.on_train_batch_start(
        SimpleNamespace(
            optimizers=[package_optimizer],
            current_epoch=0,
        ),
        SimpleNamespace(model=package),
        object(),
        0,
    )

    for package_parameter, vendor_parameter in zip(
        package_optimizer.state, vendor_optimizer.state, strict=True
    ):
        package_state = package_optimizer.state[package_parameter]
        vendor_state = vendor_optimizer.state[vendor_parameter]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert package_state[key] is not vendor_state[key]
            assert package_state[key].data_ptr() != vendor_state[key].data_ptr()


def test_s3_run_preserves_prior_artifacts(tmp_path: Path) -> None:
    first = _fresh_s3_run_root(tmp_path)
    first.mkdir(parents=True)
    second = _fresh_s3_run_root(tmp_path)

    assert first != second
    assert first.name == "run-001"
    assert second.name == "run-002"


def test_s4_records_authoritative_train_validation_stream_evidence() -> None:
    source = inspect.getsource(_s4)

    for required_surface in (
        "train_dataloader()",
        "val_dataloader()",
        '"split_membership"',
        '"serialized_sha256"',
        '"package_stream_sha256"',
        '"vendor_stream_sha256"',
    ):
        assert required_surface in source


def test_s4_points_vendor_retrieval_cache_at_authoritative_cache() -> None:
    source = inspect.getsource(_s4)

    assert "image2layout.train.global_variables" in source
    assert "PRECOMPUTED_WEIGHT_DIR" in source


def test_s4_uses_existing_vendor_legacy_torch_load_mode() -> None:
    source = inspect.getsource(_s4)

    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in source


def test_s4_derives_order_from_the_production_batch_sampler() -> None:
    source = inspect.getsource(_s4)

    assert "batch_sampler" in source
    assert "torch.empty((), dtype=torch.int64).random_()" in source


def test_s4_checks_vendor_loader_order_by_authoritative_ids() -> None:
    source = inspect.getsource(_s4)

    assert 'vendor_batch["id"]' in source


def test_s4_restores_rng_between_lazy_loader_iterators() -> None:
    source = inspect.getsource(_s4)

    assert "package_loader_rng" in source
    assert "restore_rng_state(package_loader_rng)" in source


def test_s4_compares_vendor_collated_retrieval_batch() -> None:
    source = inspect.getsource(_s4)

    assert 'batch["retrieved"]' in source

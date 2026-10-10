import torch

from canvas_vae.training.parity import (
    KERAS_ADAM_EPSILON,
    WELL_CONDITIONED_SQRT_V,
    check_keras_adam_update,
    keras_adam_float64,
)


def test_keras_adam_float64_returns_clipped_gradient_and_updated_moments():
    gradient = torch.tensor([3.0, 4.0])
    zeros = torch.zeros_like(gradient)

    update, clipped, first, second = keras_adam_float64(gradient, zeros, zeros, step=1)

    torch.testing.assert_close(clipped, torch.tensor([0.6, 0.8], dtype=torch.float64))
    torch.testing.assert_close(first, torch.tensor([0.06, 0.08], dtype=torch.float64))
    torch.testing.assert_close(
        second, torch.tensor([0.00036, 0.00064], dtype=torch.float64)
    )
    assert update.dtype == torch.float64
    assert KERAS_ADAM_EPSILON == 1e-7


def test_adam_update_check_gates_each_reference_and_reports_cross_system_updates():
    package_gradient = torch.tensor([0.1, 0.2])
    original_gradient = torch.tensor([0.11, 0.18])
    package_first = torch.tensor([0.01, 0.02])
    package_second = torch.tensor([1e-4, 2e-4])
    original_first = torch.tensor([0.012, 0.018])
    original_second = torch.tensor([1.2e-4, 1.8e-4])
    package_update = keras_adam_float64(
        package_gradient, package_first, package_second, step=2
    )[0]
    original_update = keras_adam_float64(
        original_gradient, original_first, original_second, step=2
    )[0]

    result = check_keras_adam_update(
        package_update,
        original_update,
        package_gradient,
        original_gradient,
        package_first,
        package_second,
        original_first,
        original_second,
        step=2,
        adam_rule_limit=3.5e-4,
    )

    assert result["sqrt_v_threshold"] == WELL_CONDITIONED_SQRT_V
    assert result["well_conditioned_elements"] == 2
    assert result["norm_rel"] > 0
    assert result["well_conditioned_norm_rel"] > 0.01
    assert result["package_adam_rule"] == 0
    assert result["original_adam_rule"] == 0
    assert result["cross_system_update_report_only"]
    assert result["within"]


def test_adam_update_check_rejects_either_system_outside_its_own_reference():
    gradient = torch.tensor([0.1])
    zeros = torch.zeros_like(gradient)
    exact = keras_adam_float64(gradient, zeros, zeros, step=1)[0]

    package_result = check_keras_adam_update(
        exact + 0.1,
        exact,
        gradient,
        gradient,
        zeros,
        zeros,
        zeros,
        zeros,
        step=1,
        adam_rule_limit=1e-3,
    )
    original_result = check_keras_adam_update(
        exact,
        exact + 0.1,
        gradient,
        gradient,
        zeros,
        zeros,
        zeros,
        zeros,
        step=1,
        adam_rule_limit=1e-3,
    )

    assert not package_result["package_adam_rule_within"]
    assert package_result["original_adam_rule_within"]
    assert not package_result["within"]
    assert original_result["package_adam_rule_within"]
    assert not original_result["original_adam_rule_within"]
    assert not original_result["within"]

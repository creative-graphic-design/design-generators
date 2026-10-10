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


def test_adam_update_check_reports_full_error_but_gates_conditioned_elements():
    package_gradient = torch.tensor([1e-8, 0.1])
    original_gradient = torch.tensor([2e-8, 0.1])
    zeros = torch.zeros_like(package_gradient)
    package_update = keras_adam_float64(package_gradient, zeros, zeros, step=1)[0]
    original_update = keras_adam_float64(original_gradient, zeros, zeros, step=1)[0]

    result = check_keras_adam_update(
        package_update,
        original_update,
        package_gradient,
        original_gradient,
        zeros,
        zeros,
        step=1,
        adam_rule_limit=3.5e-4,
        well_conditioned_limit=0.0,
    )

    assert result["sqrt_v_threshold"] == WELL_CONDITIONED_SQRT_V
    assert result["well_conditioned_elements"] == 1
    assert result["elements"] == 2
    assert result["norm_rel"] > 0
    assert result["well_conditioned_norm_rel"] == 0
    assert result["package_adam_rule"] == 0
    assert result["original_adam_rule"] == 0
    assert result["within"]


def test_adam_update_check_calibration_mode_allows_no_well_conditioned_elements():
    gradient = torch.tensor([1e-8])
    zeros = torch.zeros_like(gradient)
    update = keras_adam_float64(gradient, zeros, zeros, step=1)[0]

    result = check_keras_adam_update(
        update,
        update,
        gradient,
        gradient,
        zeros,
        zeros,
        step=1,
        adam_rule_limit=3.5e-4,
        well_conditioned_limit=None,
    )

    assert result["well_conditioned_elements"] == 0
    assert result["well_conditioned_norm_rel"] == 0
    assert result["well_conditioned_within"] is None
    assert result["within"]


def test_adam_update_check_rejects_a_well_conditioned_difference():
    gradient = torch.tensor([0.1])
    zeros = torch.zeros_like(gradient)
    exact = keras_adam_float64(gradient, zeros, zeros, step=1)[0]

    result = check_keras_adam_update(
        exact + 1e-4,
        exact,
        gradient,
        gradient,
        zeros,
        zeros,
        step=1,
        adam_rule_limit=1.0,
        well_conditioned_limit=1e-3,
    )

    assert result["well_conditioned_elements"] == 1
    assert result["well_conditioned_norm_rel"] > 1e-3
    assert result["well_conditioned_within"] is False
    assert result["within"] is False

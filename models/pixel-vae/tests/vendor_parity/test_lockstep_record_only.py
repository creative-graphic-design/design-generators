from vendor_parity.run_real_scale_lockstep import (
    _retain_first_divergence,
    _should_stop_after_divergence,
)


def test_record_only_continues_after_divergence() -> None:
    divergence = {"relative_loss_difference": 1.1e-2}
    later_divergence = {"relative_loss_difference": 1.2e-2}

    assert _should_stop_after_divergence(None, record_only=False) is False
    assert _should_stop_after_divergence(divergence, record_only=False) is True
    assert _should_stop_after_divergence(divergence, record_only=True) is False
    assert _retain_first_divergence(None, divergence) is divergence
    assert _retain_first_divergence(divergence, later_divergence) is divergence

import pytest
import torch

from traingen.lightning.steps import (
    finish_training_step,
    log_training_losses,
    log_validation_loss,
    sum_loss_values,
)


class RecordingLogger:
    def __init__(self) -> None:
        self.records: list[
            tuple[str, float, bool, bool | None, bool | None, int | None]
        ] = []

    def log(
        self,
        name: str,
        value: torch.Tensor,
        *,
        prog_bar: bool = False,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.records.append(
            (name, float(value.detach()), prog_bar, on_step, on_epoch, batch_size)
        )


def test_sum_loss_values_reduces_scalar_losses() -> None:
    losses = {"kl_loss": torch.tensor(2.0), "aux_loss": torch.tensor(0.5)}

    assert sum_loss_values(losses).item() == pytest.approx(2.5)


def test_log_training_losses_records_names_flags_and_batch_size() -> None:
    logger = RecordingLogger()
    losses = {"kl_loss": torch.tensor(2.0), "aux_loss": torch.tensor(0.5)}

    log_training_losses(logger, losses, torch.tensor(2.5), batch_size=4)

    assert logger.records == [
        ("kl_loss", 2.0, False, True, True, 4),
        ("aux_loss", 0.5, False, True, True, 4),
        ("train_loss", 2.5, True, True, True, 4),
    ]


def test_finish_training_step_inserts_detached_train_loss() -> None:
    logger = RecordingLogger()
    losses = {
        "kl_loss": torch.tensor(2.0, requires_grad=True),
        "aux_loss": torch.tensor(0.5, requires_grad=True),
    }
    trace = {"t": torch.tensor([1, 2])}

    total, updated = finish_training_step(logger, losses, trace, batch_size=2)

    assert total.item() == pytest.approx(2.5)
    assert updated["train_loss"].requires_grad is False
    assert torch.equal(updated["train_loss"], total.detach())
    assert torch.equal(updated["t"], trace["t"])


def test_log_validation_loss_records_flags_and_batch_size() -> None:
    logger = RecordingLogger()

    log_validation_loss(logger, torch.tensor(1.25), batch_size=8)

    assert logger.records == [("val_loss", 1.25, True, False, True, 8)]

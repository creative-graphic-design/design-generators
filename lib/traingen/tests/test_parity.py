import torch

from traingen.lightning.hooks import grad_norms, learning_rates
from traingen.parity.compare import (
    compare_batch_stream,
    compare_optimizer_step,
    compare_step_trace,
    compare_tensors,
)
from traingen.parity.determinism import (
    DeterminismConfig,
    apply_determinism,
    capture_rng_state,
    restore_rng_state,
)
from traingen.parity.trace import (
    build_step_trace,
    scalar_trace_value,
    trace_training_step,
)


def test_rng_capture_restore_replays_torch_random_values() -> None:
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = torch.rand(3)
    restore_rng_state(state)
    assert torch.equal(torch.rand(3), expected)


def test_apply_determinism_sets_reproducible_seed() -> None:
    apply_determinism(DeterminismConfig(seed=11, deterministic_algorithms=False))
    expected = torch.rand(2)
    apply_determinism(DeterminismConfig(seed=11, deterministic_algorithms=False))
    assert torch.equal(torch.rand(2), expected)


def test_step_trace_and_comparison_reports_match() -> None:
    trace = build_step_trace("a", {"x": torch.ones(2), "loss": torch.tensor(0.0)})
    report = compare_step_trace(trace, trace)
    assert report.passed
    assert not report.missing
    assert compare_tensors("x", torch.ones(1), torch.ones(1)).passed


def test_comparison_reports_mismatches_and_streams() -> None:
    vendor = build_step_trace("a", {"x": torch.ones(2), "missing": torch.zeros(1)})
    target = build_step_trace("b", {"x": torch.zeros(2)})
    report = compare_step_trace(vendor, target)
    assert not report.passed
    assert report.missing == ("missing",)
    assert not report.comparisons[0].passed
    assert not compare_tensors("shape", torch.ones(2), torch.ones(3)).passed
    assert compare_optimizer_step({"x": torch.ones(1)}, {"x": torch.ones(1)}).passed
    assert compare_batch_stream(
        [{"x": torch.ones(1)}], [{"x": torch.ones(1)}], steps=1
    ).passed
    mismatch = compare_batch_stream(
        [{"x": torch.ones(1)}], [{"x": torch.zeros(1)}], steps=1
    )
    assert mismatch.first_mismatch == "0:x"


def test_trace_training_step_reads_latest_step_trace() -> None:
    class Module:
        def training_step(self, batch, batch_idx):
            del batch, batch_idx
            self.latest_step_trace = {"x": torch.tensor([1.0])}
            return torch.tensor(2.0)

    trace = trace_training_step(Module(), {}, None, ("x", "train_loss"))
    assert trace.tensors["x"].item() == 1.0
    assert trace.tensors["train_loss"].item() == 2.0
    assert scalar_trace_value(trace.tensors["train_loss"]) == 2.0


def test_lightning_hooks_report_lr_and_grad_norms() -> None:
    module = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    output = module(torch.ones(1, 2)).sum()
    output.backward()
    assert learning_rates(optimizer) == (0.1,)
    norms = grad_norms(module)
    assert set(norms) == {"weight", "bias"}

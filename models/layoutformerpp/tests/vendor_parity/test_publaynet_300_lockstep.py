"""CPU-only checks for the lockstep trace helpers."""

import importlib
import random
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from layoutformerpp.training.recipes import LayoutFormerPPTrainingRecipe

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LOCKSTEP = importlib.import_module("run_publaynet_300_lockstep")


def test_runner_training_stream_path_uses_cpu_loaders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = LOCKSTEP.parity.TRAINING_RECIPES_BY_NAME["publaynet_unconditional"]
    batch: dict[str, object] = {"input_ids": torch.tensor([[1, 2]])}
    calls: list[tuple[str, str, Path | None, int, int]] = []

    def reference_stream(
        requested_recipe: LayoutFormerPPTrainingRecipe,
        data_root: Path,
        *,
        batches: int,
        seed: int,
    ) -> list[dict[str, object]]:
        calls.append(("reference", requested_recipe.name, data_root, batches, seed))
        return [batch]

    def package_stream(
        requested_recipe: LayoutFormerPPTrainingRecipe,
        *,
        batches: int,
        seed: int,
    ) -> list[dict[str, object]]:
        calls.append(("package", requested_recipe.name, None, batches, seed))
        return [batch]

    monkeypatch.setattr(LOCKSTEP, "reference_train_loader_stream", reference_stream)
    monkeypatch.setattr(LOCKSTEP.parity, "_package_train_loader_stream", package_stream)

    source, package = LOCKSTEP._training_streams(recipe, tmp_path, batches=1)

    assert calls == [
        ("reference", recipe.name, tmp_path, 1, 0),
        ("package", recipe.name, None, 1, 0),
    ]
    for stream in (source, package):
        input_ids = stream[0]["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        assert input_ids.device.type == "cpu"


def test_batch_hash_and_stream_parity_are_cpu_only() -> None:
    reference = {
        "names": ("sample",),
        "task_names": ("label",),
        "task_ids": (1,),
        "input_bytes": (b"input",),
        "output_bytes": (b"output",),
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[True, True]]),
        "labels": torch.tensor([[2, 1]]),
        "target_mask": torch.tensor([[True, False]]),
    }
    package = dict(reference)
    assert LOCKSTEP._batch_sha256(reference) == LOCKSTEP._batch_sha256(package)
    LOCKSTEP._equal_batches(reference, package)
    normalized = LOCKSTEP._batch(reference, torch.device("cpu"))
    assert normalized["task_ids"] is not None
    assert torch.equal(normalized["task_ids"], torch.tensor([1], dtype=torch.long))
    package["target_mask"] = torch.tensor([[True, True]])
    with pytest.raises(RuntimeError, match="target_mask"):
        LOCKSTEP._equal_batches(reference, package)
    package["target_mask"] = reference["target_mask"]
    package["input_ids"] = torch.tensor([[1, 3]])
    with pytest.raises(RuntimeError, match="input_ids"):
        LOCKSTEP._equal_batches(reference, package)


def test_package_stream_seeds_at_production_loader_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loader_started = False
    seed_calls: list[tuple[str, int]] = []
    original_random_seed = random.seed
    original_numpy_seed = np.random.seed
    original_torch_seed = torch.manual_seed

    def checked_seed(kind: str, original, seed: int) -> None:
        assert loader_started
        seed_calls.append((kind, seed))
        original(seed)

    class DataModule:
        def __init__(self, *, recipe_name, data_root, num_workers, seed) -> None:
            self.seed = seed

        def setup(self, stage: str) -> None:
            pass

        def train_dataloader(self):
            nonlocal loader_started
            loader_started = True
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            return [
                {
                    "names": ["layout"],
                    "task_names": ["unconditional"],
                    "task_ids": torch.tensor([0]),
                    "input_bytes": [b"input"],
                    "output_bytes": [b"output"],
                    "input_ids": torch.tensor([[1]]),
                    "attention_mask": torch.tensor([[True]]),
                    "labels": torch.tensor([[2]]),
                    "target_attention_mask": torch.tensor([[True]]),
                }
            ]

    monkeypatch.setattr(
        random,
        "seed",
        lambda seed: checked_seed("python", original_random_seed, seed),
    )
    monkeypatch.setattr(
        np.random,
        "seed",
        lambda seed: checked_seed("numpy", original_numpy_seed, seed),
    )
    monkeypatch.setattr(
        LOCKSTEP.torch,
        "manual_seed",
        lambda seed: checked_seed("torch", original_torch_seed, seed),
    )
    monkeypatch.setattr(LOCKSTEP.parity, "LayoutFormerPPDataModule", DataModule)
    monkeypatch.setattr(LOCKSTEP.parity, "_real_data_root", lambda: tmp_path)

    LOCKSTEP.parity._package_train_loader_stream(
        LOCKSTEP.parity.TRAINING_RECIPES_BY_NAME["publaynet_unconditional"],
        batches=1,
        seed=0,
    )

    assert seed_calls == [("python", 0), ("numpy", 0), ("torch", 0)]


def test_phase_comparison_gate_rejects_cross_system_exceedance_first() -> None:
    assert LOCKSTEP._gate_passed(cross_step=8, noise_step=4)
    assert LOCKSTEP._gate_passed(cross_step=None, noise_step=4)
    assert not LOCKSTEP._gate_passed(cross_step=3, noise_step=4)
    assert not LOCKSTEP._gate_passed(cross_step=8, noise_step=None)


def test_phase_comparison_passes_after_noise_floor_exceedance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    initial_states: list[torch.Tensor] = []
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    LOCKSTEP._configure_tf32()

    class TinyModel(torch.nn.Module):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.marker = torch.nn.Parameter(torch.rand(()))

    class ReferenceModel(TinyModel):
        def __init__(self, **kwargs: object) -> None:
            torch.rand(2)
            super().__init__(**kwargs)

    class TrainingModule(torch.nn.Module):
        def __init__(self, *, recipe_name: str, config: object) -> None:
            super().__init__()
            self.model = TinyModel()

        def training_step(self, batch: object, index: int) -> torch.Tensor:
            initial_states.append(self.model.marker.detach().clone())
            return torch.abs(self.model.weight - 1.0)

    def optimizer_scheduler(
        module: TrainingModule,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.SGD(module.model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        return optimizer, scheduler

    def reference_trace(model: torch.nn.Module, batch: object) -> dict[str, object]:
        assert isinstance(model, TinyModel)
        return {"loss": torch.abs(model.weight - 1.0)}

    monkeypatch.setattr(
        LOCKSTEP.parity, "reference_classes", lambda: (ReferenceModel, None)
    )
    monkeypatch.setattr(LOCKSTEP.parity, "LayoutFormerPPTrainingModule", TrainingModule)
    monkeypatch.setattr(LOCKSTEP.parity, "_s1_package_config", lambda recipe: None)
    monkeypatch.setattr(LOCKSTEP, "_optimizer_scheduler", optimizer_scheduler)
    monkeypatch.setattr(
        LOCKSTEP.parity, "_reference_pre_optimizer_trace", reference_trace
    )
    monkeypatch.setattr(
        LOCKSTEP.parity,
        "reference_optimizer_scheduler_state",
        lambda steps, learning_rate: {"lr_sequences": {str(steps[0]): [0.1, 0.1]}},
    )
    recipe = LOCKSTEP.parity.TRAINING_RECIPES_BY_NAME["publaynet_unconditional"]
    batch: dict[str, object] = {
        "names": ["sample"],
        "task_names": ["unconditional"],
        "task_ids": [0],
        "input_bytes": [b"input"],
        "output_bytes": [b"output"],
        "input_ids": torch.tensor([[1]]),
        "attention_mask": torch.tensor([[True]]),
        "labels": torch.tensor([[2]]),
        "target_mask": torch.tensor([[True]]),
    }
    batches: list[dict[str, object]] = [batch]
    cpu = torch.device("cpu")

    def run_phase(phase: str) -> dict[str, object]:
        return LOCKSTEP._run_phase(
            "unconditional", recipe, phase, batches, batches, tmp_path, cpu
        )

    cross = run_phase("cross_system")
    noise = run_phase("noise_floor")

    assert torch.equal(initial_states[0], initial_states[1])
    assert cross["first_relative_loss_exceedance"] is None
    assert noise["first_relative_loss_exceedance"] == 1
    assert LOCKSTEP._gate_passed(
        cast(int | None, cross["first_relative_loss_exceedance"]),
        cast(int | None, noise["first_relative_loss_exceedance"]),
    )

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    with pytest.raises(RuntimeError, match="TF32 is enabled"):
        run_phase("tf32_check")

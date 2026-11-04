from __future__ import annotations

from collections.abc import Mapping
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from torch import nn

from laygen.common.testing import skip_or_fail_vendor_parity
from laygen.common.vendor import vendor_root
from lace.configuration_lace import default_model_config
from lace.conversion import convert_state_dict, load_vendor_state_dict
from lace.modeling_lace import LaceTransformerModel
from lace.training.config import LaceSeedMode
from lace.training.datamodule import LaceDataModule
from lace.training.dataset import LaceProcessedDataset, collate_lace_batch
from lace.training.ema import LaceEMA
from lace.training.lightning_module import LaceTrainingModule
from lace.training.losses import (
    alignment_matrix,
    constraint_temporal_weight,
    lace_losses,
    layout_alignment,
    pairwise_iou,
    rand_fix,
    xywh_to_ltrb_reference,
)
from lace.training.seed import apply_lace_seed_mode
from lace.training.trace import (
    LaceTrainingTraceCallback,
    gradient_norm,
    _mapping_l2_norm,
    _tensor_hashes,
)
from traingen_parity.trace import tensor_sha256


def _load_vendor_transformer_encoder(vendor_dir: Path) -> type[torch.nn.Module]:
    spec = spec_from_file_location(
        "lace_vendor_training_backbone", vendor_dir / "util" / "backbone.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load vendor LACE backbone")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    transformer = getattr(module, "TransformerEncoder")
    if not isinstance(transformer, type):
        raise TypeError("Vendor TransformerEncoder must be a class")
    return cast(type[torch.nn.Module], transformer)


@pytest.mark.parametrize(
    ("dataset", "checkpoint"),
    [
        ("publaynet", "publaynet_best.pt"),
        ("rico25", "rico25_best.pt"),
    ],
)
def test_s0_copied_checkpoint_forward_matches_package(
    dataset: str,
    checkpoint: str,
) -> None:
    root = Path(__file__).parents[3]
    checkpoint_path = root / ".cache" / "lace" / "original" / "model" / checkpoint
    if not checkpoint_path.exists():
        skip_or_fail_vendor_parity(
            "LACE vendor checkpoint is local-only",
            missing_paths=[checkpoint_path],
            regeneration_hint="download the LACE vendor checkpoints into .cache/lace/original/model",
        )
    vendor_dir = vendor_root("lace", marker=Path("util") / "backbone.py")
    TransformerEncoder = _load_vendor_transformer_encoder(vendor_dir)
    config = default_model_config(dataset)
    device = torch.device("cpu")
    vendor = TransformerEncoder(
        num_layers=config["num_layers"],
        dim_seq=config["seq_dim"],
        dim_transformer=config["dim_transformer"],
        nhead=config["nhead"],
        dim_feedforward=config["dim_feedforward"],
        diffusion_step=config["diffusion_step"],
        device=device,
    ).eval()
    package = LaceTransformerModel(**config).eval()
    state = convert_state_dict(load_vendor_state_dict(checkpoint_path))
    vendor.load_state_dict(state, strict=True)
    package.load_state_dict(state, strict=True)
    assert sum(parameter.numel() for parameter in vendor.parameters()) == sum(
        parameter.numel() for parameter in package.parameters()
    )
    assert set(vendor.state_dict()) == set(package.state_dict())
    generator = torch.Generator(device=device).manual_seed(123)
    sample = torch.randn(
        2,
        config["max_seq_length"],
        config["seq_dim"],
        device=device,
        generator=generator,
    )
    timestep = torch.tensor([1, 201], device=device)
    with torch.no_grad():
        expected = vendor(sample, timestep=timestep)
        actual = package(sample=sample, timestep=timestep).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _write_processed_data(
    root: Path,
    *,
    attr: object = None,
    rows: int = 3,
) -> None:
    processed = root / "publaynet-max25" / "processed"
    processed.mkdir(parents=True)
    x = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.2, 0.2],
            [0.3, 0.4, 0.1, 0.1],
            [0.4, 0.5, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    data = SimpleNamespace(x=x, y=y, attr=attr or {"name": ["row-0", "row-1", "row-2"]})
    x_slices = torch.tensor([0, 2, 3, 4])[: rows + 1]
    slices = {"x": x_slices, "y": x_slices}
    for split in ("train", "val", "test"):
        torch.save((data, slices), processed / f"{split}.pt")


def test_training_config_seed_and_ema() -> None:
    assert LaceSeedMode.default.value == "default"
    assert LaceSeedMode.deterministic.value == "deterministic"
    with pytest.raises(ValueError):
        LaceSeedMode("invalid")

    apply_lace_seed_mode("default", seed=17)
    first = torch.rand(3)
    apply_lace_seed_mode("default", seed=17)
    assert torch.equal(first, torch.rand(3))

    apply_lace_seed_mode(LaceSeedMode.deterministic, seed=19)
    assert torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)

    module = nn.Linear(2, 1)
    frozen = nn.Parameter(torch.ones(1), requires_grad=False)
    module.register_parameter("frozen", frozen)
    ema = LaceEMA(mu=0.5)
    ema.register(module)
    original = ema.state_dict()
    with torch.no_grad():
        module.weight.fill_(3.0)
    ema.update(module)
    assert torch.equal(ema.shadow["weight"], (original["weight"] + 3.0) / 2)
    assert "frozen" not in ema.shadow
    loaded = LaceEMA(mu=0.5)
    loaded.load_state_dict(ema.state_dict())
    assert torch.equal(loaded.shadow["weight"], ema.shadow["weight"])
    assert not torch.equal(original["weight"], loaded.shadow["weight"])

    empty = LaceEMA(mu=0.5)
    empty.update(module)
    assert set(empty.shadow) == {"weight", "bias"}


def test_training_trace_summaries_use_shared_helpers() -> None:
    values = {
        "b": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "a": torch.tensor([3.0], dtype=torch.float32),
    }
    parameters = {
        "weight": nn.Parameter(torch.zeros(1, 2)),
        "bias": nn.Parameter(torch.zeros(1)),
    }
    parameters["weight"].grad = values["b"].view(1, 2)
    parameters["bias"].grad = values["a"]
    expected = torch.cat([values["a"], values["b"]]).to(dtype=torch.float64)
    expected_norm = float(expected.square().sum().sqrt().item())

    assert _tensor_hashes(values) == {
        name: tensor_sha256(value) for name, value in values.items()
    }
    assert _mapping_l2_norm(values) == expected_norm
    assert gradient_norm(parameters) == expected_norm


def test_training_trace_callback_records_optimizer_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = tmp_path / "trace.json"
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=1e-5)

    class FakeModule(nn.Module):
        latest_step_trace = {"train_loss": torch.tensor(2.0)}
        latest_ema_state = {"weight": torch.tensor([1.0])}

        def __init__(self) -> None:
            super().__init__()
            self.weight = parameter
            self.model = SimpleNamespace(
                state_dict=lambda: {"weight": parameter.detach()}
            )

        def optimizers(self) -> torch.optim.Optimizer:
            return optimizer

    class FakeTrainer:
        global_step = 1
        current_epoch = 0
        max_steps = 1
        num_training_batches = 1
        logger = SimpleNamespace(log_dir=tmp_path / "logs")

    module = FakeModule()
    trainer = FakeTrainer()
    callback = LaceTrainingTraceCallback()
    monkeypatch.setenv("LACE_TRAINING_TRACE_PATH", str(trace_path))
    callback.setup(cast(Trainer, trainer), cast(LightningModule, module), stage="fit")
    callback.on_train_batch_start(
        cast(Trainer, trainer),
        cast(LightningModule, module),
        {"id": ["sample"]},
        0,
    )
    loss = (parameter.square()).sum()
    loss.backward()
    callback.on_after_backward(cast(Trainer, trainer), cast(LightningModule, module))
    callback.on_before_optimizer_step(
        cast(Trainer, trainer), cast(LightningModule, module), optimizer
    )
    optimizer.step()
    callback.on_train_batch_end(
        cast(Trainer, trainer), cast(LightningModule, module), None, {}, 0
    )
    callback.on_train_end(cast(Trainer, trainer), cast(LightningModule, module))

    payload = json.loads(trace_path.read_text())
    assert len(payload["records"]) == 1
    assert payload["records"][0]["batch_ids"] == ["sample"]
    assert payload["trainer_state"]["global_step"] == 1


def test_processed_dataset_and_batch_collation(tmp_path: Path) -> None:
    _write_processed_data(tmp_path)
    dataset = LaceProcessedDataset(
        processed_data_dir=tmp_path / "publaynet-max25" / "processed",
        dataset_name="publaynet",
        split="validation",
        max_seq_length=25,
    )
    assert len(dataset) == 3
    item = dataset[0]
    assert item["id"] == "row-0"
    assert cast_tensor(item["bbox"]).shape == (2, 4)
    assert torch.equal(cast_tensor(item["mask"]), torch.ones(2, dtype=torch.bool))

    short = {
        "bbox": torch.ones(3, 4),
        "labels": torch.arange(3),
        "mask": torch.ones(3, dtype=torch.bool),
        "id": "short",
    }
    missing_id = {key: value for key, value in short.items() if key != "id"}
    batch = collate_lace_batch([short, missing_id], max_seq_length=2)
    assert cast_tensor(batch["bbox"]).shape == (2, 2, 4)
    assert "id" not in batch
    assert cast_tensor(batch["mask"]).tolist() == [[True, True], [True, True]]
    with pytest.raises(ValueError, match="cannot be empty"):
        collate_lace_batch([])

    dataset.data.attr = {"name": "single"}
    assert dataset._sample_id(0) == "single"
    assert dataset._sample_id(1) is None
    dataset.data.attr = cast(Mapping[str, object], object())
    assert dataset._sample_id(0) is None

    with pytest.raises(FileNotFoundError):
        LaceProcessedDataset(
            processed_data_dir=tmp_path / "missing",
            dataset_name="publaynet",
            split="train",
        )


def test_data_module_builds_all_loader_streams(tmp_path: Path) -> None:
    _write_processed_data(tmp_path)
    data_module = LaceDataModule(
        processed_data_dir=tmp_path / "publaynet-max25" / "processed",
        dataset_name="publaynet",
        batch_size=2,
        max_seq_length=25,
        num_workers=0,
        pin_memory=False,
        loader_seed=11,
    )
    data_module.setup()
    assert len(data_module.train_dataloader()) == 2
    assert len(data_module.val_dataloader()) == 2
    assert len(data_module.test_dataloader()) == 2
    assert next(iter(data_module.train_dataloader()))["bbox"].shape == (2, 25, 4)

    lazy = LaceDataModule(
        processed_data_dir=tmp_path / "publaynet-max25" / "processed",
        dataset_name="publaynet",
        batch_size=2,
        max_seq_length=25,
        num_workers=0,
        pin_memory=False,
    )
    assert len(lazy.val_dataloader()) == 2
    assert len(lazy.test_dataloader()) == 2
    with pytest.raises(RuntimeError, match="initialized"):
        data_module._loader(None, shuffle=False)


def test_training_loss_helpers_cover_reference_operations() -> None:
    bbox = torch.tensor(
        [[[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0]]]
    )
    mask = torch.tensor([[True, True, False]])
    assert xywh_to_ltrb_reference(bbox).shape == bbox.shape
    assert alignment_matrix(bbox, mask).shape == (1, 3, 6, 3)
    assert pairwise_iou(bbox, mask.float()).shape == (1, 3, 3)
    _, normalized = layout_alignment(bbox, mask)
    assert torch.isfinite(normalized).all()
    _, empty_normalized = layout_alignment(bbox, torch.zeros_like(mask))
    assert torch.equal(empty_normalized, torch.zeros(1))
    weights = constraint_temporal_weight(torch.tensor([0, 3]))
    assert weights.shape == (2,)
    torch.manual_seed(3)
    fixed = rand_fix(1, mask, ratio=1.0)
    assert fixed.shape == mask.shape
    assert not bool((fixed & ~mask).any())

    num_classes = 5
    layout_input = torch.randn(2, 3, num_classes + 4)
    output = torch.randn(8, 3, num_classes + 4)
    noise = torch.randn_like(output)
    reconstructed = torch.randn_like(output)
    losses = lace_losses(
        layout_input=layout_input,
        model_output=output,
        noise=noise,
        reconstructed=reconstructed,
        bbox=bbox.repeat(8, 1, 1),
        mask=mask.repeat(8, 1),
        timesteps=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        num_classes=num_classes,
    )
    assert set(losses) == {
        "diffusion_loss",
        "alignment_loss",
        "overlap_loss",
        "constraint_loss",
        "reconstruct_loss",
        "train_loss",
    }
    assert all(torch.isfinite(value) for value in losses.values())


def cast_tensor(value: object) -> torch.Tensor:
    assert isinstance(value, torch.Tensor)
    return value


def test_lightning_training_module_runs_tiny_cpu_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = dict(default_model_config("publaynet"))
    config.update(
        {"dim_transformer": 16, "nhead": 2, "num_layers": 1, "dim_feedforward": 32}
    )
    model = LaceTransformerModel(**config)
    module = LaceTrainingModule(
        dataset_name="publaynet",
        model=model,
        dim_transformer=16,
        nhead=2,
        num_layers=1,
        feature_dim=32,
        batch_size=2,
        sample_t_max=8,
        seed_mode="default",
    )
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    module.on_fit_start()
    assert {value.device for value in module.ema_helper.shadow.values()} == {
        next(module.model.parameters()).device
    }
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 1e-5
    batch = {
        "bbox": torch.rand(2, 3, 4),
        "labels": torch.tensor([[0, 1, 2], [2, 1, 0]]),
        "mask": torch.ones(2, 3, dtype=torch.bool),
    }
    loss = module.training_step(
        cast(dict[str, torch.Tensor | list[str]], batch),
        0,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert module.latest_step_trace["model_output"].shape[0] == 8
    module.on_train_batch_end(loss, batch, 0)
    assert module.latest_ema_state

    model_free = LaceTrainingModule(
        dataset_name="rico25",
        dim_transformer=16,
        nhead=2,
        num_layers=1,
        feature_dim=32,
        sample_t_max=8,
    )
    assert model_free.num_classes == 26

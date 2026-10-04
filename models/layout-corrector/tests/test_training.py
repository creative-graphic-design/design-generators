from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

pytest.importorskip("lightning")
pytest.importorskip("traingen")

import layout_corrector.training.dataset as dataset_module
import layout_corrector.training.reference as reference_module
import layout_corrector.training.lightning_module as lightning_module
import layout_corrector.training.parity as parity_module
from layout_corrector import LayoutCorrectorConfig, LayoutCorrectorModel
from layout_corrector.training import (
    FROZEN_LAYOUT_DM_SHA256,
    FrozenLayoutDMReference,
    LayoutCorrectorDataModule,
    LayoutCorrectorTrainingModule,
)
from layout_corrector.training.dataset import first_sample_ids
from layout_dm.configuration_layout_dm import LayoutDMConfig


@pytest.mark.parametrize("dataset", ("rico25", "publaynet"))
def test_shipped_training_config_prints_with_traingen(dataset: str) -> None:
    """Ensure the documented package training config is accepted by traingen."""
    root = Path(__file__).resolve().parents[3]
    config = (
        root
        / "models"
        / "layout-corrector"
        / "configs"
        / "training"
        / f"layoutcorrector_{dataset}.yaml"
    )
    executable = Path(sys.executable).with_name("traingen")
    result = subprocess.run(
        [str(executable), "fit", "--config", str(config), "--print_config"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _tiny_config() -> LayoutCorrectorConfig:
    return LayoutCorrectorConfig(
        dataset_name="rico25",
        vocab_size=8,
        max_seq_length=2,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=16,
        pos_emb="none",
        num_timesteps=4,
    )


def _registration_config() -> LayoutCorrectorConfig:
    return LayoutCorrectorConfig(
        dataset_name="rico25",
        vocab_size=155,
        max_seq_length=25,
        hidden_size=432,
        intermediate_size=1728,
        num_attention_heads=8,
        num_hidden_layers=4,
        pos_emb="none",
        num_timesteps=100,
    )


EXPECTED_PARAMETER_NAMES = (
    "backbone.layers.0.self_attn.in_proj_weight",
    "backbone.layers.0.self_attn.in_proj_bias",
    "backbone.layers.0.self_attn.out_proj.weight",
    "backbone.layers.0.self_attn.out_proj.bias",
    "backbone.layers.0.linear1.weight",
    "backbone.layers.0.linear1.bias",
    "backbone.layers.0.linear2.weight",
    "backbone.layers.0.linear2.bias",
    "backbone.layers.0.norm1.emb.weight",
    "backbone.layers.0.norm1.linear.weight",
    "backbone.layers.0.norm1.linear.bias",
    "backbone.layers.0.norm2.weight",
    "backbone.layers.0.norm2.bias",
    "backbone.layers.1.self_attn.in_proj_weight",
    "backbone.layers.1.self_attn.in_proj_bias",
    "backbone.layers.1.self_attn.out_proj.weight",
    "backbone.layers.1.self_attn.out_proj.bias",
    "backbone.layers.1.linear1.weight",
    "backbone.layers.1.linear1.bias",
    "backbone.layers.1.linear2.weight",
    "backbone.layers.1.linear2.bias",
    "backbone.layers.1.norm1.emb.weight",
    "backbone.layers.1.norm1.linear.weight",
    "backbone.layers.1.norm1.linear.bias",
    "backbone.layers.1.norm2.weight",
    "backbone.layers.1.norm2.bias",
    "backbone.layers.2.self_attn.in_proj_weight",
    "backbone.layers.2.self_attn.in_proj_bias",
    "backbone.layers.2.self_attn.out_proj.weight",
    "backbone.layers.2.self_attn.out_proj.bias",
    "backbone.layers.2.linear1.weight",
    "backbone.layers.2.linear1.bias",
    "backbone.layers.2.linear2.weight",
    "backbone.layers.2.linear2.bias",
    "backbone.layers.2.norm1.emb.weight",
    "backbone.layers.2.norm1.linear.weight",
    "backbone.layers.2.norm1.linear.bias",
    "backbone.layers.2.norm2.weight",
    "backbone.layers.2.norm2.bias",
    "backbone.layers.3.self_attn.in_proj_weight",
    "backbone.layers.3.self_attn.in_proj_bias",
    "backbone.layers.3.self_attn.out_proj.weight",
    "backbone.layers.3.self_attn.out_proj.bias",
    "backbone.layers.3.linear1.weight",
    "backbone.layers.3.linear1.bias",
    "backbone.layers.3.linear2.weight",
    "backbone.layers.3.linear2.bias",
    "backbone.layers.3.norm1.emb.weight",
    "backbone.layers.3.norm1.linear.weight",
    "backbone.layers.3.norm1.linear.bias",
    "backbone.layers.3.norm2.weight",
    "backbone.layers.3.norm2.bias",
    "cat_emb.weight",
    "head.0.weight",
    "head.0.bias",
    "head.1.weight",
    "enc.0.weight",
    "enc.0.bias",
    "dec.0.weight",
    "dec.0.bias",
)


class _FakeReference:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_timesteps=4)
        self.checkpoint_path = Path("checkpoint.pt")
        self.checkpoint_sha256 = "checkpoint-hash"
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.scheduler = SimpleNamespace()
        self.model = nn.Linear(1, 1)
        self.lt_history = torch.zeros(4)
        self.lt_count = torch.zeros(4)

    def corrupt(
        self, input_ids: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del timesteps
        return input_ids, torch.zeros(input_ids.shape[0], 8, input_ids.shape[1])

    def prediction_log_probs(
        self, log_xt: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        del timesteps
        return log_xt

    def reconstruct_previous(
        self,
        log_x0: torch.Tensor,
        log_xt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        del log_x0, timesteps
        return log_xt.argmax(dim=1)


def test_training_namespace_and_registration_order() -> None:
    assert FrozenLayoutDMReference is not None
    assert LayoutCorrectorDataModule is not None
    assert LayoutCorrectorTrainingModule is not None
    assert FROZEN_LAYOUT_DM_SHA256["rico25"]

    model = LayoutCorrectorModel(**dict(_registration_config().config))
    names = [name for name, _ in model.model.named_parameters()]
    assert tuple(names) == EXPECTED_PARAMETER_NAMES


def test_initialization_device_is_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert lightning_module._initialization_device() == torch.device("cpu")


def test_training_parity_digests_capture_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    values = {
        "weight": torch.tensor([1.0, 2.0]),
        "bias": torch.tensor([3.0]),
    }
    assert parity_module._tensor_digest(values["weight"])
    assert parity_module._parameter_state_digest(values)

    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
    assert parity_module._optimizer_state_digest(optimizer)
    assert parity_module._rng_digest(parity_module.capture_rng_state())
    assert parity_module._scheduler_state_digest({"last_epoch": 1, "best": 0.5})


class _FakeProcessedDataset(Dataset[dict[str, object]]):
    def __init__(self, **kwargs: object) -> None:
        self.split = str(kwargs["split"])

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, object]:
        return {"id": f"{self.split}-{index}"}


def test_training_data_module_loads_each_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dataset_module, "LayoutDMProcessedDataset", _FakeProcessedDataset
    )
    module = LayoutCorrectorDataModule(
        dataset_name="rico25",
        config=cast(LayoutDMConfig, object()),
        processed_data_dir="processed",
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    module.setup("fit")
    assert set(module._datasets) == {"train", "validation"}
    train_loader = module.train_dataloader()
    assert isinstance(train_loader.sampler, RandomSampler)
    assert train_loader.worker_init_fn is dataset_module._preserve_torch_worker_seed
    assert isinstance(module.val_dataloader().sampler, SequentialSampler)
    assert isinstance(module.test_dataloader().sampler, SequentialSampler)
    module.setup()
    assert set(module._datasets) == {"train", "validation", "test"}


def test_first_sample_ids_handles_batch_shapes() -> None:
    batches = [
        {"id": ["a", "b"]},
        {"id": ("c", "d")},
        {"id": "e"},
        {},
    ]
    loader = cast(DataLoader[dict[str, torch.Tensor | str]], batches)
    assert first_sample_ids(loader, count=4) == ["a", "b", "c", "d"]


def test_training_module_runs_sampler_loss_and_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lightning_module.torch.cuda, "is_available", lambda: False)
    reference = _FakeReference()
    monkeypatch.setattr(
        lightning_module.FrozenLayoutDMReference,
        "from_checkpoint",
        classmethod(lambda cls, **kwargs: reference),
    )
    module = LayoutCorrectorTrainingModule(
        config=_tiny_config(),
        layout_dm_checkpoint_path="checkpoint.pt",
        cluster_centers_path="clusters.pkl",
    )
    monkeypatch.setattr(
        LayoutCorrectorTrainingModule,
        "log",
        lambda self, *args, **kwargs: None,
    )
    assert module.layout_dm_checkpoint_sha256 == "checkpoint-hash"
    assert module._reference_value() is reference
    assert module._reference_model_value() is reference.model
    assert module.initialization_device == "cpu"
    assert len(module.optim_groups()) == 2
    module._reference_model_value().train()
    module.on_train_epoch_start()
    assert module._reference_model_value().training is False

    optimizer_config = module.configure_optimizers()
    assert isinstance(optimizer_config, dict)
    module.scheduler = None
    assert isinstance(module.configure_optimizers(), torch.optim.Optimizer)
    module.scheduler = "reduce_on_plateau"

    batch: dict[str, torch.Tensor | str] = {
        "input_ids": torch.tensor([[1] * 10, [2] * 10])
    }
    prepared = module.preprocess(batch)
    assert set(prepared) == {
        "t",
        "pt",
        "x0",
        "xt",
        "x0_recon",
        "recon_acc",
        "padding_mask",
    }
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0
    assert "train_loss" in module.latest_step_trace
    assert module.validation_step(batch, 0).ndim == 0

    loss.backward()
    module.configure_gradient_clipping(
        torch.optim.AdamW(module.model.parameters(), lr=1.0e-3),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    assert module.latest_gradient_norm is not None

    reference.lt_count.fill_(11)
    reference.lt_history.copy_(torch.arange(4, dtype=torch.float32))
    timesteps, probabilities = module._sample_time(3, torch.device("cpu"))
    assert timesteps.shape == (3,)
    assert probabilities.shape == (3,)

    saved_count = module._buffers["layout_dm_lt_count"]
    module._buffers["layout_dm_lt_count"] = None
    with pytest.raises(RuntimeError, match="buffers were not registered"):
        module._layout_dm_buffers()
    module._buffers["layout_dm_lt_count"] = saved_count


class _FakeReferenceConfig:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)
        self.vocab_size = 2
        self.max_token_length = 4
        self.mask_token_id = 1
        self.pad_token_id = 0
        self.var_order = "c-x"
        self.num_timesteps = 3
        self.att_1 = 0.1
        self.att_T = 0.2
        self.ctt_1 = 0.3
        self.ctt_T = 0.4


class _FakeTokenizer:
    var_names = ("c", "x")

    def __init__(self, config: _FakeReferenceConfig) -> None:
        self.config = config
        self.pad_token_id = config.pad_token_id

    def full_id_maps(self) -> dict[str, tuple[int, int]]:
        return {key: (0, 1) for key in self.var_names}

    def full_to_partial_ids(self, values: torch.Tensor, key: str) -> torch.Tensor:
        del key
        return values.remainder(2)

    def partial_to_full_log_probs(self, values: torch.Tensor, key: str) -> torch.Tensor:
        del key
        if values.ndim == 3:
            return values

        return torch.nn.functional.one_hot(values, 2).movedim(-1, 1).float().log()

    def partial_to_full_ids(self, values: torch.Tensor, key: str) -> torch.Tensor:
        del key
        return values


class _FakeDenoiser(nn.Module):
    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def initialize_weights(self) -> None:
        pass

    def forward(
        self, *, input_ids: torch.Tensor, timesteps: torch.Tensor
    ) -> SimpleNamespace:
        del timesteps
        return SimpleNamespace(
            logits=torch.zeros(input_ids.shape[0], 2, input_ids.shape[1])
        )


class _FakeScheduler:
    def __init__(self, **kwargs: object) -> None:
        self.schedules: dict[str, tuple[torch.Tensor, ...]] = {}

    def _q_pred(
        self, log_start: torch.Tensor, timesteps: torch.Tensor, key: str
    ) -> torch.Tensor:
        del timesteps, key
        return torch.zeros_like(log_start)

    def predict_start(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def q_posterior(
        self,
        log_x0: torch.Tensor,
        log_xt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        del log_x0, timesteps
        return torch.zeros_like(log_xt)


def test_frozen_reference_loads_and_runs_all_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw: dict[str, object] = {
        "model.module.Lt_history": torch.ones(3),
        "model.module.Lt_count": torch.ones(3) * 11,
    }
    for key in _FakeTokenizer.var_names:
        for suffix in (
            "log_at",
            "log_bt",
            "log_ct",
            "log_cumprod_at",
            "log_cumprod_bt",
            "log_cumprod_ct",
        ):
            raw[f"model.module.{key}_{suffix}"] = torch.ones(3)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(raw, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setitem(reference_module.FROZEN_LAYOUT_DM_SHA256, "rico25", digest)
    monkeypatch.setattr(reference_module, "LayoutDMConfig", _FakeReferenceConfig)
    monkeypatch.setattr(reference_module, "LayoutDMTokenizer", _FakeTokenizer)
    monkeypatch.setattr(reference_module, "LayoutDMDenoiser", _FakeDenoiser)
    monkeypatch.setattr(reference_module, "LayoutDMScheduler", _FakeScheduler)
    monkeypatch.setattr(
        reference_module,
        "split_original_state_dict",
        lambda value: {"weight": torch.ones(1)},
    )

    reference = FrozenLayoutDMReference.from_checkpoint(
        dataset_name="rico25",
        checkpoint_path=checkpoint,
        cluster_centers_path=tmp_path / "clusters.pkl",
    )
    assert reference.checkpoint_sha256 == digest
    assert reference.model.training is False
    assert all(
        not parameter.requires_grad for parameter in reference.model.parameters()
    )
    assert reference.to("cpu") is reference

    reference.lt_count.zero_()
    sampled, probabilities = reference.sample_time(2, device=torch.device("cpu"))
    assert sampled.shape == (2,)
    assert torch.allclose(probabilities, torch.full((2,), 1 / 100))
    reference.lt_count.fill_(11)
    sampled, probabilities = reference.sample_time(2, device=torch.device("cpu"))
    assert sampled.shape == (2,)
    assert probabilities.shape == (2,)

    input_ids = torch.tensor([[0, 1, 1, 0]])
    timesteps = torch.tensor([1])
    corrupted, log_xt = reference.corrupt(input_ids, timesteps)
    assert corrupted.shape == input_ids.shape
    clean = reference.prediction_log_probs(log_xt, timesteps)
    reconstructed = reference.reconstruct_previous(clean, log_xt, timesteps)
    assert clean.shape == log_xt.shape
    assert reconstructed.shape == input_ids.shape

    monkeypatch.setitem(reference_module.FROZEN_LAYOUT_DM_SHA256, "rico25", "wrong")
    with pytest.raises(ValueError, match="Unexpected frozen LayoutDM"):
        FrozenLayoutDMReference.from_checkpoint(
            dataset_name="rico25",
            checkpoint_path=checkpoint,
            cluster_centers_path=tmp_path / "clusters.pkl",
        )


def test_reconstruct_previous_softmaxes_over_vocab_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = object.__new__(FrozenLayoutDMReference)
    reference.config = SimpleNamespace(mask_token_id=2)
    log_previous = torch.tensor(
        [[[0.0, -0.2], [-0.7, -0.4], [-1.1, -0.9]]], dtype=torch.float32
    )

    class Scheduler:
        def q_posterior(
            self,
            log_x0: torch.Tensor,
            log_xt: torch.Tensor,
            timesteps: torch.Tensor,
        ) -> torch.Tensor:
            del log_x0, log_xt, timesteps
            return log_previous.clone()

    reference.scheduler = Scheduler()
    captured: dict[str, torch.Tensor] = {}

    def capture_multinomial(
        probabilities: torch.Tensor,
        num_samples: int,
        *,
        replacement: bool,
        device: torch.device,
    ) -> torch.Tensor:
        del num_samples, replacement
        captured["probabilities"] = probabilities
        return torch.zeros((probabilities.shape[0], 1), dtype=torch.long, device=device)

    monkeypatch.setattr(reference_module, "multinomial", capture_multinomial)
    reference.reconstruct_previous(log_previous, log_previous, torch.tensor([1]))

    expected = log_previous.clone()
    expected[:, reference.config.mask_token_id, :] = -70.0
    expected = expected.softmax(dim=1).movedim(1, -1).reshape(-1, 3)
    assert torch.equal(captured["probabilities"], expected)

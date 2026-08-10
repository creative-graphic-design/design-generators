from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypedDict, cast

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig

pytest.importorskip("traingen_parity")

from laygen.common.testing import skip_or_fail_vendor_parity
from layoutdiffusion import LayoutDiffusionConfig
from layoutdiffusion.conversion import remap_transformer_state_dict
from layoutdiffusion.modeling_layoutdiffusion import LayoutDiffusionTransformer
from layoutdiffusion.training.config import LayoutDiffusionTrainingDatasetName
from layoutdiffusion.training.datamodule import LayoutDiffusionDataModule
from layoutdiffusion.training.lightning_module import LayoutDiffusionTrainingModule
from layoutdiffusion.training.parity import (
    TRACE_POINTS,
    compare_layoutdiffusion_step,
    trace_layoutdiffusion_step,
)
from layoutdiffusion.training.vocab import build_training_tokenizer
from layoutdiffusion_training_reference import (
    PROJECT_ROOT,
    REFERENCE_DATASETS,
    VENDOR_ROOT,
    _add_vendor_to_path,
    generate_s3_s4_reference_fixture,
    reference_fixture_path,
    reference_s3_s4_fixture_path,
    tiny_reference_batch,
    tiny_training_config,
)
from traingen_parity.compare import TensorTolerance
from traingen_parity.trace import StepTrace, build_step_trace

pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]


class ReferenceFixture(TypedDict):
    batch: dict[str, torch.Tensor]
    trace: StepTrace
    trace_tensors: dict[str, torch.Tensor]
    rng_state_before_step: torch.Tensor
    cuda_rng_state_before_step: torch.Tensor | None
    model_state_before: dict[str, torch.Tensor]
    metadata: dict[str, object]


class ReferenceS3S4Fixture(TypedDict):
    batches: list[dict[str, torch.Tensor]]
    trajectory: list[dict[str, float | int]]
    rng_state_before_run: torch.Tensor
    cuda_rng_state_before_run: torch.Tensor | None
    model_state_before: dict[str, torch.Tensor]
    model_state_after: dict[str, torch.Tensor]
    ema_state_after: dict[str, torch.Tensor]
    metadata: dict[str, str | int | float]


class _InputIdsDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: torch.Tensor) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index].long()
        return {"input_ids": row, "attention_mask": row.ne(3), "mask": row.ne(3)}


@pytest.mark.parametrize("dataset_name", REFERENCE_DATASETS)
def test_s0_training_static_model_topology_matches_vendor(
    dataset_name: str,
) -> None:
    """Guard the real package/vendor transformer topology before trace tests."""
    config = _full_training_config(_checked_dataset_name(dataset_name))
    package_model = LayoutDiffusionTransformer(
        vocab_size=config.vocab_size,
        num_channels=config.num_channels,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        dropout=config.dropout,
        max_position_embeddings=config.max_position_embeddings,
    )
    vendor_model, _ = _build_vendor_training_components(config)

    assert _parameter_count(package_model) == _parameter_count(vendor_model)

    vendor_state = vendor_model.state_dict()
    remapped = remap_transformer_state_dict(vendor_state)
    ignored_vendor_keys = sorted(set(vendor_state) - set(remapped))
    assert ignored_vendor_keys == ["position_ids"]
    # BERT position_ids is a non-learned deterministic arange buffer; the package
    # registers it as non-persistent, so checkpoint/remap coverage intentionally
    # excludes it instead of silently tolerating an arbitrary unexpected key.
    assert "position_ids" not in package_model.state_dict()
    assert set(remapped) == set(package_model.state_dict())

    missing, unexpected = package_model.load_state_dict(remapped, strict=True)
    assert missing == []
    assert unexpected == []


@pytest.mark.parametrize("dataset_name", REFERENCE_DATASETS)
def test_s0_training_static_forward_matches_vendor_with_copied_weights(
    dataset_name: str,
) -> None:
    """Copy vendor weights into the package model and compare logits directly."""
    config = _full_training_config(_checked_dataset_name(dataset_name))
    package_model = LayoutDiffusionTransformer(
        vocab_size=config.vocab_size,
        num_channels=config.num_channels,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        dropout=config.dropout,
        max_position_embeddings=config.max_position_embeddings,
    )
    vendor_model, _ = _build_vendor_training_components(config)
    package_model.load_state_dict(
        remap_transformer_state_dict(vendor_model.state_dict()), strict=True
    )
    package_model.eval()
    vendor_model.eval()

    torch.manual_seed(102)
    input_ids = torch.randint(0, config.vocab_size, (2, config.seq_length)).long()
    timesteps = torch.tensor([0, 137], dtype=torch.long)
    with torch.no_grad():
        package_logits = package_model(input_ids=input_ids, timesteps=timesteps).logits
        vendor_logits = vendor_model(input_ids, timesteps)
    assert torch.allclose(package_logits, vendor_logits, atol=2e-5, rtol=2e-6)


@pytest.mark.parametrize("dataset_name", REFERENCE_DATASETS)
def test_s0_training_static_schedule_buffers_match_vendor(dataset_name: str) -> None:
    """Compare real-scale discrete schedule buffers for each dataset vocabulary."""
    config = _full_training_config(_checked_dataset_name(dataset_name))
    module = LayoutDiffusionTrainingModule(config=config, time_sampler="uniform")
    _, vendor_diffusion = _build_vendor_training_components(config)
    package_scheduler = module.diffusion_scheduler
    for name in (
        "log_at",
        "log_bt1",
        "log_bt2",
        "log_ct",
        "log_at1",
        "log_ct1",
        "log_cumprod_at",
        "log_cumprod_bt1",
        "log_cumprod_bt2",
        "log_cumprod_ct",
        "log_cumprod_at1",
        "log_cumprod_ct1",
        "log_1_min_ct",
        "log_1_min_cumprod_ct",
        "log_1_min_ct1",
        "log_1_min_cumprod_ct1",
        "q_onestep_mats",
        "q_mats",
    ):
        package_value = getattr(package_scheduler, name)
        vendor_value = getattr(vendor_diffusion, name)
        torch.testing.assert_close(
            package_value,
            vendor_value,
            atol=1e-12,
            rtol=0,
            equal_nan=True,
            msg=lambda message: f"{dataset_name} {name}: {message}",
        )


@pytest.mark.parametrize("dataset_name", REFERENCE_DATASETS)
def test_s0_training_static_optimizer_ema_and_sampler_state(
    dataset_name: str,
) -> None:
    """Pin optimizer defaults, EMA initialization, Lt buffers, and uniform sampling."""
    config = _full_training_config(_checked_dataset_name(dataset_name))
    module = LayoutDiffusionTrainingModule(
        config=config,
        learning_rate={"rico25": 4e-5, "publaynet": 5e-5}[dataset_name],
        weight_decay=0.0,
        scheduler="linear_anneal",
        lr_anneal_steps={"rico25": 175_000, "publaynet": 400_000}[dataset_name],
        ema_rate=0.9999,
        time_sampler="uniform",
    )
    optimizer_config = cast(dict[str, object], module.configure_optimizers())
    optimizer = cast(torch.optim.AdamW, optimizer_config["optimizer"])
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(
        {"rico25": 4e-5, "publaynet": 5e-5}[dataset_name], rel=0, abs=0
    )
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == pytest.approx(1e-8, rel=0, abs=0)
    assert group["weight_decay"] == pytest.approx(0.0, rel=0, abs=0)
    assert module.ema_rate == pytest.approx(0.9999, rel=0, abs=0)
    for name, parameter in module.model.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(module._ema_params[name], parameter.detach())
    lt_history = cast(torch.Tensor, module.lt_history)
    lt_count = cast(torch.Tensor, module.lt_count)
    assert torch.equal(lt_history, torch.zeros_like(lt_history))
    assert torch.equal(lt_count, torch.zeros_like(lt_count))
    _, vendor_diffusion = _build_vendor_training_components(config)
    vendor_lt_history = cast(torch.Tensor, vendor_diffusion.Lt_history)
    vendor_lt_count = cast(torch.Tensor, vendor_diffusion.Lt_count)
    assert torch.equal(vendor_lt_history, torch.zeros_like(vendor_lt_history))
    assert torch.equal(vendor_lt_count, torch.zeros_like(vendor_lt_count))
    assert module.time_sampler == "uniform"
    _, package_pt = module._sample_time(16, torch.device("cpu"))
    assert torch.allclose(
        package_pt, torch.full_like(package_pt, 1 / module.num_timesteps)
    )


@pytest.mark.parametrize("dataset_name", REFERENCE_DATASETS)
def test_s0_training_static_tokenizer_matches_vendor_vocab(dataset_name: str) -> None:
    """Pin tokenizer static values against vendor vocab artifacts."""
    config = _full_training_config(_checked_dataset_name(dataset_name))
    vendor_vocab_path = _vendor_s5_vocab_path(dataset_name)
    tokenizer = build_training_tokenizer(config, vocab_file=str(vendor_vocab_path))
    _require_paths(
        [vendor_vocab_path],
        "LayoutDiffusion vendor S5 vocab artifact is missing.",
        "Run the vendor S5 prelaunch or full-run command to write vocab.json.",
    )
    vendor_vocab = {
        str(key): int(value)
        for key, value in json.loads(vendor_vocab_path.read_text()).items()
    }
    package_vocab_without_mask = dict(tokenizer.get_vocab())
    package_vocab_without_mask.pop("MASK")
    assert package_vocab_without_mask == vendor_vocab
    vendor_label_order = [
        token
        for token, _ in sorted(vendor_vocab.items(), key=lambda item: item[1])
        if token not in {"START", "END", "UNK", "PAD", "|"} and not token.isdigit()
    ]
    assert config.id2label == dict(enumerate(vendor_label_order))
    assert tokenizer.vocab_size == {"rico25": 159, "publaynet": 139}[dataset_name]
    assert config.max_token_length == config.seq_length == 121
    assert tokenizer.pad_token_id == vendor_vocab["PAD"] == 3
    assert config.mask_token_id == max(vendor_vocab.values()) + 1


def test_s0_training_static_vendor_gpu_effective_sampler_is_uniform() -> None:
    """Codify the discovered vendor GPU behavior: Lt buffers never update."""
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is required to prove vendor CPU-diffusion/GPU-model Lt behavior"
        )
    config = _full_training_config("publaynet")
    vendor_model, vendor_diffusion = _build_vendor_training_components(config)
    from improved_diffusion import discrete_diffusion

    assert "vendor_run/improved-diffusion" in str(Path(discrete_diffusion.__file__))
    device = torch.device("cuda")
    vendor_model.to(device)
    # vendor model exposes .device as a plain attribute for sampling helpers
    setattr(vendor_model, "device", device)  # noqa: B010
    batch = torch.randint(0, config.vocab_size, (2, config.seq_length), device=device)
    torch.manual_seed(3000)
    training_losses = cast(
        "Callable[..., dict[str, torch.Tensor]]",
        vendor_diffusion.training_losses,
    )
    losses = training_losses(vendor_model, batch, model_kwargs={"input_ids": batch})
    assert losses["loss"].shape == (2,)
    gpu_lt_history = cast(torch.Tensor, vendor_diffusion.Lt_history)
    gpu_lt_count = cast(torch.Tensor, vendor_diffusion.Lt_count)
    assert gpu_lt_history.device.type == "cpu"
    assert gpu_lt_count.device.type == "cpu"
    assert torch.equal(gpu_lt_history, torch.zeros_like(gpu_lt_history))
    assert torch.equal(gpu_lt_count, torch.zeros_like(gpu_lt_count))
    sample_time = cast(
        "Callable[..., tuple[torch.Tensor, torch.Tensor]]",
        vendor_diffusion.sample_time,
    )
    t, pt = sample_time(8, device, "importance")
    assert t.device == batch.device
    num_timesteps = cast(int, vendor_diffusion.num_timesteps)
    assert torch.equal(pt, torch.full_like(pt, 1 / num_timesteps))


def test_package_training_trace_surface_has_s0_s2_points() -> None:
    config = tiny_training_config("publaynet")
    module = LayoutDiffusionTrainingModule(config=config, scheduler=None)
    torch.manual_seed(102)
    trace = trace_layoutdiffusion_step(module, tiny_reference_batch("publaynet"))
    assert set(TRACE_POINTS).issubset(trace.tensors)


@pytest.mark.parametrize("dataset_name", REFERENCE_DATASETS)
def test_original_training_reference_adapter_fixture_is_available(
    dataset_name: str,
) -> None:
    reference = _load_required_reference(dataset_name)
    assert set(TRACE_POINTS).issubset(reference["trace"].tensors)
    assert set(TRACE_POINTS).issubset(reference["trace_tensors"])
    assert reference["metadata"]["dataset_name"] == dataset_name


@pytest.mark.parametrize("dataset_name", REFERENCE_DATASETS)
def test_package_training_trace_matches_original_fixture(dataset_name: str) -> None:
    reference = _load_required_reference(dataset_name)
    metadata = reference["metadata"]
    auxiliary_loss_weight = metadata["auxiliary_loss_weight"]
    assert isinstance(auxiliary_loss_weight, float)
    config = tiny_training_config(dataset_name)
    module = LayoutDiffusionTrainingModule(
        config=config,
        scheduler=None,
        auxiliary_loss_weight=auxiliary_loss_weight,
        time_sampler="uniform",
        vocab_file=str(_vendor_s5_vocab_path(dataset_name)),
    )
    missing, unexpected = module.model.load_state_dict(
        reference["model_state_before"], strict=False
    )
    assert missing == []
    assert unexpected == ["position_ids"]
    metadata = reference["metadata"]
    assert metadata.get("time_sampler") == "effective_uniform"
    assert metadata.get("vocab_file") == str(_vendor_s5_vocab_path(dataset_name))
    batch = reference["batch"]
    trace_device = str(metadata.get("trace_device", "cpu"))
    if trace_device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for CUDA-generated S1 reference fixtures")
        device = torch.device("cuda")
        module.to(device)
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        cuda_rng_state = reference.get("cuda_rng_state_before_step")
        assert cuda_rng_state is not None
        torch.cuda.set_rng_state(cuda_rng_state, device)
    torch.set_rng_state(reference["rng_state_before_step"])
    target = _cpu_trace(trace_layoutdiffusion_step(module, batch))
    reference_trace = _without_trace_points(
        reference["trace"], {"lt_history", "lt_count"}
    )
    target_trace = _without_trace_points(target, {"lt_history", "lt_count"})
    assert torch.equal(
        reference["trace"].tensors["lt_history"],
        torch.zeros_like(reference["trace"].tensors["lt_history"]),
    )
    assert torch.equal(
        reference["trace"].tensors["lt_count"],
        torch.zeros_like(reference["trace"].tensors["lt_count"]),
    )
    assert (
        target.tensors["lt_count"].sum().item()
        == reference["batch"]["input_ids"].shape[0]
    )
    report = compare_layoutdiffusion_step(
        reference_trace,
        target_trace,
        tolerance=TensorTolerance(atol=2e-5, rtol=2e-6),
    )
    assert report.passed, [comparison.message for comparison in report.comparisons]


@pytest.mark.parametrize("dataset_name", REFERENCE_DATASETS)
def test_s3_repeated_training_matches_original_fixture(dataset_name: str) -> None:
    reference = _load_or_generate_s3_s4_reference(dataset_name)
    metadata = reference["metadata"]
    auxiliary_loss_weight = metadata["auxiliary_loss_weight"]
    learning_rate = metadata["learning_rate"]
    lr_anneal_steps = metadata["lr_anneal_steps"]
    assert isinstance(auxiliary_loss_weight, float)
    assert isinstance(learning_rate, float)
    assert isinstance(lr_anneal_steps, int)
    config = tiny_training_config(dataset_name)
    module = LayoutDiffusionTrainingModule(
        config=config,
        learning_rate=learning_rate,
        scheduler="linear_anneal",
        lr_anneal_steps=lr_anneal_steps,
        auxiliary_loss_weight=auxiliary_loss_weight,
        time_sampler="uniform",
        vocab_file=str(_vendor_s5_vocab_path(dataset_name)),
    )
    missing, unexpected = module.model.load_state_dict(
        reference["model_state_before"], strict=False
    )
    assert missing == []
    assert unexpected == ["position_ids"]
    module._ema_params = {
        name: parameter.detach().clone()
        for name, parameter in module.model.named_parameters()
        if parameter.requires_grad
    }
    optimizer_config = cast(dict[str, object], module.configure_optimizers())
    optimizer = cast(torch.optim.Optimizer, optimizer_config["optimizer"])
    scheduler_config = cast(dict[str, object], optimizer_config["lr_scheduler"])
    lr_scheduler = cast(
        torch.optim.lr_scheduler.LRScheduler, scheduler_config["scheduler"]
    )

    metadata = reference["metadata"]
    assert metadata.get("time_sampler") == "effective_uniform"
    assert metadata.get("vocab_file") == str(_vendor_s5_vocab_path(dataset_name))
    batches = reference["batches"]
    trace_device = str(metadata.get("trace_device", "cpu"))
    if trace_device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for CUDA-generated S3 reference fixtures")
        device = torch.device("cuda")
        module.to(device)
        batches = [
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            for batch in batches
        ]
        cuda_rng_state = reference.get("cuda_rng_state_before_run")
        assert cuda_rng_state is not None
        torch.cuda.set_rng_state(cuda_rng_state, device)
    torch.set_rng_state(reference["rng_state_before_run"])
    package_trajectory = []
    for step, batch in enumerate(batches):
        step_lr = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
        loss = module.training_step(batch, step)
        loss.backward()
        grad_norm = _grad_norm(module.model.parameters())
        optimizer.step()
        module.update_ema()
        lr_scheduler.step()
        package_trajectory.append(
            {
                "step": step,
                "lr": step_lr,
                "train_loss": float(loss.detach().cpu().item()),
                "kl_loss": float(
                    loss.detach().cpu().item()
                    - module.latest_step_trace["aux_loss"].cpu().item()
                ),
                "aux_loss": float(module.latest_step_trace["aux_loss"].cpu().item()),
                "grad_norm": grad_norm,
            }
        )

    for reference_step, package_step in zip(
        reference["trajectory"], package_trajectory, strict=True
    ):
        assert package_step["step"] == reference_step["step"]
        assert package_step["lr"] == pytest.approx(reference_step["lr"], rel=0, abs=0)
        assert package_step["train_loss"] == pytest.approx(
            reference_step["train_loss"], rel=2e-5, abs=2e-4
        )
        assert package_step["kl_loss"] == pytest.approx(
            reference_step["kl_loss"], rel=2e-5, abs=2e-4
        )
        assert package_step["aux_loss"] == pytest.approx(
            reference_step["aux_loss"], rel=2e-5, abs=2e-4
        )
        assert package_step["grad_norm"] == pytest.approx(
            reference_step["grad_norm"], rel=5e-5, abs=2e-4
        )
    for name, reference_value in reference["model_state_after"].items():
        if name == "position_ids":
            continue
        package_value = module.model.state_dict()[name].detach().cpu()
        assert torch.allclose(package_value, reference_value, atol=3e-4, rtol=1e-4), (
            name
        )
    for name, reference_value in reference["ema_state_after"].items():
        package_value = module.ema_state_dict()[name].detach().cpu()
        assert torch.allclose(package_value, reference_value, atol=3e-4, rtol=1e-4), (
            name
        )


@pytest.mark.parametrize("dataset_name", REFERENCE_DATASETS)
def test_s4_processed_stream_first_trained_batch_matches_original_order(
    dataset_name: str,
) -> None:
    seed = 102
    reference = _reference_processed_stream_batch(dataset_name, seed=seed)
    package_root = PROJECT_ROOT / ".cache" / "layoutdiffusion" / "original-data"
    _require_paths(
        [package_root],
        "LayoutDiffusion processed package stream mirror is missing.",
        "Run models/layoutdiffusion/scripts/download_original.py first.",
    )
    torch.manual_seed(seed)
    checked_dataset_name = _checked_dataset_name(dataset_name)
    config = _full_training_config(checked_dataset_name)
    module = LayoutDiffusionTrainingModule(config=config)
    del module
    dm = LayoutDiffusionDataModule(
        dataset_name=checked_dataset_name,
        config=config,
        batch_size=64,
        dataset_source="processed",
        processed_data_dir=str(package_root),
        preconsume_train_batches=1,
        processed_stream_rng_warmup=True,
        num_workers=0,
    )
    package_batch = cast(dict[str, torch.Tensor], next(iter(dm.train_dataloader())))
    assert torch.equal(package_batch["input_ids"], reference)


def _build_vendor_training_components(
    config: LayoutDiffusionConfig,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    _add_vendor_to_path()
    from improved_diffusion.discrete_diffusion import (
        DiffusionTransformer,
    )
    from improved_diffusion.transformer_model import (
        DiscreteTransformerModel,
    )

    bert_config = BertConfig(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
        max_position_embeddings=config.max_position_embeddings,
    )
    model = DiscreteTransformerModel(
        in_channels=8,
        model_channels=config.num_channels,
        num_res_blocks=2,
        dropout=config.dropout,
        config=bert_config,
        training_mode="discrete",
        vocab_size=config.vocab_size,
        matrix_policy=1,
    )
    diffusion = DiffusionTransformer(
        diffusion_step=config.diffusion_steps,
        alpha_init_type=config.noise_schedule,
        auxiliary_loss_weight=1e-3,
        adaptive_auxiliary_loss=True,
        mask_weight=[1, 1],
        matrix_policy=1,
        num_classes=config.vocab_size,
        content_seq_len=config.seq_length,
        pow_num=config.pow_num,
        mul_num=config.mul_num,
    )
    return model, diffusion


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _vendor_s5_vocab_path(dataset_name: str) -> Path:
    stream_name = {
        "rico25": "RICO_ltrb_lex",
        "publaynet": "PublayNet_ltrb_lex",
    }[dataset_name]
    return (
        PROJECT_ROOT
        / ".cache"
        / "layoutdiffusion"
        / "original-data"
        / stream_name
        / "vocab.json"
    )


def _load_required_reference(dataset_name: str) -> ReferenceFixture:
    fixture = reference_fixture_path(dataset_name)
    missing = [path for path in (VENDOR_ROOT, fixture) if not path.exists()]
    if missing:
        skip_or_fail_vendor_parity(
            "LayoutDiffusion training S0-S2 reference fixture is missing; build the "
            "plain-PyTorch TrainLoop adapter fixture before requiring parity.",
            missing_paths=missing,
            regeneration_hint=(
                "CUDA_VISIBLE_DEVICES= uv run --package layoutdiffusion --extra "
                "training python models/layoutdiffusion/tests/vendor_parity/"
                f"layoutdiffusion_training_reference.py --dataset {dataset_name}"
            ),
        )
    return cast(
        ReferenceFixture, torch.load(fixture, map_location="cpu", weights_only=False)
    )


def _load_or_generate_s3_s4_reference(dataset_name: str) -> ReferenceS3S4Fixture:
    fixture = reference_s3_s4_fixture_path(dataset_name)
    if not fixture.exists():
        _require_paths(
            [VENDOR_ROOT],
            "LayoutDiffusion training S3/S4 reference source is missing.",
            (
                "git submodule update --init vendor/ms-layout-generation && "
                "CUDA_VISIBLE_DEVICES= uv run --package layoutdiffusion --extra training "
                "--extra vendor python models/layoutdiffusion/tests/vendor_parity/"
                f"layoutdiffusion_training_reference.py --dataset {dataset_name}"
            ),
        )
        generate_s3_s4_reference_fixture(dataset_name, output_path=fixture)
    return cast(
        ReferenceS3S4Fixture,
        torch.load(fixture, map_location="cpu", weights_only=False),
    )


def _reference_processed_stream_batch(dataset_name: str, *, seed: int) -> torch.Tensor:
    stream_name = {
        "rico25": "RICO_ltrb_lex",
        "publaynet": "PublayNet_ltrb_lex",
    }[dataset_name]
    source_path = (
        PROJECT_ROOT
        / ".cache"
        / "layoutdiffusion"
        / "original"
        / "data"
        / "processed_datasets"
        / stream_name
        / "src1_train.txt"
    )
    _require_paths(
        [source_path],
        "LayoutDiffusion original processed training stream is missing.",
        "Run models/layoutdiffusion/scripts/download_original.py first.",
    )
    config = _full_training_config(_checked_dataset_name(dataset_name))
    tokenizer = build_training_tokenizer(
        config, vocab_file=str(_vendor_s5_vocab_path(dataset_name))
    )
    rows = tokenizer.text_to_token_ids(
        source_path.read_text(encoding="utf-8").splitlines()
    )
    torch.manual_seed(seed)
    module = LayoutDiffusionTrainingModule(config=config)
    del module
    random_embedding = torch.nn.Embedding(config.vocab_size - 1, 8)
    torch.nn.init.normal_(random_embedding.weight)
    del random_embedding
    loader = DataLoader(
        _InputIdsDataset(rows),
        batch_size=64,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    iterator = iter(loader)
    next(iterator)
    return cast(dict[str, torch.Tensor], next(iterator))["input_ids"]


def _checked_dataset_name(dataset_name: str) -> LayoutDiffusionTrainingDatasetName:
    if dataset_name not in REFERENCE_DATASETS:
        raise ValueError(f"Unsupported LayoutDiffusion dataset: {dataset_name}")
    return cast(LayoutDiffusionTrainingDatasetName, dataset_name)


def _full_training_config(
    dataset_name: LayoutDiffusionTrainingDatasetName,
) -> LayoutDiffusionConfig:
    config = LayoutDiffusionConfig(
        dataset_name=dataset_name,
        seq_length=121,
        diffusion_steps=200,
        noise_schedule="gaussian_refine_pow2.5",
        num_channels=128,
        dropout=0.1,
        training_mode="discrete1",
        vocab_size={"rico25": 159, "publaynet": 139}[dataset_name],
    )
    vocab_path = _vendor_s5_vocab_path(dataset_name)
    if vocab_path.exists():
        build_training_tokenizer(config, vocab_file=str(vocab_path))
    return config


def _require_paths(paths: list[Path], reason: str, hint: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        skip_or_fail_vendor_parity(
            reason, missing_paths=missing, regeneration_hint=hint
        )


def _cpu_trace(trace: StepTrace) -> StepTrace:
    return build_step_trace(
        trace.name,
        {key: value.detach().cpu() for key, value in trace.tensors.items()},
        metadata=trace.metadata,
    )


def _without_trace_points(trace: StepTrace, names: set[str]) -> StepTrace:
    return build_step_trace(
        trace.name,
        {key: value for key, value in trace.tensors.items() if key not in names},
        metadata=trace.metadata,
    )


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    sqsum = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            sqsum += float(parameter.grad.detach().pow(2).sum().item())
    return sqsum**0.5

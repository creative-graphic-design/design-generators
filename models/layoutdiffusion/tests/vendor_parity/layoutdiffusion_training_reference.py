"""Generate LayoutDiffusion training reference traces from the original code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import types
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol, cast

import torch
from layoutdiffusion import LayoutDiffusionConfig
from layoutdiffusion.training.dataset import LayoutDiffusionSyntheticDataset
from layoutdiffusion.training.vocab import build_training_tokenizer
from laygen.common.training import LAYOUTDIFFUSION_TRAINING_TRACE_POINTS
from torch.utils.data import DataLoader
from traingen_parity.trace import StepTrace, build_step_trace
from transformers import BertConfig

PROJECT_ROOT = Path(__file__).resolve().parents[4]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "ms-layout-generation" / "LayoutDiffusion"
VENDOR_IMPROVED_DIFFUSION = VENDOR_ROOT / "improved-diffusion"
CACHE_ROOT = PROJECT_ROOT / ".cache" / "layoutdiffusion"
REFERENCE_ROOT = CACHE_ROOT / "training-parity"
PATCHED_VENDOR_IMPROVED_DIFFUSION = CACHE_ROOT / "vendor_run" / "improved-diffusion"
REFERENCE_DATASETS = ("publaynet", "rico25")


class _TraceableOriginalDiffusion(Protocol):
    sample_time: Callable[[int, torch.device, str], tuple[torch.Tensor, torch.Tensor]]
    q_sample: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    predict_start: Callable[
        [torch.Tensor, torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor
    ]
    q_posterior: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


TRACE_POINTS: tuple[str, ...] = LAYOUTDIFFUSION_TRAINING_TRACE_POINTS


def tiny_training_config(dataset_name: str) -> LayoutDiffusionConfig:
    """Return the deterministic tiny config used by S0-S2 reference traces."""
    config = LayoutDiffusionConfig(
        dataset_name=dataset_name,
        seq_length=19,
        max_num_elements=3,
        diffusion_steps=10,
        num_channels=8,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=32,
        dropout=0.0,
        max_position_embeddings=19,
    )
    build_training_tokenizer(config, vocab_file=str(stable_vocab_path(dataset_name)))
    return config


def stable_vocab_path(dataset_name: str) -> Path:
    """Return the stable vendor corpus-order vocab path for a dataset."""
    stream_name = {
        "rico25": "RICO_ltrb_lex",
        "publaynet": "PublayNet_ltrb_lex",
    }[dataset_name]
    return CACHE_ROOT / "original-data" / stream_name / "vocab.json"


def tiny_reference_batch(
    dataset_name: str,
    *,
    batch_size: int = 2,
    preconsume_batches: int = 1,
) -> dict[str, torch.Tensor]:
    """Build the deterministic S0 batch after the vendor loader pre-consumption."""
    return tiny_reference_batches(
        dataset_name,
        batch_size=batch_size,
        preconsume_batches=preconsume_batches,
        train_batches=1,
    )[0]


def tiny_reference_batches(
    dataset_name: str,
    *,
    batch_size: int = 2,
    preconsume_batches: int = 1,
    train_batches: int = 3,
) -> list[dict[str, torch.Tensor]]:
    """Build deterministic train batches after the vendor loader pre-consumption."""
    config = tiny_training_config(dataset_name)
    dataset = LayoutDiffusionSyntheticDataset(
        config=config,
        size=batch_size * (preconsume_batches + train_batches),
        elements=config.max_num_elements,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    iterator = iter(loader)
    for _ in range(preconsume_batches):
        next(iterator)
    batches = []
    for _ in range(train_batches):
        batch = cast(dict[str, torch.Tensor], next(iterator))
        batches.append(dict(batch))
    return batches


def reference_fixture_path(dataset_name: str) -> Path:
    """Return the cache path for one generated S0-S2 reference fixture."""
    return REFERENCE_ROOT / dataset_name / "s0_s2_reference.pt"


def reference_s3_s4_fixture_path(dataset_name: str) -> Path:
    """Return the cache path for one generated S3/S4 reference fixture."""
    return REFERENCE_ROOT / dataset_name / "s3_s4_reference.pt"


def generate_reference_fixture(
    dataset_name: str,
    *,
    output_path: Path | None = None,
    seed: int = 102,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    ema_rate: float = 0.9999,
    auxiliary_loss_weight: float = 1e-3,
    device: str = "cuda",
) -> dict[str, object]:
    """Generate a one-step original-code reference fixture for S0-S2 parity."""
    _add_vendor_to_path()
    from improved_diffusion.discrete_diffusion import (
        DiffusionTransformer,
        index_to_log_onehot,
        log_categorical,
        sum_except_batch,
    )
    from improved_diffusion.nn import update_ema
    from improved_diffusion.transformer_model import (
        DiscreteTransformerModel,
    )

    requested_device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if requested_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    config = tiny_training_config(dataset_name)
    batch = tiny_reference_batch(dataset_name)
    x_start = batch["input_ids"].long().to(requested_device)
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
        auxiliary_loss_weight=auxiliary_loss_weight,
        adaptive_auxiliary_loss=auxiliary_loss_weight != 0,
        mask_weight=[1, 1],
        matrix_policy=1,
        num_classes=config.vocab_size,
        content_seq_len=config.seq_length,
        pow_num=config.pow_num,
        mul_num=config.mul_num,
    )
    model.to(requested_device)
    model.train()
    model.device = x_start.device
    optimizer = torch.optim.AdamW(
        list(model.parameters()), lr=learning_rate, weight_decay=weight_decay
    )
    ema_params = [parameter.detach().clone() for parameter in model.parameters()]

    capture: dict[str, torch.Tensor] = {}
    _install_trace_hooks(diffusion, capture)

    model_state_before = _cpu_state_dict(model.state_dict())
    rng_state_before_step = torch.get_rng_state()
    cuda_rng_state_before_step = (
        torch.cuda.get_rng_state(requested_device)
        if requested_device.type == "cuda"
        else None
    )

    optimizer.zero_grad(set_to_none=True)
    losses = diffusion.training_losses(
        model, x_start, model_kwargs={"input_ids": x_start}
    )
    train_loss = losses["loss"].mean()
    train_loss.backward()
    grad_norm = _grad_norm(model.parameters())
    optimizer.step()
    update_ema(ema_params, list(model.parameters()), rate=ema_rate)

    log_x_start = index_to_log_onehot(x_start, config.vocab_size)
    kl_tokens = diffusion.multinomial_kl(
        capture["log_true_prob"], capture["log_model_prob"]
    )
    mask_region = capture["xt"].eq(config.mask_token_id).float()
    mask_weight = mask_region + (1.0 - mask_region)
    kl = sum_except_batch(kl_tokens * mask_weight)
    decoder_nll = sum_except_batch(
        -log_categorical(log_x_start, capture["log_model_prob"])
    )
    at_zero = capture["t"].eq(torch.zeros_like(capture["t"])).float()
    kl_loss = at_zero * decoder_nll + (1.0 - at_zero) * kl

    trace_tensors = {
        "t": capture["t"].detach().cpu(),
        "pt": capture["pt"].detach().cpu(),
        "xt": capture["xt"].detach().cpu(),
        "log_x_t": capture["log_x_t"].detach().cpu(),
        "log_x0_recon": capture["log_x0_recon"].detach().cpu(),
        "log_model_prob": capture["log_model_prob"].detach().cpu(),
        "log_true_prob": capture["log_true_prob"].detach().cpu(),
        "kl": kl.detach().cpu(),
        "decoder_nll": decoder_nll.detach().cpu(),
        "kl_loss": kl_loss.detach().cpu(),
        "lt_history": diffusion.Lt_history.detach().cpu().clone(),
        "lt_count": diffusion.Lt_count.detach().cpu().clone(),
        "aux_loss": losses.get("loss2", torch.zeros_like(losses["loss"]))
        .mean()
        .detach()
        .cpu(),
        "train_loss": train_loss.detach().cpu(),
    }
    trace = build_step_trace(
        "layoutdiffusion_original_train_loop_step",
        trace_tensors,
        metadata={
            "dataset_name": dataset_name,
            "seed": seed,
            "preconsume_batches": 1,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "ema_rate": ema_rate,
            "auxiliary_loss_weight": auxiliary_loss_weight,
            "time_sampler": "effective_uniform",
            "trace_device": str(requested_device),
            "vocab_file": str(stable_vocab_path(dataset_name)),
            "vocab_sha256": _sha256(stable_vocab_path(dataset_name)),
            "vocab_size": config.vocab_size,
            "id2label": {str(key): value for key, value in config.id2label.items()},
            "patched_vendor_root": str(_patched_vendor_tree()),
            "original_functions": (
                "scripts/train.py:main",
                "improved_diffusion.train_util.TrainLoop.forward_backward",
                "improved_diffusion.discrete_diffusion.DiffusionTransformer.training_losses",
                "improved_diffusion.discrete_diffusion.DiffusionTransformer.sample_time",
                "improved_diffusion.discrete_diffusion.DiffusionTransformer.q_sample",
                "improved_diffusion.discrete_diffusion.DiffusionTransformer.predict_start",
                "improved_diffusion.discrete_diffusion.DiffusionTransformer.q_posterior",
            ),
        },
    )
    fixture: dict[str, object] = {
        "batch": {key: value.detach().cpu() for key, value in batch.items()},
        "trace": trace,
        "trace_tensors": trace.tensors,
        "rng_state_before_step": rng_state_before_step.detach().cpu().clone(),
        "cuda_rng_state_before_step": None
        if cuda_rng_state_before_step is None
        else cuda_rng_state_before_step.detach().cpu().clone(),
        "model_state_before": model_state_before,
        "model_state_after": _cpu_state_dict(model.state_dict()),
        "ema_state_after": [
            parameter.detach().cpu().clone() for parameter in ema_params
        ],
        "optimizer_state_after": optimizer.state_dict(),
        "grad_norm": torch.tensor(grad_norm),
        "metadata": trace.metadata,
    }
    destination = output_path or reference_fixture_path(dataset_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, destination)
    return fixture


def generate_s3_s4_reference_fixture(
    dataset_name: str,
    *,
    output_path: Path | None = None,
    seed: int = 102,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    ema_rate: float = 0.9999,
    auxiliary_loss_weight: float = 1e-3,
    lr_anneal_steps: int = 4,
    steps: int = 3,
    device: str = "cuda",
) -> dict[str, object]:
    """Generate a repeated-step original-code reference fixture for S3/S4 parity."""
    _install_single_rank_mpi_stub()
    _add_vendor_to_path()
    from improved_diffusion import dist_util, logger
    from improved_diffusion.discrete_diffusion import DiffusionTransformer
    from improved_diffusion.train_util import TrainLoop
    from improved_diffusion.transformer_model import DiscreteTransformerModel

    requested_device = torch.device(device if torch.cuda.is_available() else "cpu")
    dist_util.setup_dist()
    if requested_device.type == "cuda":
        requested_device = dist_util.dev()
        torch.cuda.set_device(requested_device)
    logger.configure(dir=str(REFERENCE_ROOT / dataset_name / "vendor-trainloop-log"))
    torch.manual_seed(seed)
    if requested_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    config = tiny_training_config(dataset_name)
    batches = tiny_reference_batches(dataset_name, train_batches=steps)
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
        auxiliary_loss_weight=auxiliary_loss_weight,
        adaptive_auxiliary_loss=auxiliary_loss_weight != 0,
        mask_weight=[1, 1],
        matrix_policy=1,
        num_classes=config.vocab_size,
        content_seq_len=config.seq_length,
        pow_num=config.pow_num,
        mul_num=config.mul_num,
    )
    model.to(requested_device)
    model.train()
    model.device = requested_device

    captured_losses: list[dict[str, torch.Tensor]] = []
    original_training_losses = diffusion.training_losses

    def capture_training_losses(
        model_arg: torch.nn.Module,
        x_start: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        losses = original_training_losses(model_arg, x_start, *args, **kwargs)
        captured_losses.append(
            {key: value.detach().cpu().clone() for key, value in losses.items()}
        )
        return losses

    diffusion.training_losses = capture_training_losses
    loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=iter(()),
        batch_size=batches[0]["input_ids"].shape[0],
        microbatch=batches[0]["input_ids"].shape[0],
        lr=learning_rate,
        ema_rate=ema_rate,
        log_interval=steps + 1,
        save_interval=steps + 1,
        resume_checkpoint="",
        schedule_sampler=None,
        weight_decay=weight_decay,
        lr_anneal_steps=lr_anneal_steps,
        checkpoint_path=str(
            REFERENCE_ROOT / dataset_name / "vendor-trainloop-checkpoints"
        ),
        gradient_clipping=-1,
        training_mode="discrete",
    )

    model_state_before = _cpu_state_dict(model.state_dict())
    rng_state_before_run = torch.get_rng_state()
    cuda_rng_state_before_run = (
        torch.cuda.get_rng_state(requested_device)
        if requested_device.type == "cuda"
        else None
    )
    trajectory: list[dict[str, float | int]] = []
    for step, batch in enumerate(batches):
        x_start = batch["input_ids"].long()
        model.device = requested_device
        captured_losses.clear()
        loop.run_step(x_start, {"input_ids": x_start})
        losses = captured_losses[-1]
        step_lr = float(loop.opt.param_groups[0]["lr"])
        trajectory.append(
            {
                "step": step,
                "lr": step_lr,
                "train_loss": float(losses["loss"].mean().item()),
                "kl_loss": float(losses["loss1"].mean().item()),
                "aux_loss": float(
                    losses.get("loss2", torch.zeros_like(losses["loss"])).mean().item()
                ),
                "grad_norm": float(_grad_norm(model.parameters())),
            }
        )
        loop.step += 1

    named_parameters = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    fixture: dict[str, object] = {
        "batches": [
            {key: value.detach().cpu() for key, value in batch.items()}
            for batch in batches
        ],
        "trajectory": trajectory,
        "rng_state_before_run": rng_state_before_run.detach().cpu().clone(),
        "cuda_rng_state_before_run": None
        if cuda_rng_state_before_run is None
        else cuda_rng_state_before_run.detach().cpu().clone(),
        "model_state_before": model_state_before,
        "model_state_after": _cpu_state_dict(model.state_dict()),
        "ema_state_after": {
            name: value.detach().cpu().clone()
            for name, value in zip(named_parameters, loop.ema_params[0], strict=True)
        },
        "optimizer_state_after": loop.opt.state_dict(),
        "metadata": {
            "dataset_name": dataset_name,
            "seed": seed,
            "preconsume_batches": 1,
            "steps": steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "ema_rate": ema_rate,
            "auxiliary_loss_weight": auxiliary_loss_weight,
            "lr_anneal_steps": lr_anneal_steps,
            "time_sampler": "effective_uniform",
            "trace_device": str(requested_device),
            "vocab_file": str(stable_vocab_path(dataset_name)),
            "vocab_sha256": _sha256(stable_vocab_path(dataset_name)),
            "vocab_size": config.vocab_size,
            "id2label": {str(key): value for key, value in config.id2label.items()},
            "patched_vendor_root": str(_patched_vendor_tree()),
            "original_functions": (
                "improved_diffusion.train_util.TrainLoop.run_step",
                "improved_diffusion.train_util.TrainLoop.forward_backward",
                "improved_diffusion.train_util.TrainLoop.optimize_normal",
                "improved_diffusion.discrete_diffusion.DiffusionTransformer.training_losses",
            ),
        },
    }
    destination = output_path or reference_s3_s4_fixture_path(dataset_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, destination)
    return fixture


def _install_single_rank_mpi_stub() -> None:
    mpi_module = types.ModuleType("mpi4py")

    class _Comm:
        rank = 0
        size = 1

        def Get_rank(self) -> int:
            return 0

        def Get_size(self) -> int:
            return 1

        def bcast(self, value: object, root: int = 0) -> object:
            del root
            return value

    class _MPI:
        COMM_WORLD = _Comm()

    setattr(mpi_module, "MPI", _MPI())  # noqa: B010
    sys.modules.setdefault("mpi4py", mpi_module)


def _add_vendor_to_path() -> None:
    vendor_work = _patched_vendor_tree()
    sys.path = [path for path in sys.path if path != str(VENDOR_IMPROVED_DIFFUSION)]
    if str(vendor_work) not in sys.path:
        sys.path.insert(0, str(vendor_work))
    _drop_unpatched_vendor_modules(vendor_work)


def _drop_unpatched_vendor_modules(vendor_work: Path) -> None:
    vendor_root = vendor_work.resolve()
    for name, module in list(sys.modules.items()):
        if name != "improved_diffusion" and not name.startswith("improved_diffusion."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            del sys.modules[name]
            continue
        if not Path(module_file).resolve().is_relative_to(vendor_root):
            del sys.modules[name]


def _patched_vendor_tree() -> Path:
    if PATCHED_VENDOR_IMPROVED_DIFFUSION.exists():
        return PATCHED_VENDOR_IMPROVED_DIFFUSION
    if not VENDOR_IMPROVED_DIFFUSION.exists():
        raise FileNotFoundError(
            f"Missing LayoutDiffusion original source: {VENDOR_IMPROVED_DIFFUSION}. "
            "Run `git submodule update --init vendor/ms-layout-generation`."
        )
    script_path = (
        PROJECT_ROOT
        / "models"
        / "layoutdiffusion"
        / "scripts"
        / "generate_reference_outputs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "layoutdiffusion_generate_reference_outputs", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(Path, module.prepare_vendor_tree(VENDOR_IMPROVED_DIFFUSION, CACHE_ROOT))


def _install_trace_hooks(
    diffusion: _TraceableOriginalDiffusion, capture: dict[str, torch.Tensor]
) -> None:
    from improved_diffusion.discrete_diffusion import (
        log_onehot_to_index,
    )

    sample_time = diffusion.sample_time
    q_sample = diffusion.q_sample
    predict_start = diffusion.predict_start
    q_posterior = diffusion.q_posterior
    posterior_calls = 0

    def traced_sample_time(b: int, device: torch.device, method: str = "uniform"):
        t, pt = sample_time(b, device, method)
        capture["t"] = t.detach().clone()
        capture["pt"] = pt.detach().clone()
        return t, pt

    def traced_q_sample(log_x_start: torch.Tensor, t: torch.Tensor):
        log_x_t = q_sample(log_x_start, t)
        capture["log_x_t"] = log_x_t.detach().clone()
        capture["xt"] = log_onehot_to_index(log_x_t).detach().clone()
        return log_x_t

    def traced_predict_start(
        log_x_t: torch.Tensor, model: torch.nn.Module, y: torch.Tensor, t: torch.Tensor
    ):
        log_x0_recon = predict_start(log_x_t, model, y, t)
        capture["log_x0_recon"] = log_x0_recon.detach().clone()
        return log_x0_recon

    def traced_q_posterior(
        log_x_start: torch.Tensor, log_x_t: torch.Tensor, t: torch.Tensor
    ):
        nonlocal posterior_calls
        value = q_posterior(log_x_start, log_x_t, t)
        if posterior_calls == 0:
            capture["log_model_prob"] = value.detach().clone()
        elif posterior_calls == 1:
            capture["log_true_prob"] = value.detach().clone()
        posterior_calls += 1
        return value

    diffusion.sample_time = traced_sample_time
    diffusion.q_sample = traced_q_sample
    diffusion.predict_start = traced_predict_start
    diffusion.q_posterior = traced_q_posterior


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    sqsum = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            sqsum += float(parameter.grad.detach().pow(2).sum().item())
    return sqsum**0.5


def _cpu_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=REFERENCE_DATASETS,
        dest="datasets",
        help="Dataset fixture to generate. Repeat to generate multiple datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REFERENCE_ROOT,
        help="Root cache directory for generated reference fixtures.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for the original reference trace.",
    )
    parser.add_argument(
        "--include-s3-s4",
        action="store_true",
        help="Also regenerate the repeated-step S3/S4 reference fixture.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = args.datasets or list(REFERENCE_DATASETS)
    for dataset_name in datasets:
        path = args.output_root / dataset_name / "s0_s2_reference.pt"
        fixture = generate_reference_fixture(
            dataset_name, output_path=path, device=args.device
        )
        trace = cast(StepTrace, fixture["trace"])
        print(f"wrote {path}")
        print(f"trace points: {', '.join(sorted(trace.tensors))}")
        if args.include_s3_s4:
            s3_s4_path = args.output_root / dataset_name / "s3_s4_reference.pt"
            generate_s3_s4_reference_fixture(
                dataset_name, output_path=s3_s4_path, device=args.device
            )
            print(f"wrote {s3_s4_path}")


if __name__ == "__main__":
    main()

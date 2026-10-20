from typing import cast

import pytest
import torch

pytest.importorskip("lightning")

from layoutformerpp import (
    LayoutFormerPPForConditionalGeneration,
)
from layoutformerpp.training.lightning_module import (
    LayoutFormerPPTrainingConfig,
    LayoutFormerPPTrainingModule,
    vendor_effective_cross_entropy,
)
from layoutformerpp.training.scheduler import LayoutFormerPPWarmupLR

pytestmark = pytest.mark.training


def tiny_module() -> LayoutFormerPPTrainingModule:
    config: LayoutFormerPPTrainingConfig = {
        "dataset": "rico25",
        "task": "gen_t",
        "vocab_size": 159,
        "max_position_embeddings": 150,
        "d_model": 8,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "dim_feedforward": 16,
        "dropout": 0.0,
    }
    return LayoutFormerPPTrainingModule(recipe_name="rico25_label", config=config)


def test_s0_training_module_owns_genuine_model_and_explicit_shift() -> None:
    module = tiny_module()
    assert type(module.model) is LayoutFormerPPForConditionalGeneration
    labels = torch.tensor([[8, 2, 2]])
    assert module.model.prepare_decoder_input_ids_from_labels(labels).tolist() == [
        [0, 8, 2]
    ]


def test_s0_effective_loss_includes_padding_without_changing_runtime_loss() -> None:
    logits = torch.tensor([[[5.0, 0.0, -2.0], [5.0, 0.0, -2.0]]])
    labels = torch.tensor([[0, 2]])
    pad_inclusive = vendor_effective_cross_entropy(logits, labels)
    pad_masked = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 3),
        torch.tensor([0, -100]),
        ignore_index=-100,
    )
    assert pad_inclusive > pad_masked

    module = tiny_module()
    input_ids = torch.tensor([[5, 1]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    runtime = module.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=torch.tensor([[5, module.model.pad_token_id]]),
    )
    assert runtime.loss is not None
    runtime_logits = runtime.logits.detach()
    masked_targets = torch.tensor([[5, -100]])
    masked_loss = torch.nn.functional.cross_entropy(
        runtime_logits.reshape(-1, runtime_logits.size(-1)),
        masked_targets.reshape(-1),
        ignore_index=-100,
    )
    changed_pad_logits = runtime_logits.clone()
    changed_pad_logits[:, 1, :] = torch.linspace(
        -100.0,
        100.0,
        runtime_logits.size(-1),
    )
    changed_masked_loss = torch.nn.functional.cross_entropy(
        changed_pad_logits.reshape(-1, changed_pad_logits.size(-1)),
        masked_targets.reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(runtime.loss, masked_loss)
    torch.testing.assert_close(masked_loss, changed_masked_loss)
    assert not torch.isclose(
        vendor_effective_cross_entropy(runtime_logits, targets=torch.tensor([[5, 2]])),
        vendor_effective_cross_entropy(
            changed_pad_logits,
            targets=torch.tensor([[5, 2]]),
        ),
    )


@pytest.mark.parametrize("warmup_steps", [1000, 2000, 3000, 4000])
def test_s0_adam_and_scheduler_static_state(warmup_steps: int) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    scheduler = LayoutFormerPPWarmupLR(
        optimizer,
        warmup_num_steps=warmup_steps,
        warmup_max_lr=1e-4,
    )

    assert scheduler.last_batch_iteration == -1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["weight_decay"] == 0
    assert optimizer.defaults["amsgrad"] is False

    scheduler.step()
    assert scheduler.last_batch_iteration == 0
    assert scheduler.get_last_lr() == [pytest.approx(0.0)]
    scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(
        1e-4
        * torch.log(torch.tensor(2.0)).item()
        / torch.log(torch.tensor(float(warmup_steps))).item()
    )


def test_s0_configure_optimizers_pins_post_update_interval() -> None:
    module = tiny_module()
    configured = cast(dict[str, object], module.configure_optimizers())
    optimizer = configured["optimizer"]
    scheduler_config = cast(dict[str, object], configured["lr_scheduler"])
    scheduler = scheduler_config["scheduler"]
    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(scheduler, LayoutFormerPPWarmupLR)
    assert scheduler_config["interval"] == "step"
    assert scheduler.last_batch_iteration == -1

from functools import partial
import importlib
from pathlib import Path
import sys

import pytest
import torch

from dlt import DLTConfig
from dlt.training.dataset import SyntheticDLTDataset, collate_dlt_batch
from dlt.training.losses import masked_cross_entropy, masked_l2

pytest.importorskip("lightning")

from dlt.training.datamodule import DLTDataModule
from dlt.training.lightning_module import DLTTrainingModule
from dlt.training.parity import DLTSyntheticStepTraceAdapter


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def tiny_training_module() -> DLTTrainingModule:
    config = DLTConfig(
        dataset_name="publaynet",
        max_num_comp=4,
        categories_num=7,
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        cond_emb_size=12,
        cat_emb_size=8,
        num_cont_timesteps=4,
        num_discrete_steps=2,
    )
    return DLTTrainingModule(
        config=config,
        optimizer=partial(
            torch.optim.AdamW,
            lr=0.0001,
            betas=(0.95, 0.999),
            eps=1e-8,
            weight_decay=1e-6,
        ),
    )


def _load_vendor_modules():
    pytest.importorskip("labml_nn")
    pytest.importorskip("cv2")
    vendor_root = Path(__file__).resolve().parents[4] / "vendor/dlt/dlt"
    sys.path.insert(0, str(vendor_root))
    importlib.invalidate_caches()
    return (
        importlib.import_module("diffusion"),
        importlib.import_module("models.dlt"),
        importlib.import_module("utils"),
    )


@pytest.mark.vendor_parity
@pytest.mark.training
def test_dlt_package_model_topology_matches_vendor() -> None:
    _, vendor_modeling, _ = _load_vendor_modules()
    cases = [
        (
            "tiny-test",
            {
                "categories_num": 7,
                "latent_dim": 32,
                "num_layers": 1,
                "num_heads": 4,
                "dropout_r": 0.0,
                "activation": "gelu",
                "cond_emb_size": 12,
                "cat_emb_size": 8,
            },
            11_187,
            29,
        ),
        (
            "full-publaynet",
            {
                "categories_num": 7,
                "latent_dim": 512,
                "num_layers": 4,
                "num_heads": 8,
                "dropout_r": 0.0,
                "activation": "gelu",
                "cond_emb_size": 224,
                "cat_emb_size": 64,
            },
            8_944_459,
            65,
        ),
    ]
    for name, kwargs, expected_params, expected_state_keys in cases:
        torch.manual_seed(17)
        module = DLTTrainingModule(
            config=DLTConfig(dataset_name="publaynet", **kwargs),
            optimizer=partial(torch.optim.AdamW, lr=0.0001),
        )
        torch.manual_seed(17)
        vendor_model = vendor_modeling.DLT(**kwargs)
        package_state = module.model.state_dict()
        vendor_state = vendor_model.state_dict()

        assert _parameter_count(module.model) == expected_params, name
        assert _parameter_count(vendor_model) == expected_params, name
        assert len(package_state) == expected_state_keys, name
        assert len(vendor_state) == expected_state_keys, name
        assert set(package_state) == set(vendor_state), name
        for key in package_state:
            torch.testing.assert_close(
                package_state[key], vendor_state[key], rtol=0, atol=0
            )


@pytest.mark.vendor_parity
@pytest.mark.training
def test_dlt_synthetic_training_trace_adapter() -> None:
    data = DLTDataModule(batch_size=2, length=2, max_num_comp=4, categories_num=7)
    batch = next(iter(data.train_dataloader()))
    trace = DLTSyntheticStepTraceAdapter().trace_training_step(
        tiny_training_module(), batch
    )
    assert "loss" in trace.tensors
    assert "pred_box" in trace.tensors


@pytest.mark.vendor_parity
@pytest.mark.training
def test_dlt_real_vendor_s0_s2_training_trace_matches() -> None:
    vendor_diffusion, vendor_modeling, vendor_utils = _load_vendor_modules()
    config = DLTConfig(
        dataset_name="publaynet",
        max_num_comp=4,
        categories_num=7,
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        cond_emb_size=12,
        cat_emb_size=8,
        num_cont_timesteps=4,
        num_discrete_steps=2,
    )
    batch = collate_dlt_batch(
        [SyntheticDLTDataset(length=1)[0], SyntheticDLTDataset(length=1, seed=1)[0]]
    )

    torch.manual_seed(17)
    vendor_model = vendor_modeling.DLT(
        categories_num=config.categories_num,
        latent_dim=config.latent_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout_r=config.dropout_r,
        activation=config.activation,
        cond_emb_size=config.cond_emb_size,
        cat_emb_size=config.cat_emb_size,
    )
    vendor_scheduler = vendor_diffusion.JointDiffusionScheduler(
        alpha=0.0,
        seq_max_length=config.max_num_comp,
        device="cpu",
        discrete_features_names=[("cat", config.categories_num)],
        num_discrete_steps=[config.num_discrete_steps],
        num_train_timesteps=config.num_cont_timesteps,
        beta_schedule=config.beta_schedule,
        prediction_type="sample",
        clip_sample=False,
    )

    torch.manual_seed(17)
    module = DLTTrainingModule(
        config=config,
        optimizer=partial(torch.optim.AdamW, lr=0.0001),
    )
    module.model.eval()
    vendor_model.eval()
    assert set(vendor_model.state_dict()) == set(module.model.state_dict())

    torch.manual_seed(23)
    module.training_step({key: value.clone() for key, value in batch.items()}, 0)
    ours = module.latest_step_trace

    torch.manual_seed(23)
    vendor_batch = {key: value.clone() for key, value in batch.items()}
    noise = torch.randn(vendor_batch["box"].shape)
    timesteps = torch.randint(
        0, vendor_scheduler.num_cont_steps, (vendor_batch["box"].shape[0],)
    ).long()
    cont_vec, noisy_batch = vendor_scheduler.add_noise_jointly(
        vendor_batch["box"], vendor_batch, timesteps, noise
    )
    noisy_batch["box"] = cont_vec
    pred_box, pred_cat = vendor_model(vendor_batch, noisy_batch, timesteps)
    vendor_l2 = vendor_utils.masked_l2(
        vendor_batch["box_cond"], pred_box, vendor_batch["mask_box"]
    )
    vendor_ce = vendor_utils.masked_cross_entropy(
        pred_cat, vendor_batch["cat"], vendor_batch["mask_cat"]
    )
    vendor_loss = (5.0 * vendor_l2 + vendor_ce).mean()

    torch.testing.assert_close(ours["noise"], noise, rtol=0, atol=0)
    torch.testing.assert_close(ours["t"], timesteps, rtol=0, atol=0)
    torch.testing.assert_close(ours["noised_box"], cont_vec, rtol=0, atol=0)
    torch.testing.assert_close(ours["noised_cat"], noisy_batch["cat"], rtol=0, atol=0)
    torch.testing.assert_close(ours["pred_box"], pred_box, rtol=0, atol=0)
    torch.testing.assert_close(ours["pred_cat"], pred_cat, rtol=0, atol=0)
    torch.testing.assert_close(
        masked_l2(vendor_batch["box_cond"], pred_box, vendor_batch["mask_box"]),
        vendor_l2,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        masked_cross_entropy(pred_cat, vendor_batch["cat"], vendor_batch["mask_cat"]),
        vendor_ce,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(ours["masked_l2"], vendor_l2, rtol=0, atol=0)
    torch.testing.assert_close(ours["masked_ce"], vendor_ce, rtol=0, atol=0)
    torch.testing.assert_close(ours["loss"], vendor_loss, rtol=0, atol=0)

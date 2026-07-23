import torch

from cgb_dm.training import CGBDMDataModule, CGBDMTrainingModule
from cgb_dm.training.parity import CGBDMStepTraceAdapter
from cgb_dm.training.seed import apply_seed_mode


def test_training_step_records_required_trace():
    module = CGBDMTrainingModule(
        config={
            "max_seq_length": 2,
            "image_size": (32, 32),
            "dim_model": 16,
            "n_head": 2,
            "feature_dim": 32,
            "num_layers": 1,
            "num_train_timesteps": 10,
            "ddim_num_steps": 1,
        }
    )
    batch = {
        "pixel_values": torch.zeros(1, 4, 32, 32),
        "layout": torch.zeros(1, 2, 8),
        "saliency_box": torch.zeros(1, 1, 4),
    }
    batch["layout"][:, :, 0] = 1

    loss = module.training_step(batch, 0)

    assert loss.ndim == 0
    assert CGBDMStepTraceAdapter.trace_points[-1] == "loss"
    assert set(CGBDMStepTraceAdapter.trace_points) <= set(module.latest_step_trace)


def test_datamodule_synthetic_batch():
    datamodule = CGBDMDataModule(
        config={
            "max_seq_length": 2,
            "image_size": (32, 32),
            "dim_model": 16,
            "n_head": 2,
            "feature_dim": 32,
            "num_layers": 1,
        },
        source="synthetic",
        batch_size=2,
    )
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["pixel_values"].shape == (2, 4, 32, 32)
    assert batch["layout"].shape == (2, 2, 8)


def test_datamodule_original_requires_root_and_optimizer_config():
    datamodule = CGBDMDataModule(source="original_zip")
    try:
        datamodule.setup()
    except ValueError as exc:
        assert "data_root is required" in str(exc)
    else:
        raise AssertionError("expected data_root error")

    module = CGBDMTrainingModule(
        config={
            "max_seq_length": 2,
            "image_size": (32, 32),
            "dim_model": 16,
            "n_head": 2,
            "feature_dim": 32,
            "num_layers": 1,
        }
    )
    optimizers = module.configure_optimizers()
    assert "optimizer" in optimizers
    assert apply_seed_mode("default")["mode"] == "default"
    assert apply_seed_mode("deterministic", seed=2)["deterministic_algorithms"] is True

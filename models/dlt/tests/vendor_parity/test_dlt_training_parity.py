import pytest

from dlt import DLTConfig
from dlt.training.config import DLTOptimizerConfig

pytest.importorskip("lightning")

from dlt.training.datamodule import DLTDataModule
from dlt.training.lightning_module import DLTTrainingModule
from dlt.training.parity import DLTSyntheticStepTraceAdapter


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
    return DLTTrainingModule(config=config, optimizer_config=DLTOptimizerConfig())


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

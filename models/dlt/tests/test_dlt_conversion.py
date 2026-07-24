import tempfile

from dlt import DLT, DLTConfig, DLTPipeline
from dlt.conversion import build_pipeline, convert_save_pretrained_directory


def test_convert_save_pretrained_directory() -> None:
    config = DLTConfig(
        dataset_name="publaynet",
        max_num_comp=3,
        categories_num=7,
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        cond_emb_size=12,
        cat_emb_size=8,
        num_cont_timesteps=3,
        num_discrete_steps=2,
    )
    pipe = build_pipeline(config)
    with (
        tempfile.TemporaryDirectory() as model_dir,
        tempfile.TemporaryDirectory() as out_dir,
    ):
        pipe.model.save_pretrained(model_dir)
        converted = convert_save_pretrained_directory(model_dir, out_dir, config=config)
        loaded = DLTPipeline.from_pretrained(out_dir)
    assert isinstance(converted.model, DLT)
    assert loaded.dlt_config.dataset_name == "publaynet"

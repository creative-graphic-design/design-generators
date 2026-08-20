from pathlib import Path
from typing import cast

import pytest
import yaml
from lightning.pytorch.cli import LightningCLI

from laygen.common import ConditionType, DatasetName

from layoutformerpp import LayoutFormerPPTask
from layoutformerpp.labels import (
    RICO25_LABEL_TRANSLATION,
    build_label_translation,
    normalize_label_name,
)
from layoutformerpp.training import TRAINING_RECIPES, TRAINING_RECIPES_BY_NAME


CONDITIONS = tuple(
    ConditionType(name)
    for name in (
        "label",
        "label_size",
        "relation",
        "refinement",
        "completion",
        "unconditional",
    )
)


def test_s0_training_recipe_matrix_is_complete_and_immutable() -> None:
    assert len(TRAINING_RECIPES) == 12
    assert set(TRAINING_RECIPES) == {
        (dataset, condition)
        for dataset in (DatasetName.rico25, DatasetName.publaynet)
        for condition in CONDITIONS
    }
    assert set(TRAINING_RECIPES_BY_NAME) == {
        f"{dataset}_{condition}"
        for dataset in (DatasetName.rico25, DatasetName.publaynet)
        for condition in CONDITIONS
    }
    expected_task_ids = {
        LayoutFormerPPTask.refinement: 0,
        LayoutFormerPPTask.completion: 1,
        LayoutFormerPPTask.ugen: 2,
        LayoutFormerPPTask.gen_t: 3,
        LayoutFormerPPTask.gen_ts: 4,
        LayoutFormerPPTask.gen_r: 5,
    }
    for recipe in TRAINING_RECIPES.values():
        assert recipe.trainer_mode == "basic"
        assert recipe.learning_rate == 1e-4
        assert recipe.gradient_accumulation == 1
        assert recipe.loss_mode == "vendor_effective_cross_entropy"
        assert recipe.scheduler_timing == "post_optimizer_step"
        assert recipe.precision == "32-true"
        assert not recipe.use_gradient_clipping
        assert not recipe.use_ema
        assert not recipe.use_amp
        assert recipe.task_ids == tuple(
            expected_task_ids[task] for task in recipe.tasks
        )
        assert recipe.canonical_hub_id.startswith(
            f"creative-graphic-design/layoutformerpp-{recipe.dataset}-"
        )


def test_s0_publaynet_relation_keeps_effective_six_task_recipe() -> None:
    recipe = TRAINING_RECIPES[(DatasetName.publaynet, ConditionType.relation)]
    assert recipe.tasks == (
        LayoutFormerPPTask.refinement,
        LayoutFormerPPTask.gen_ts,
        LayoutFormerPPTask.gen_t,
        LayoutFormerPPTask.completion,
        LayoutFormerPPTask.ugen,
        LayoutFormerPPTask.gen_r,
    )
    assert recipe.task_ids == (0, 4, 3, 1, 2, 5)
    assert recipe.eval_tasks == (LayoutFormerPPTask.gen_r,)
    assert recipe.partition_buckets == (-1, -1, -2, 0, 0, -3)
    assert {"add_task_prompt", "partition_training_data"} <= set(
        recipe.serialization_flags
    )
    assert recipe.vocab_size == 178


def test_s0_all_rico25_families_share_reviewed_dual_map() -> None:
    expected = {
        0: 1,
        1: 2,
        2: 3,
        3: 5,
        4: 4,
        5: 8,
        6: 11,
        7: 9,
        8: 7,
        9: 13,
        10: 12,
        11: 14,
        12: 10,
        13: 17,
        14: 16,
        15: 19,
        16: 18,
        17: 21,
        18: 6,
        19: 24,
        20: 15,
        21: 25,
        22: 20,
        23: 22,
        24: 23,
    }
    assert dict(RICO25_LABEL_TRANSLATION.public_id2label).keys() == set(range(25))
    assert dict(RICO25_LABEL_TRANSLATION.sequence_id2label).keys() == set(range(1, 26))
    assert dict(RICO25_LABEL_TRANSLATION.public_to_sequence) == expected
    assert dict(RICO25_LABEL_TRANSLATION.sequence_to_public) == {
        sequence_id: public_id for public_id, sequence_id in expected.items()
    }
    for condition in CONDITIONS:
        recipe = TRAINING_RECIPES[(DatasetName.rico25, condition)]
        assert recipe.label_translation_sha256 == RICO25_LABEL_TRANSLATION.sha256


def test_s0_label_map_normalization_and_invalid_maps_fail_closed() -> None:
    assert normalize_label_name("  ＴＥＸＴ\t Button  ") == "text button"
    with pytest.raises(ValueError, match="contiguous from zero"):
        build_label_translation({1: "Text"}, {1: "Text"})
    with pytest.raises(ValueError, match="normalization collision"):
        build_label_translation(
            {0: "Text", 1: " text "},
            {1: "Text", 2: "Image"},
        )
    with pytest.raises(ValueError, match="sets differ"):
        build_label_translation(
            {0: "Text", 1: "Image"},
            {1: "Text", 2: "Icon"},
        )


def test_s0_twelve_standalone_yamls_match_registry() -> None:
    config_dir = Path("models/layoutformerpp/configs/training")
    paths = sorted(config_dir.glob("*.yaml"))
    assert [path.stem for path in paths] == sorted(TRAINING_RECIPES_BY_NAME)
    invariant_wiring: set[tuple[object, ...]] = set()
    for path in paths:
        recipe = TRAINING_RECIPES_BY_NAME[path.stem]
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        trainer = data["trainer"]
        init_args = data["model"]["init_args"]
        config_args = init_args["config"]
        assert data["seed_everything"] is False
        assert data["model"]["class_path"] == (
            "layoutformerpp.training.lightning_module.LayoutFormerPPTrainingModule"
        )
        assert trainer["max_epochs"] == recipe.epochs
        assert trainer["check_val_every_n_epoch"] == recipe.eval_interval
        assert trainer["accumulate_grad_batches"] == recipe.gradient_accumulation
        assert trainer["gradient_clip_val"] is None
        assert trainer["precision"] == recipe.precision
        checkpoint = trainer["callbacks"][0]["init_args"]
        assert checkpoint == {
            "monitor": "val_loss",
            "mode": "min",
            "save_top_k": 1,
            "save_last": False,
        }
        assert init_args["recipe_name"] == recipe.name
        assert config_args["dataset"] == str(recipe.dataset)
        assert config_args["vocab_size"] == recipe.vocab_size
        assert config_args["max_position_embeddings"] == recipe.max_position_embeddings
        assert data["data"] == {
            "class_path": (
                "layoutformerpp.training.lightning_module.LayoutFormerPPDataModule"
            ),
            "init_args": {"recipe_name": recipe.name},
        }
        invariant_wiring.add(
            (
                trainer["accelerator"],
                trainer["devices"],
                trainer["precision"],
                trainer["accumulate_grad_batches"],
                trainer["gradient_clip_val"],
                trainer["num_sanity_val_steps"],
                trainer["use_distributed_sampler"],
                data["model"]["class_path"],
                data["data"]["class_path"],
                data["trainer"]["callbacks"][0]["class_path"],
                tuple(sorted(checkpoint.items())),
            )
        )
    assert len(invariant_wiring) == 1


@pytest.mark.parametrize("config_name", sorted(TRAINING_RECIPES_BY_NAME))
def test_s0_twelve_yamls_construct_with_bounded_lightning_cli(
    config_name: str,
) -> None:
    path = Path("models/layoutformerpp/configs/training") / f"{config_name}.yaml"
    cli = LightningCLI(
        model_class=None,
        datamodule_class=None,
        subclass_mode_model=True,
        subclass_mode_data=True,
        args=[
            "--config",
            str(path),
            "--trainer.accelerator=cpu",
            "--trainer.devices=1",
            "--trainer.enable_model_summary=false",
            "--model.init_args.config.d_model=8",
            "--model.init_args.config.encoder_layers=1",
            "--model.init_args.config.decoder_layers=1",
            "--model.init_args.config.encoder_attention_heads=2",
            "--model.init_args.config.decoder_attention_heads=2",
            "--model.init_args.config.dim_feedforward=16",
            "--model.init_args.config.dropout=0.0",
        ],
        run=False,
    )
    module = cast(object, cli.model)
    assert module.__class__.__name__ == "LayoutFormerPPTrainingModule"
    assert cli.datamodule.__class__.__name__ == "LayoutFormerPPDataModule"

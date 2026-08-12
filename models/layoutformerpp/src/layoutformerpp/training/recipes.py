"""Immutable training recipes for LayoutFormer++ checkpoint families."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from laygen.common import ConditionType, DatasetName

from ..labels import label_translation_for_dataset
from ..tasks import LayoutFormerPPTask


@dataclass(frozen=True, slots=True)
class LayoutFormerPPTrainingRecipe:
    """Static training identity for one dataset and condition family."""

    name: str
    dataset: DatasetName
    condition: ConditionType
    tasks: tuple[LayoutFormerPPTask, ...]
    task_ids: tuple[int, ...]
    eval_tasks: tuple[LayoutFormerPPTask, ...]
    epochs: int
    batch_size: int
    eval_batch_size: int
    max_position_embeddings: int
    decode_max_length: int
    warmup_num_steps: int
    eval_seed: int
    eval_interval: int
    vocab_size: int
    serialization_flags: frozenset[str]
    partition_buckets: tuple[int, ...] = ()
    learning_rate: float = 1e-4
    num_layers: int = 8
    attention_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1
    max_num_elements: int = 20
    discrete_x_grid: int = 128
    discrete_y_grid: int = 128
    gradient_accumulation: int = 1
    trainer_mode: str = "basic"
    precision: str = "32-true"
    loss_mode: str = "vendor_effective_cross_entropy"
    scheduler_timing: str = "post_optimizer_step"
    use_gradient_clipping: bool = False
    use_ema: bool = False
    use_amp: bool = False

    @property
    def label_translation_sha256(self) -> str:
        """Return the semantic label-map fingerprint for this recipe."""
        return label_translation_for_dataset(self.dataset).sha256

    @property
    def canonical_hub_id(self) -> str:
        """Return the canonical public checkpoint identity."""
        suffix = self.condition.replace("_", "-")
        return f"creative-graphic-design/layoutformerpp-{self.dataset}-{suffix}"


_RICO_COMMON: Final[frozenset[str]] = frozenset(
    {
        "refinement_sort_by_pos_before_sort_by_label",
        "gen_ts_shuffle_before_sort_by_label",
        "gen_t_sort_by_pos_before_sort_by_label",
        "completion_sort_by_pos_before_sort_by_label",
        "ugen_sort_by_pos_before_sort_by_label",
        "gen_r_discrete_before_induce_relations",
        "gen_r_sort_by_pos_before_sort_by_label",
        "add_sep_token",
        "sort_by_dict",
        "share_embedding",
    }
)
_RICO_RELATION: Final[frozenset[str]] = frozenset(
    (_RICO_COMMON - {"refinement_sort_by_pos_before_sort_by_label"})
    | {"refinement_shuffle_before_sort_by_label", "load_vocab"}
)
_PUB_BASE: Final[frozenset[str]] = frozenset(
    {
        "refinement_sort_by_pos_before_sort_by_label",
        "gen_ts_shuffle_before_sort_by_label",
        "gen_t_sort_by_pos_before_sort_by_label",
        "completion_sort_by_pos_before_sort_by_label",
        "ugen_sort_by_pos_before_sort_by_label",
        "gen_r_compact",
        "gen_r_add_unk_token",
        "gen_r_discrete_before_induce_relations",
        "gen_r_shuffle_before_sort_by_label",
        "add_sep_token",
        "sort_by_dict",
        "share_embedding",
        "load_vocab",
    }
)
_PUB_POSITION_FLAGS: Final[frozenset[str]] = frozenset(
    {
        "completion_sort_by_pos",
        "ugen_sort_by_pos",
    }
)


def _recipe(
    *,
    dataset: DatasetName,
    condition: ConditionType,
    tasks: tuple[LayoutFormerPPTask, ...],
    task_ids: tuple[int, ...],
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    max_position_embeddings: int,
    decode_max_length: int,
    warmup_num_steps: int,
    eval_seed: int,
    eval_interval: int,
    vocab_size: int,
    serialization_flags: frozenset[str],
    eval_tasks: tuple[LayoutFormerPPTask, ...] | None = None,
    partition_buckets: tuple[int, ...] = (),
) -> LayoutFormerPPTrainingRecipe:
    return LayoutFormerPPTrainingRecipe(
        name=f"{dataset}_{condition}",
        dataset=dataset,
        condition=condition,
        tasks=tasks,
        task_ids=task_ids,
        eval_tasks=eval_tasks or tasks,
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        max_position_embeddings=max_position_embeddings,
        decode_max_length=decode_max_length,
        warmup_num_steps=warmup_num_steps,
        eval_seed=eval_seed,
        eval_interval=eval_interval,
        vocab_size=vocab_size,
        serialization_flags=serialization_flags,
        partition_buckets=partition_buckets,
    )


_RECIPES: Final[tuple[LayoutFormerPPTrainingRecipe, ...]] = (
    _recipe(
        dataset=DatasetName.rico25,
        condition=ConditionType.label,
        tasks=(LayoutFormerPPTask.gen_t,),
        task_ids=(3,),
        epochs=100,
        batch_size=32,
        eval_batch_size=1,
        max_position_embeddings=150,
        decode_max_length=120,
        warmup_num_steps=1000,
        eval_seed=500,
        eval_interval=20,
        vocab_size=159,
        serialization_flags=_RICO_COMMON,
    ),
    _recipe(
        dataset=DatasetName.rico25,
        condition=ConditionType.label_size,
        tasks=(LayoutFormerPPTask.gen_ts,),
        task_ids=(4,),
        epochs=100,
        batch_size=32,
        eval_batch_size=1,
        max_position_embeddings=120,
        decode_max_length=120,
        warmup_num_steps=1000,
        eval_seed=500,
        eval_interval=20,
        vocab_size=159,
        serialization_flags=_RICO_COMMON,
    ),
    _recipe(
        dataset=DatasetName.rico25,
        condition=ConditionType.relation,
        tasks=(LayoutFormerPPTask.gen_r,),
        task_ids=(5,),
        epochs=150,
        batch_size=32,
        eval_batch_size=1,
        max_position_embeddings=400,
        decode_max_length=150,
        warmup_num_steps=1000,
        eval_seed=500,
        eval_interval=20,
        vocab_size=191,
        serialization_flags=_RICO_RELATION,
    ),
    _recipe(
        dataset=DatasetName.rico25,
        condition=ConditionType.refinement,
        tasks=(LayoutFormerPPTask.refinement,),
        task_ids=(0,),
        epochs=100,
        batch_size=32,
        eval_batch_size=100,
        max_position_embeddings=120,
        decode_max_length=120,
        warmup_num_steps=1000,
        eval_seed=100,
        eval_interval=20,
        vocab_size=159,
        serialization_flags=_RICO_COMMON,
    ),
    _recipe(
        dataset=DatasetName.rico25,
        condition=ConditionType.completion,
        tasks=(LayoutFormerPPTask.completion,),
        task_ids=(1,),
        epochs=100,
        batch_size=32,
        eval_batch_size=100,
        max_position_embeddings=120,
        decode_max_length=120,
        warmup_num_steps=1000,
        eval_seed=100,
        eval_interval=20,
        vocab_size=159,
        serialization_flags=_RICO_COMMON,
    ),
    _recipe(
        dataset=DatasetName.rico25,
        condition=ConditionType.unconditional,
        tasks=(LayoutFormerPPTask.ugen,),
        task_ids=(2,),
        epochs=100,
        batch_size=32,
        eval_batch_size=100,
        max_position_embeddings=350,
        decode_max_length=120,
        warmup_num_steps=1000,
        eval_seed=100,
        eval_interval=20,
        vocab_size=159,
        serialization_flags=_RICO_COMMON,
    ),
    _recipe(
        dataset=DatasetName.publaynet,
        condition=ConditionType.label,
        tasks=(LayoutFormerPPTask.gen_t,),
        task_ids=(3,),
        epochs=200,
        batch_size=64,
        eval_batch_size=1,
        max_position_embeddings=400,
        decode_max_length=150,
        warmup_num_steps=1000,
        eval_seed=500,
        eval_interval=50,
        vocab_size=139,
        serialization_flags=_PUB_BASE | _PUB_POSITION_FLAGS,
    ),
    _recipe(
        dataset=DatasetName.publaynet,
        condition=ConditionType.label_size,
        tasks=(LayoutFormerPPTask.gen_ts,),
        task_ids=(4,),
        epochs=200,
        batch_size=64,
        eval_batch_size=1,
        max_position_embeddings=400,
        decode_max_length=150,
        warmup_num_steps=4000,
        eval_seed=500,
        eval_interval=50,
        vocab_size=139,
        serialization_flags=_PUB_BASE | _PUB_POSITION_FLAGS | {"gen_t_add_unk_token"},
    ),
    _recipe(
        dataset=DatasetName.publaynet,
        condition=ConditionType.relation,
        tasks=(
            LayoutFormerPPTask.refinement,
            LayoutFormerPPTask.gen_ts,
            LayoutFormerPPTask.gen_t,
            LayoutFormerPPTask.completion,
            LayoutFormerPPTask.ugen,
            LayoutFormerPPTask.gen_r,
        ),
        task_ids=(0, 4, 3, 1, 2, 5),
        eval_tasks=(LayoutFormerPPTask.gen_r,),
        epochs=200,
        batch_size=64,
        eval_batch_size=1,
        max_position_embeddings=400,
        decode_max_length=150,
        warmup_num_steps=3000,
        eval_seed=500,
        eval_interval=50,
        vocab_size=178,
        serialization_flags=_PUB_BASE
        | _PUB_POSITION_FLAGS
        | {
            "gen_t_add_unk_token",
            "gen_ts_add_unk_token",
            "add_task_prompt",
            "partition_training_data",
        },
        partition_buckets=(-1, -1, -2, 0, 0, -3),
    ),
    _recipe(
        dataset=DatasetName.publaynet,
        condition=ConditionType.refinement,
        tasks=(LayoutFormerPPTask.refinement,),
        task_ids=(0,),
        epochs=200,
        batch_size=64,
        eval_batch_size=64,
        max_position_embeddings=400,
        decode_max_length=150,
        warmup_num_steps=2000,
        eval_seed=100,
        eval_interval=50,
        vocab_size=139,
        serialization_flags=_PUB_BASE
        | _PUB_POSITION_FLAGS
        | {"gen_t_add_unk_token", "gen_ts_add_unk_token"},
    ),
    _recipe(
        dataset=DatasetName.publaynet,
        condition=ConditionType.completion,
        tasks=(LayoutFormerPPTask.completion,),
        task_ids=(1,),
        epochs=200,
        batch_size=64,
        eval_batch_size=64,
        max_position_embeddings=400,
        decode_max_length=150,
        warmup_num_steps=3000,
        eval_seed=100,
        eval_interval=50,
        vocab_size=139,
        serialization_flags=_PUB_BASE,
    ),
    _recipe(
        dataset=DatasetName.publaynet,
        condition=ConditionType.unconditional,
        tasks=(LayoutFormerPPTask.ugen,),
        task_ids=(2,),
        epochs=200,
        batch_size=64,
        eval_batch_size=64,
        max_position_embeddings=400,
        decode_max_length=150,
        warmup_num_steps=1000,
        eval_seed=100,
        eval_interval=50,
        vocab_size=139,
        serialization_flags=_PUB_BASE,
    ),
)

TRAINING_RECIPES: Final[
    Mapping[tuple[DatasetName, ConditionType], LayoutFormerPPTrainingRecipe]
] = MappingProxyType(
    {(recipe.dataset, recipe.condition): recipe for recipe in _RECIPES}
)
TRAINING_RECIPES_BY_NAME: Final[Mapping[str, LayoutFormerPPTrainingRecipe]] = (
    MappingProxyType({recipe.name: recipe for recipe in _RECIPES})
)


def get_training_recipe(name: str) -> LayoutFormerPPTrainingRecipe:
    """Return one faithful recipe by its canonical config stem."""
    try:
        return TRAINING_RECIPES_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unknown LayoutFormer++ training recipe: {name}") from exc


__all__ = [
    "LayoutFormerPPTrainingRecipe",
    "TRAINING_RECIPES",
    "TRAINING_RECIPES_BY_NAME",
    "get_training_recipe",
]

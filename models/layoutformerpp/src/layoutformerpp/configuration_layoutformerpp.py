"""Configuration for LayoutFormer++."""

from __future__ import annotations

from typing import Final, TypedDict

from laygen.common.bbox import BoxFormat, normalize_box_format
from laygen.common import ConditionType, DatasetName
from transformers import PretrainedConfig

from .tasks import (
    LayoutFormerPPTask,
    TASK_TO_CONDITION,
    layoutformerpp_dataset_slug,
    normalize_layoutformerpp_dataset,
    normalize_layoutformerpp_task,
)


class TaskDefaults(TypedDict, total=False):
    """Generation defaults for one dataset/condition pair."""

    max_position_embeddings: int
    decode_max_length: int
    eval_seed: int


TASK_DEFAULTS: Final[dict[tuple[DatasetName, ConditionType], TaskDefaults]] = {
    (DatasetName.rico25, ConditionType.label): {
        "max_position_embeddings": 150,
        "decode_max_length": 120,
        "eval_seed": 500,
    },
    (DatasetName.rico25, ConditionType.label_size): {
        "max_position_embeddings": 120,
        "decode_max_length": 120,
        "eval_seed": 500,
    },
    (DatasetName.rico25, ConditionType.relation): {
        "max_position_embeddings": 400,
        "decode_max_length": 150,
        "eval_seed": 500,
    },
    (DatasetName.rico25, ConditionType.refinement): {
        "max_position_embeddings": 120,
        "decode_max_length": 120,
        "eval_seed": 100,
    },
    (DatasetName.rico25, ConditionType.completion): {
        "max_position_embeddings": 120,
        "decode_max_length": 120,
        "eval_seed": 100,
    },
    (DatasetName.rico25, ConditionType.unconditional): {
        "max_position_embeddings": 350,
        "decode_max_length": 120,
        "eval_seed": 100,
    },
    (DatasetName.publaynet, ConditionType.label): {
        "max_position_embeddings": 400,
        "decode_max_length": 150,
        "eval_seed": 500,
    },
    (DatasetName.publaynet, ConditionType.label_size): {
        "max_position_embeddings": 400,
        "decode_max_length": 150,
        "eval_seed": 500,
    },
    (DatasetName.publaynet, ConditionType.relation): {
        "max_position_embeddings": 400,
        "decode_max_length": 150,
        "eval_seed": 500,
    },
    (DatasetName.publaynet, ConditionType.refinement): {
        "max_position_embeddings": 400,
        "decode_max_length": 150,
        "eval_seed": 100,
    },
    (DatasetName.publaynet, ConditionType.completion): {
        "max_position_embeddings": 400,
        "decode_max_length": 150,
        "eval_seed": 100,
    },
    (DatasetName.publaynet, ConditionType.unconditional): {
        "max_position_embeddings": 400,
        "decode_max_length": 150,
        "eval_seed": 100,
    },
}


class LayoutFormerPPConfig(PretrainedConfig):
    """Stores model, tokenizer, and task defaults for converted checkpoints."""

    model_type = "layoutformerpp"

    def __init__(
        self,
        vocab_size: int = 0,
        max_position_embeddings: int | None = None,
        d_model: int = 512,
        encoder_layers: int = 8,
        decoder_layers: int | None = None,
        encoder_attention_heads: int = 8,
        decoder_attention_heads: int | None = None,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
        share_embedding: bool = True,
        dataset: DatasetName | str = DatasetName.rico25,
        task: LayoutFormerPPTask | ConditionType | str = LayoutFormerPPTask.gen_t,
        max_num_elements: int = 20,
        bbox_format: BoxFormat | str = BoxFormat.ltwh,
        default_box_format: BoxFormat | str = BoxFormat.xywh,
        discrete_x_grid: int = 128,
        discrete_y_grid: int = 128,
        add_sep_token: bool = True,
        sort_by_dict: bool = True,
        add_task_embedding: bool = False,
        add_task_prompt_token_in_model: bool = False,
        num_task_prompt_token: int = 1,
        task_id: int | None = None,
        decode_max_length: int | None = None,
        eval_seed: int | None = None,
        gen_t_add_unk_token: bool = False,
        gen_ts_add_unk_token: bool = False,
        gen_r_add_unk_token: bool = False,
        gen_r_compact: bool = False,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 2,
        is_encoder_decoder: bool = True,
        condition_type: ConditionType | str | None = None,
        model_type: str | None = None,
        transformers_version: str | None = None,
        architectures: list[str] | None = None,
        id2label: dict[int | str, str] | None = None,
        label2id: dict[str, int] | None = None,
        torch_dtype: str | None = None,
        dtype: str | None = None,
        tie_word_embeddings: bool = True,
        task_specific_params: dict[str, object] | None = None,
        name_or_path: str = "",
        _commit_hash: str | None = None,
        attn_implementation: str | None = None,
    ) -> None:
        """Initialize architecture and task-specific generation defaults."""
        _ = (condition_type, model_type, transformers_version)
        normalized_dataset = normalize_layoutformerpp_dataset(dataset)
        normalized_task = normalize_layoutformerpp_task(task)
        condition_type = TASK_TO_CONDITION[normalized_task]
        self.dataset = layoutformerpp_dataset_slug(normalized_dataset)
        self.task = str(normalized_task)
        self.condition_type = str(condition_type)
        defaults = TASK_DEFAULTS.get((normalized_dataset, condition_type), {})
        if max_position_embeddings is None:
            max_position_embeddings = int(defaults.get("max_position_embeddings", 150))
        if decode_max_length is None:
            decode_max_length = int(defaults.get("decode_max_length", 120))
        if eval_seed is None:
            eval_seed = int(defaults.get("eval_seed", 100))

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = (
            decoder_layers if decoder_layers is not None else encoder_layers
        )
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = (
            decoder_attention_heads
            if decoder_attention_heads is not None
            else encoder_attention_heads
        )
        self.dropout = dropout
        self.dim_feedforward = (
            dim_feedforward if dim_feedforward is not None else d_model * 4
        )
        self.share_embedding = share_embedding
        self.max_num_elements = max_num_elements
        self.bbox_format = str(normalize_box_format(bbox_format))
        self.default_box_format = str(normalize_box_format(default_box_format))
        self.discrete_x_grid = discrete_x_grid
        self.discrete_y_grid = discrete_y_grid
        self.add_sep_token = add_sep_token
        self.sort_by_dict = sort_by_dict
        self.add_task_embedding = add_task_embedding
        self.add_task_prompt_token_in_model = add_task_prompt_token_in_model
        self.num_task_prompt_token = num_task_prompt_token
        self.task_id = task_id
        self.decode_max_length = decode_max_length
        self.eval_seed = eval_seed
        self.gen_t_add_unk_token = gen_t_add_unk_token
        self.gen_ts_add_unk_token = gen_ts_add_unk_token
        self.gen_r_add_unk_token = gen_r_add_unk_token
        self.gen_r_compact = gen_r_compact
        pretrained_kwargs: dict[str, object] = {
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "is_encoder_decoder": is_encoder_decoder,
            "vocab_size": vocab_size,
            "tie_word_embeddings": tie_word_embeddings,
        }
        optional_pretrained_kwargs: dict[str, object | None] = {
            "architectures": architectures,
            "id2label": id2label,
            "label2id": label2id,
            "torch_dtype": torch_dtype or dtype,
            "task_specific_params": task_specific_params,
            "name_or_path": name_or_path,
            "_commit_hash": _commit_hash,
            "attn_implementation": attn_implementation,
        }
        pretrained_kwargs.update(
            {
                key: value
                for key, value in optional_pretrained_kwargs.items()
                if value is not None
            }
        )
        super_init = getattr(super(), "__init__")
        super_init(**pretrained_kwargs)

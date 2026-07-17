"""Configuration for LayoutFormer++."""

from __future__ import annotations

from transformers import PretrainedConfig


TASK_DEFAULTS: dict[tuple[str, str], dict[str, int | bool]] = {
    ("rico", "gen_t"): {
        "max_position_embeddings": 150,
        "decode_max_length": 120,
        "eval_seed": 500,
    },
    ("rico", "gen_ts"): {
        "max_position_embeddings": 120,
        "decode_max_length": 120,
        "eval_seed": 500,
    },
    ("rico", "gen_r"): {
        "max_position_embeddings": 400,
        "decode_max_length": 150,
        "eval_seed": 500,
    },
    ("rico", "refinement"): {
        "max_position_embeddings": 120,
        "decode_max_length": 120,
        "eval_seed": 100,
    },
    ("rico", "completion"): {
        "max_position_embeddings": 120,
        "decode_max_length": 120,
        "eval_seed": 100,
    },
    ("rico", "ugen"): {
        "max_position_embeddings": 350,
        "decode_max_length": 120,
        "eval_seed": 100,
    },
    ("publaynet", "gen_t"): {
        "max_position_embeddings": 400,
        "decode_max_length": 150,
        "eval_seed": 500,
    },
    ("publaynet", "gen_ts"): {
        "max_position_embeddings": 400,
        "decode_max_length": 150,
        "eval_seed": 500,
    },
    ("publaynet", "gen_r"): {
        "max_position_embeddings": 400,
        "decode_max_length": 150,
        "eval_seed": 500,
    },
    ("publaynet", "refinement"): {
        "max_position_embeddings": 400,
        "decode_max_length": 150,
        "eval_seed": 100,
    },
    ("publaynet", "completion"): {
        "max_position_embeddings": 400,
        "decode_max_length": 150,
        "eval_seed": 100,
    },
    ("publaynet", "ugen"): {
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
        dataset: str = "rico",
        task: str = "gen_t",
        max_num_elements: int = 20,
        bbox_format: str = "ltwh",
        default_box_format: str = "xywh",
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
        **kwargs: object,
    ) -> None:
        """Initialize architecture and task-specific generation defaults."""
        self.dataset = dataset
        self.task = task
        defaults = TASK_DEFAULTS.get((dataset, task), {})
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
        self.bbox_format = bbox_format
        self.default_box_format = default_box_format
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
        kwargs.pop("is_encoder_decoder", None)
        kwargs.pop("vocab_size", None)
        pretrained_kwargs: dict[str, object] = {
            "bos_token_id": kwargs.pop("bos_token_id", 0),
            "eos_token_id": kwargs.pop("eos_token_id", 1),
            "pad_token_id": kwargs.pop("pad_token_id", 2),
            "is_encoder_decoder": True,
            "vocab_size": vocab_size,
            **kwargs,
        }
        super().__init__(**pretrained_kwargs)  # type: ignore

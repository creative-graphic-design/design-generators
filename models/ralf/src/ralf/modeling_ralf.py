"""PyTorch model wrapper for RALF checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
import os
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from typing import cast
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_ralf import RalfConfig


class RalfForConditionalLayoutGeneration(PreTrainedModel):
    """Forward-compatible `PreTrainedModel` for RALF autoregressive decoding."""

    config_class = RalfConfig
    base_model_prefix = "ralf"
    main_input_name = "input_ids"
    _tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: RalfConfig) -> None:
        """Initialize a compact transformer decoder surface."""
        super().__init__(config)
        self._uses_vendor_modules = False
        if config.use_vendor_modules:  # pragma: no cover
            self._init_vendor_modules(config)
            self.all_tied_weights_keys = dict(self._tied_weights_keys)
            return
        self.embed_tokens = nn.Embedding(config.vocab_size, config.decoder_d_model)
        self.image_projection = nn.Linear(config.image_channels, config.decoder_d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_d_model,
            nhead=config.num_attention_heads,
            dim_feedforward=config.decoder_d_model * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=config.decoder_layers)
        self.lm_head = nn.Linear(config.decoder_d_model, config.vocab_size, bias=False)
        self.all_tied_weights_keys = dict(self._tied_weights_keys)
        self.post_init()

    def _init_vendor_modules(self, config: RalfConfig) -> None:  # pragma: no cover
        """Attach vendor RALF modules under original checkpoint key prefixes."""
        vendor_root = self._find_vendor_root()
        if vendor_root is None:
            raise FileNotFoundError(
                "vendor/ralf is required for use_vendor_modules=True"
            )
        if str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

        from datasets import ClassLabel, Dataset, Features, Sequence
        import image2layout.train.fid.model as fid_model
        import image2layout.train.helpers.layout_tokenizer as vendor_tokenizer
        import image2layout.train.models.common.image as vendor_image
        from image2layout.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
        from image2layout.train.models.retrieval_augmented_autoreg import (
            ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg,
        )

        cache_dir = self._find_vendor_cache_dir(config)
        precomputed_dir = str((cache_dir / "PRECOMPUTED_WEIGHT_DIR").resolve())
        fid_model.PRECOMPUTED_WEIGHT_DIR = precomputed_dir
        vendor_image.PRECOMPUTED_WEIGHT_DIR = precomputed_dir
        vendor_tokenizer.PRECOMPUTED_WEIGHT_DIR = precomputed_dir

        id2label = cast(Mapping[int | str, str], config.id2label)
        label_names = [
            label
            for _, label in sorted(id2label.items(), key=lambda item: int(item[0]))
        ]
        features = Features({"label": Sequence(ClassLabel(names=label_names))})
        tokenizer = LayoutSequenceTokenizer(
            label_feature=features["label"].feature,
            max_seq_length=config.max_seq_length,
            num_bin=config.num_bin,
            var_order=list(config.var_order),
            pad_until_max=False,
            special_tokens=list(config.special_tokens),
            is_loc_vocab_shared=config.is_loc_vocab_shared,
            geo_quantization=config.geo_quantization,
        )
        vendor_dataset_name = "pku" if config.dataset_name.startswith("pku") else "cgl"
        vendor_model = ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg(
            features=features,
            tokenizer=tokenizer,
            dataset_name=vendor_dataset_name,
            max_seq_length=config.max_seq_length,
            db_dataset=Dataset.from_dict({"id": []}),
            d_model=config.d_model,
            decoder_d_model=config.decoder_d_model,
            top_k=config.top_k,
            layout_backbone=config.layout_backbone,
            use_reference_image=config.use_reference_image,
            freeze_layout_encoder=config.freeze_layout_encoder,
            retrieval_backbone=config.retrieval_backbone,
            random_retrieval=False,
            saliency_k="None",
            auxilary_task=self._canonical_to_vendor_task(config.task),
            use_flag_embedding=config.use_flag_embedding,
            use_multitask=config.use_multitask,
            RELATION_SIZE=config.relation_size,
            global_task_embedding=config.global_task_embedding,
        )
        for name, module in vendor_model.named_children():
            setattr(self, name, module)
        for name, buffer in vendor_model.named_buffers(recurse=False):
            self.register_buffer(name, buffer.detach().clone())
        object.__setattr__(self, "_vendor_model", vendor_model)
        self._uses_vendor_modules = True

    @staticmethod
    def _canonical_to_vendor_task(task: str) -> str:
        return {
            "unconditional": "uncond",
            "label": "c",
            "label_size": "cwh",
            "completion": "partial",
        }.get(task, task)

    @staticmethod
    def _find_vendor_root() -> Path | None:  # pragma: no cover
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "vendor" / "ralf"
            if (candidate / "image2layout").is_dir():
                return candidate
        return None

    @staticmethod
    def _find_vendor_cache_dir(config: RalfConfig) -> Path:  # pragma: no cover
        candidates: list[Path] = []
        if config.vendor_cache_dir:
            candidates.append(Path(config.vendor_cache_dir))
        for parent in Path(__file__).resolve().parents:
            candidates.append(parent / ".cache" / "ralf" / "cache")
        for candidate in candidates:
            if (candidate / "PRECOMPUTED_WEIGHT_DIR").is_dir():
                return candidate
        raise FileNotFoundError(
            "RALF vendor cache with PRECOMPUTED_WEIGHT_DIR is required for "
            "use_vendor_modules=True"
        )

    def _image_context(
        self,
        pixel_values: Float[torch.Tensor, "batch channels height width"] | None,
        saliency: Float[torch.Tensor, "batch 1 height width"] | None = None,
    ) -> Float[torch.Tensor, "batch hidden"]:
        if pixel_values is None:
            return self.embed_tokens.weight.new_zeros((1, self.config.decoder_d_model))
        if saliency is None:
            saliency = pixel_values.new_zeros(
                pixel_values.size(0), 1, pixel_values.size(2), pixel_values.size(3)
            )
        merged = torch.cat([pixel_values, saliency], dim=1)
        pooled = merged.mean(dim=(-1, -2))
        return self.image_projection(pooled)

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"] | None = None,
        pixel_values: Float[torch.Tensor, "batch channels height width"] | None = None,
        saliency: Float[torch.Tensor, "batch 1 height width"] | None = None,
        attention_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        labels: Int[torch.Tensor, "batch tokens"] | None = None,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> CausalLMOutput | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """Run teacher-forced token prediction.

        Args:
            input_ids: Autoregressive token ids.
            pixel_values: Optional RGB images.
            saliency: Optional one-channel saliency maps.
            attention_mask: Optional valid-token mask.
            labels: Optional next-token labels.
            return_dict: Whether to return a ModelOutput.
            kwargs: Reserved converted-checkpoint inputs.

        Returns:
            CausalLMOutput or tuple with logits/loss.
        """
        vendor_inputs = kwargs.pop("vendor_inputs", None)
        if self._uses_vendor_modules:  # pragma: no cover
            if vendor_inputs is None:
                raise NotImplementedError(
                    "Vendor-backed RALF models require vendor_inputs for forward"
                )
            self._sync_vendor_model_references()
            return self._vendor_model(cast(dict[str, object], vendor_inputs))
        _ = kwargs
        if input_ids is None:
            raise ValueError("input_ids is required for non-vendor RALF models")
        embeddings = self.embed_tokens(input_ids)
        context = self._image_context(pixel_values, saliency)
        if context.size(0) == 1 and embeddings.size(0) != 1:
            context = context.expand(embeddings.size(0), -1)
        hidden = embeddings + context.unsqueeze(1)
        causal_mask = torch.triu(
            torch.full(
                (input_ids.size(1), input_ids.size(1)),
                -math.inf,
                device=input_ids.device,
            ),
            diagonal=1,
        )
        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        hidden = self.decoder(
            hidden, mask=causal_mask, src_key_padding_mask=padding_mask
        )
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            targets = labels.clone()
            targets[targets == self.config.pad_token_id] = -100
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        if return_dict is False:
            return (logits,) if loss is None else (loss, logits)
        return CausalLMOutput(loss=cast(torch.FloatTensor | None, loss), logits=logits)

    def _sync_vendor_model_references(self) -> None:  # pragma: no cover
        """Point the non-registered vendor object at registered child modules."""
        vendor_model = self._vendor_model
        for name, _module in vendor_model.named_children():
            setattr(vendor_model, name, getattr(self, name))
        for name, _buffer in vendor_model.named_buffers(recurse=False):
            setattr(vendor_model, name, getattr(self, name))

    @torch.no_grad()
    def _generate_sequences(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        pixel_values: Float[torch.Tensor, "batch channels height width"] | None = None,
        saliency: Float[torch.Tensor, "batch 1 height width"] | None = None,
        attention_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        *,
        max_length: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
        token_mask: Bool[torch.Tensor, "tokens vocab"] | None = None,
    ) -> Int[torch.Tensor, "batch tokens"]:
        """Run the autoregressive token loop used by `RalfPipeline`."""
        generated = input_ids[:, :1].clone()
        max_length = max_length or self.config.max_token_length
        for step in range(max_length):
            outputs = self(
                input_ids=generated,
                pixel_values=pixel_values,
                saliency=saliency,
                attention_mask=torch.ones_like(generated, dtype=torch.bool),
            )
            logits = outputs.logits[:, -1, :] / temperature
            if token_mask is not None and step < token_mask.size(0):
                logits = logits.masked_fill(
                    ~token_mask[step].to(logits.device), -math.inf
                )
            if top_k is not None and top_k > 0 and top_k < logits.size(-1):
                values = torch.topk(logits, top_k).values
                logits = logits.masked_fill(logits < values[:, [-1]], -math.inf)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

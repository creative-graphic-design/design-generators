"""Frozen LayoutDM reference used by Layout-Corrector training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Float, Int
from laygen.common.discrete import index_to_log_onehot
from laygen.common.randomness import multinomial, rand, randint

from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.conversion import split_original_state_dict
from layout_dm.modeling_layout_dm import (
    LayoutDMDenoiser,
    skip_layoutdm_initialization,
)
from layout_dm.scheduling_layout_dm import LayoutDMScheduler
from layout_dm.tokenization_layout_dm import LayoutDMTokenizer

FROZEN_LAYOUT_DM_SHA256: Final[dict[str, str]] = {
    "rico25": "7759bdf9e05cccef7a6a7e4260adc50f8c1ef6e6faa10351b79fb63f6b51c853",
    "publaynet": "9f7aee8ca600cc7cc96182affc85f96ebafc2b41a9ae72b05dfacfd64e89791d",
}


@dataclass
class FrozenLayoutDMReference:
    """Package-side frozen LayoutDM model, tokenizer, and schedules."""

    dataset_name: str
    checkpoint_path: Path
    config: LayoutDMConfig
    tokenizer: LayoutDMTokenizer
    model: LayoutDMDenoiser
    scheduler: LayoutDMScheduler
    lt_history: Float[torch.Tensor, "timesteps"]
    lt_count: Float[torch.Tensor, "timesteps"]
    checkpoint_sha256: str

    @classmethod
    def from_checkpoint(
        cls,
        *,
        dataset_name: str,
        checkpoint_path: str | Path,
        cluster_centers_path: str | Path,
        initialization_device: torch.device | str = "cpu",
    ) -> "FrozenLayoutDMReference":
        """Load the selected seed-0 checkpoint and preserve its state tensors."""
        path = Path(checkpoint_path)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict):
            raise TypeError("LayoutDM checkpoint must be a state dictionary")

        expected = FROZEN_LAYOUT_DM_SHA256[dataset_name]
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(
                f"Unexpected frozen LayoutDM SHA-256 for {dataset_name}: {observed}"
            )

        config = LayoutDMConfig(
            dataset_name=dataset_name,
            max_seq_length=25,
            num_bin_bboxes=32,
            bbox_quantization="kmeans",
            cluster_centers_path=str(cluster_centers_path),
            hidden_size=464,
            num_attention_heads=8,
            num_hidden_layers=4,
            intermediate_size=1856,
            dropout=0.0,
            timestep_type="adalayernorm",
            num_timesteps=100,
            q_type="constrained",
        )
        tokenizer = LayoutDMTokenizer(config)
        with skip_layoutdm_initialization():
            with torch.device("cpu"):
                model = LayoutDMDenoiser(
                    vocab_size=config.vocab_size,
                    max_token_length=config.max_token_length,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_hidden_layers=config.num_hidden_layers,
                    intermediate_size=config.intermediate_size,
                    dropout=config.dropout,
                    timestep_type=config.timestep_type,
                )

        target_device = torch.device(initialization_device)
        torch.nn.Module.to(model, target_device)
        model.initialize_weights()
        model.load_state_dict(split_original_state_dict(raw), strict=True)
        torch.nn.Module.to(model, "cpu")
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        model.eval()

        scheduler = LayoutDMScheduler(
            num_timesteps=config.num_timesteps,
            q_type="constrained",
            vocab_size=config.vocab_size,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
            var_order=tuple(config.var_order.split("-")),
            per_var_full_ids=tokenizer.full_id_maps(),
            att_1=config.att_1,
            att_T=config.att_T,
            ctt_1=config.ctt_1,
            ctt_T=config.ctt_T,
        )
        for key in tokenizer.var_names:
            scheduler.schedules[key] = tuple(
                raw[f"model.module.{key}_{suffix}"].clone()
                for suffix in (
                    "log_at",
                    "log_bt",
                    "log_ct",
                    "log_cumprod_at",
                    "log_cumprod_bt",
                    "log_cumprod_ct",
                )
            )

        return cls(
            dataset_name=dataset_name,
            checkpoint_path=path,
            config=config,
            tokenizer=tokenizer,
            model=model,
            scheduler=scheduler,
            lt_history=raw["model.module.Lt_history"].clone(),
            lt_count=raw["model.module.Lt_count"].clone(),
            checkpoint_sha256=observed,
        )

    def to(self, device: torch.device | str) -> "FrozenLayoutDMReference":
        """Move the reference model and its comparison buffers."""
        target_device = torch.device(device)
        torch.nn.Module.to(self.model, target_device)
        self.lt_history = self.lt_history.to(target_device)
        self.lt_count = self.lt_count.to(target_device)
        return self

    def sample_time(
        self, batch_size: int, *, device: torch.device
    ) -> tuple[Int[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
        """Use the released importance sampler and its frozen loss history."""
        if not bool((self.lt_count > 10).all()):
            t = randint(0, 100, size=(batch_size,), device=device).long()
            return t, torch.full_like(t, 1.0 / 100, dtype=torch.float32)

        history = torch.sqrt(self.lt_history + 1.0e-10) + 0.0001
        history[0] = history[1]
        probabilities = history / history.sum()
        t = multinomial(probabilities, batch_size, replacement=True, device=device)
        return t, probabilities.gather(dim=0, index=t)

    def corrupt(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        timesteps: Int[torch.Tensor, "batch"],
    ) -> tuple[
        Int[torch.Tensor, "batch tokens"], Float[torch.Tensor, "batch vocab tokens"]
    ]:
        """Apply constrained LayoutDM corruption with reference random sampling."""
        batch_size, token_length = input_ids.shape
        step = len(self.tokenizer.var_names)
        reshaped = input_ids.reshape(batch_size, token_length // step, step)
        full_logs: list[Float[torch.Tensor, "batch vocab tokens"]] = []
        full_ids: list[Int[torch.Tensor, "batch tokens"]] = []
        for index, key in enumerate(self.tokenizer.var_names):
            partial = self.tokenizer.full_to_partial_ids(reshaped[..., index], key)
            log_start = index_to_log_onehot(
                partial, len(self.tokenizer.full_id_maps()[key])
            )
            log_prob = self.scheduler._q_pred(log_start, timesteps, key)
            uniform = rand(*log_prob.shape, device=log_prob.device)
            gumbel = -torch.log(-torch.log(uniform + 1.0e-30) + 1.0e-30)
            sampled = (gumbel + log_prob).argmax(dim=1)
            log_sample = index_to_log_onehot(
                sampled, len(self.tokenizer.full_id_maps()[key])
            )
            full_logs.append(self.tokenizer.partial_to_full_log_probs(log_sample, key))
            full_ids.append(self.tokenizer.partial_to_full_ids(sampled, key))

        log_xt = torch.stack(full_logs, dim=-1).reshape(
            batch_size, self.config.vocab_size, token_length
        )
        xt = torch.stack(full_ids, dim=-1).reshape(batch_size, token_length)
        return xt, log_xt

    def prediction_log_probs(
        self,
        log_xt: Float[torch.Tensor, "batch vocab tokens"],
        timesteps: Int[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch vocab tokens"]:
        """Predict clean-token probabilities with the frozen denoiser."""
        logits = self.model(input_ids=log_xt.argmax(dim=1), timesteps=timesteps).logits
        return self.scheduler.predict_start(logits)

    def reconstruct_previous(
        self,
        log_x0: Float[torch.Tensor, "batch vocab tokens"],
        log_xt: Float[torch.Tensor, "batch vocab tokens"],
        timesteps: Int[torch.Tensor, "batch"],
    ) -> Int[torch.Tensor, "batch tokens"]:
        """Sample the reference x_t-1 reconstruction, excluding MASK."""
        log_previous = self.scheduler.q_posterior(log_x0, log_xt, timesteps)
        log_previous[:, self.config.mask_token_id, :] = -70.0
        probabilities = log_previous.softmax(dim=1).movedim(1, -1)
        sampled = multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]),
            1,
            replacement=False,
            device=log_previous.device,
        ).reshape(log_previous.shape[0], -1)
        return sampled

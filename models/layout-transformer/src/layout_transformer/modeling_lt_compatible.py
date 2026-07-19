"""Local LT-Net modules with original checkpoint-compatible state keys."""

# ruff: noqa: D102,D103,D107

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn

from .configuration_layout_transformer import LayoutTransformerConfig


def _cfg(config: LayoutTransformerConfig) -> dict[str, object]:
    return {
        "MODEL": {
            "PRETRAIN": False,
            "DECODER": {
                "HEAD_TYPE": config.decoder_head_type.upper(),
                "BOX_LOSS": config.decoder_box_loss.upper(),
                "SCHEDULE_SAMPLE": config.decoder_schedule_sample,
                "TWO_PATH": config.decoder_two_path,
                "GLOBAL_FEATURE": config.decoder_global_feature,
                "GREEDY": config.decoder_greedy,
                "XY_TEMP": config.xy_temperature,
                "WH_TEMP": config.wh_temperature,
            },
            "REFINE": {
                "REFINE": config.refine,
                "HEAD_TYPE": config.refine_head_type.title(),
                "BOX_LOSS": config.refine_box_loss.title(),
                "X_Softmax": config.refine_x_softmax,
            },
        }
    }


class MultiHeadedAttention(nn.Module):
    """Multi-head attention matching the original LT-Net implementation."""

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1) -> None:
        super().__init__()
        if size % num_heads != 0:
            raise ValueError("attention size must be divisible by num_heads")
        self.head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads
        self.k_layer = nn.Linear(size, size)
        self.v_layer = nn.Linear(size, size)
        self.q_layer = nn.Linear(size, size)
        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = k.size(0)
        num_heads = self.num_heads
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q / math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(2, 3))
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attention = self.dropout(self.softmax(scores))
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )
        return self.output_layer(context)


class ContMultiHeadedAttention(nn.Module):
    """Continuous-valued attention used by the original bbox decoder."""

    def __init__(
        self, num_heads: int, size: int, size_v: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if size % num_heads != 0:
            raise ValueError("attention size must be divisible by num_heads")
        self.head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads
        self.k_layer = nn.Linear(size, size)
        self.v_layer = nn.Linear(size_v, size)
        self.q_layer = nn.Linear(size, size)
        self.output_layer = nn.Linear(size, size_v)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = k.size(0)
        num_heads = self.num_heads
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q / math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(2, 3))
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attention = self.dropout(self.softmax(scores))
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )
        return self.output_layer(context)


class CustomAttention(nn.Module):
    """Refinement attention with optional PDF confidence reweighting."""

    def __init__(
        self, num_heads: int, size: int, dropout: float = 0.1, sent_length: int = 128
    ) -> None:
        super().__init__()
        if size % num_heads != 0:
            raise ValueError("attention size must be divisible by num_heads")
        self.head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads
        self.k_layer = nn.Linear(size // 4, num_heads * self.head_size // 4)
        self.v_layer = nn.Linear(size, num_heads * self.head_size)
        self.q_layer = nn.Linear(size // 4, num_heads * self.head_size // 4)
        self.confident_layer = nn.Sequential(
            nn.Linear(sent_length, sent_length), nn.ReLU()
        )
        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        mask: torch.Tensor | None = None,
        xy_pdf_score: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = k.size(0)
        num_heads = self.num_heads
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)
        k = k.view(batch_size, -1, num_heads, self.head_size // 4).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size // 4).transpose(1, 2)
        q = q / math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(2, 3))
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        if xy_pdf_score is None:
            attention = self.softmax(scores)
        else:
            xy_pdf_score = self.confident_layer(xy_pdf_score).view(batch_size, 1, 1, -1)
            scores_exp = scores.exp()
            new_scores = scores_exp * xy_pdf_score
            attention = new_scores / new_scores.sum(-1).unsqueeze(-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )
        return self.output_layer(context)


class GELU(nn.Module):
    """Original LT-Net GELU implementation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))
        )


class PositionwiseFeedForward(nn.Module):
    """Original pre-norm feed-forward block."""

    def __init__(self, input_size: int, ff_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pwff_layer(self.layer_norm(x)) + x


class TransformerEncoderLayer(nn.Module):
    """Original relation encoder layer."""

    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_norm = self.layer_norm(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        return self.feed_forward(self.dropout(h) + x)


class TransformerEncoder(nn.Module):
    """Original relation transformer encoder."""

    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self._output_size = hidden_size
        self._hidden_size = hidden_size

    def forward(self, embed_src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = embed_src
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)


class SentenceEmbeddings(nn.Module):
    """Original sentence/object/token-type embeddings."""

    def __init__(
        self,
        vocab_size: int = 204,
        obj_classes_size: int = 154,
        hidden_size: int = 512,
        max_rel_pair: int = 33,
        max_token_type: int = 4,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.obj_id_embeddings = nn.Embedding(
            obj_classes_size, hidden_size, padding_idx=0
        )
        self.sentence_type = nn.Embedding(max_rel_pair, hidden_size, padding_idx=0)
        self.token_type = nn.Embedding(max_token_type, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        input_token: torch.Tensor,
        input_obj_id: torch.Tensor,
        segment_label: torch.Tensor,
        token_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_embeds = self.word_embeddings(input_token)
        embeddings = (
            inputs_embeds
            + self.sentence_type(segment_label)
            + self.obj_id_embeddings(input_obj_id)
            + self.token_type(token_type)
        )
        return self.dropout(embeddings), inputs_embeds


class RelEncoder(nn.Module):
    """Original LT-Net relation encoder and token classifiers."""

    def __init__(self, config: LayoutTransformerConfig) -> None:
        super().__init__()
        self.input_embeddings = SentenceEmbeddings(
            config.vocab_size,
            config.obj_classes_size,
            config.hidden_size,
            max_rel_pair=33,
            hidden_dropout_prob=config.dropout,
        )
        self.encoder = TransformerEncoder(
            hidden_size=config.hidden_size,
            ff_size=config.hidden_size * 4,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            emb_dropout=config.dropout,
        )
        self.hidden_size = config.hidden_size
        self.vocab_classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.obj_id_classifier = nn.Linear(config.hidden_size, config.obj_classes_size)
        self.token_type_classifier = nn.Linear(config.hidden_size, 4)

    def forward(
        self,
        input_token: torch.Tensor,
        input_obj_id: torch.Tensor,
        segment_label: torch.Tensor,
        token_type: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        src, class_embeds = self.input_embeddings(
            input_token, input_obj_id, segment_label, token_type
        )
        encoder_output = self.encoder(src, src_mask)
        return (
            encoder_output,
            self.vocab_classifier(encoder_output),
            self.obj_id_classifier(encoder_output),
            self.token_type_classifier(encoder_output),
            src,
            class_embeds,
        )


class CustomTransformerDecoderLayer(nn.Module):
    """Original bbox decoder layer."""

    def __init__(
        self,
        size: int = 0,
        bb_size: int = 64,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.size = size
        self.trg_trg_att = ContMultiHeadedAttention(
            num_heads, bb_size, bb_size, dropout=dropout
        )
        self.src_trg_att = ContMultiHeadedAttention(
            num_heads, size, size, dropout=dropout
        )
        self.feed_forward_h1 = PositionwiseFeedForward(bb_size, ff_size=ff_size)
        self.feed_forward_h2 = PositionwiseFeedForward(size, ff_size=ff_size)
        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.spa_layer_norm = nn.LayerNorm(bb_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        spatial_x: torch.Tensor,
        semantic_x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        trg_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = src_mask
        spatial_x_norm = self.spa_layer_norm(spatial_x)
        self.x_layer_norm(semantic_x)
        h1 = self.trg_trg_att(
            spatial_x_norm, spatial_x_norm, spatial_x_norm, mask=trg_mask
        )
        h1 = self.dropout(h1) + spatial_x
        o1 = self.feed_forward_h1(h1)
        o2 = memory[:, 1:, :]
        return torch.cat((o2, o1), dim=-1)


class CustomTransformerDecoder(nn.Module):
    """Original custom bbox transformer decoder."""

    def __init__(
        self,
        hidden_size: int = 768,
        hidden_bb_size: int = 64,
        ff_size: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [
                CustomTransformerDecoderLayer(
                    size=hidden_size,
                    bb_size=hidden_bb_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_size + hidden_bb_size, eps=1e-6)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

    def forward(
        self,
        trg_embed_0: torch.Tensor,
        trg_embed_1: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_hidden: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
        unroll_steps: int | None = None,
        hidden: torch.Tensor | None = None,
        trg_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = (encoder_hidden, unroll_steps, hidden)
        if trg_mask is None:
            raise ValueError("trg_mask required for Transformer")
        trg_mask = trg_mask & self.subsequent_mask(trg_embed_0.size(1)).type_as(
            trg_mask
        ).to(trg_mask.device)
        x = torch.cat((trg_embed_1, trg_embed_0[:, : trg_embed_1.size(1)]), dim=-1)
        for layer in self.layers:
            x = layer(
                spatial_x=trg_embed_0,
                semantic_x=trg_embed_1,
                memory=encoder_output,
                src_mask=src_mask,
                trg_mask=trg_mask,
            )
        return self.layer_norm(x)

    @staticmethod
    def subsequent_mask(size: int) -> torch.Tensor:
        mask = torch.triu(torch.ones((1, size, size), dtype=torch.uint8), diagonal=1)
        return mask == 0


class DecoderLinearHead(nn.Module):
    """Original linear decoder box head."""

    def __init__(self, input_dim: int, box_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(input_dim, box_dim)
        self.activation = nn.Sigmoid()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        x = self.activation(self.dense(x))
        return x[:, :, 2:], x[:, :, :2], None, None, None


class LinearHead(nn.Module):
    """Original linear refinement box head."""

    def __init__(
        self, input_dim: int, box_dim: int, output_dim: int, box_emb_size: int
    ) -> None:
        super().__init__()
        self.box_emb_size = box_emb_size
        self.box_embedding = nn.Linear(box_dim, self.box_emb_size)
        self.dense = nn.Linear(input_dim + self.box_emb_size, self.box_emb_size)
        self.feed_forward = nn.Linear(self.box_emb_size, output_dim)
        self.activation = nn.Sigmoid()

    def forward(
        self, x: torch.Tensor, box: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        box_embed = self.box_embedding(box)
        x = self.dense(torch.cat((x, box_embed), dim=-1))
        x = self.activation(self.feed_forward(x + box_embed))
        return x[:, :, 2:], x[:, :, :2], None, None, None


class GMMHead(nn.Module):
    """Original GMM box head with optional generator plumbing."""

    def __init__(
        self,
        hidden_size: int,
        *,
        condition: bool = False,
        x_softmax: bool = False,
        greedy: bool = False,
        config: LayoutTransformerConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.aug_size = max(1, hidden_size // 4)
        self.gmm_comp_num = 5
        self.gmm_param_num = 6
        self.xy_bivariate = nn.Linear(
            self.hidden_size, self.gmm_comp_num * self.gmm_param_num
        )
        self.condition = condition
        self.X_Sfotmax = x_softmax
        self.greedy = greedy
        self.xy_temperature = config.xy_temperature
        self.wh_temperature = config.wh_temperature
        if condition:
            self.xy_embedding = nn.Linear(2, self.aug_size)
            self.dropout = nn.Dropout(0.1)
            self.wh_bivariate = nn.Linear(
                self.hidden_size + self.aug_size,
                self.gmm_comp_num * self.gmm_param_num,
            )
        self.is_training = False

    def forward(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        batch_size = x.size(0)
        xy_gmm = self.xy_bivariate(x)
        pi_xy, u_x, u_y, sigma_x, sigma_y, rho_xy = self.get_gmm_params(xy_gmm)
        sample_xy = self.sample_box(
            pi_xy,
            u_x,
            u_y,
            sigma_x,
            sigma_y,
            rho_xy,
            temp=self.xy_temperature,
            greedy=self.greedy,
            device=x.device,
            generator=generator,
        ).reshape(batch_size, -1, 2)
        sample_x = sample_xy[:, :, 0].unsqueeze(2).repeat(1, 1, self.gmm_comp_num)
        sample_y = sample_xy[:, :, 1].unsqueeze(2).repeat(1, 1, self.gmm_comp_num)
        xy_pdf = (
            self.batch_pdf(
                pi_xy,
                sample_x,
                sample_y,
                u_x,
                u_y,
                sigma_x,
                sigma_y,
                rho_xy,
                batch_size,
                self.gmm_comp_num,
                x.device,
            )
            if self.X_Sfotmax
            else None
        )
        if not self.condition:
            return sample_xy, None, xy_gmm, None, None
        xy_embed = self.dropout(self.xy_embedding(sample_xy))
        wh_gmm = self.wh_bivariate(torch.cat((x, xy_embed), dim=-1))
        pi_wh, u_w, u_h, sigma_w, sigma_h, rho_wh = self.get_gmm_params(wh_gmm)
        sample_wh = self.sample_box(
            pi_wh,
            u_w,
            u_h,
            sigma_w,
            sigma_h,
            rho_wh,
            temp=self.wh_temperature,
            greedy=self.greedy,
            device=x.device,
            generator=generator,
        ).reshape(batch_size, -1, 2)
        return sample_wh, sample_xy, wh_gmm, xy_gmm, xy_pdf

    def get_gmm_params(
        self, gmm_params: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        pi, u_x, u_y, sigma_x, sigma_y, rho_xy = torch.split(
            gmm_params, self.gmm_comp_num, dim=2
        )
        pi = nn.Softmax(dim=-1)(pi).reshape(-1, self.gmm_comp_num).detach().cpu()
        u_x = u_x.reshape(-1, self.gmm_comp_num).detach().cpu()
        u_y = u_y.reshape(-1, self.gmm_comp_num).detach().cpu()
        sigma_x = torch.exp(sigma_x).reshape(-1, self.gmm_comp_num).detach().cpu()
        sigma_y = torch.exp(sigma_y).reshape(-1, self.gmm_comp_num).detach().cpu()
        rho_xy = (
            torch.tanh(rho_xy)
            .clamp(min=-0.95, max=0.95)
            .reshape(-1, self.gmm_comp_num)
            .detach()
            .cpu()
        )
        return pi, u_x, u_y, sigma_x, sigma_y, rho_xy

    def sample_box(
        self,
        pi: torch.Tensor,
        u_x: torch.Tensor,
        u_y: torch.Tensor,
        sigma_x: torch.Tensor,
        sigma_y: torch.Tensor,
        rho_xy: torch.Tensor,
        *,
        temp: float | None,
        greedy: bool,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if temp is not None:
            pi = self.adjust_temp(pi, temp)
        try:
            sample_pi = pi
            generator_device = None if generator is None else generator.device
            if generator_device is not None:
                sample_pi = pi.to(generator_device)
            pi_idx = torch.multinomial(sample_pi, 1, generator=generator).cpu()
        except RuntimeError:
            pi_idx = torch.multinomial(pi, 1)
        except Exception:
            pi_idx = pi.argmax(1).unsqueeze(-1)
        u_x = torch.gather(u_x, dim=1, index=pi_idx)
        u_y = torch.gather(u_y, dim=1, index=pi_idx)
        sigma_x = torch.gather(sigma_x, dim=1, index=pi_idx)
        sigma_y = torch.gather(sigma_y, dim=1, index=pi_idx)
        rho_xy = torch.gather(rho_xy, dim=1, index=pi_idx)
        return self.sample_bivariate_normal(
            u_x,
            u_y,
            sigma_x,
            sigma_y,
            rho_xy,
            temp,
            greedy=greedy,
            device=device,
            generator=generator,
        )

    @staticmethod
    def adjust_temp(pi_pdf: torch.Tensor, temperature: float) -> torch.Tensor:
        pi_pdf = torch.log(pi_pdf) / temperature
        pi_pdf -= torch.max(pi_pdf)
        pi_pdf = torch.exp(pi_pdf)
        pi_pdf /= torch.sum(pi_pdf)
        return pi_pdf

    @staticmethod
    def sample_bivariate_normal(
        u_x: torch.Tensor,
        u_y: torch.Tensor,
        sigma_x: torch.Tensor,
        sigma_y: torch.Tensor,
        rho_xy: torch.Tensor,
        temperature: float | None,
        *,
        greedy: bool,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if greedy:
            return torch.cat((u_x, u_y), dim=-1).to(device)
        mean = torch.cat((u_x, u_y), dim=1)
        scale = math.sqrt(1.0 if temperature is None else temperature)
        sigma_x *= scale
        sigma_y *= scale
        cov = torch.zeros((u_x.size(0), 2, 2))
        cov[:, 0, 0] = sigma_x.flatten() * sigma_x.flatten()
        cov[:, 0, 1] = rho_xy.flatten() * sigma_x.flatten() * sigma_y.flatten()
        cov[:, 1, 0] = rho_xy.flatten() * sigma_x.flatten() * sigma_y.flatten()
        cov[:, 1, 1] = sigma_y.flatten() * sigma_y.flatten()
        det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
        for idx in (det == 0).nonzero():
            cov[idx] *= 0.0
            cov[idx, 0, 0] += 1.0
            cov[idx, 1, 1] += 1.0
        dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
        _ = generator
        sample = dist.sample()
        return sample.to(device)

    @staticmethod
    def batch_pdf(
        pi_xy: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        u_x: torch.Tensor,
        u_y: torch.Tensor,
        sigma_x: torch.Tensor,
        sigma_y: torch.Tensor,
        rho_xy: torch.Tensor,
        batch_size: int,
        gmm_comp_num: int,
        device: torch.device,
    ) -> torch.Tensor:
        u_x = u_x.reshape(batch_size, -1, gmm_comp_num).to(device)
        u_y = u_y.reshape(batch_size, -1, gmm_comp_num).to(device)
        sigma_x = sigma_x.reshape(batch_size, -1, gmm_comp_num).to(device)
        sigma_y = sigma_y.reshape(batch_size, -1, gmm_comp_num).to(device)
        pi_xy = pi_xy.reshape(batch_size, -1, gmm_comp_num).to(device)
        rho_xy = rho_xy.reshape(batch_size, -1, gmm_comp_num).to(device)
        z_x = ((x - u_x) / sigma_x) ** 2
        z_y = ((y - u_y) / sigma_y) ** 2
        z_xy = (x - u_x) * (y - u_y) / (sigma_x * sigma_y)
        z = z_x + z_y - 2 * rho_xy * z_xy
        exp = torch.exp(-z / (2 * (1 - rho_xy**2)))
        norm = torch.clamp(
            2 * math.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy**2),
            min=1e-5,
        )
        return torch.sum(pi_xy * exp / norm, dim=2).detach()


class TransformerRefineLayer(nn.Module):
    """Original refinement transformer layer."""

    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
        sent_length: int = 128,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.box_norm = nn.LayerNorm(size // 4, eps=1e-6)
        self.src_src_att = CustomAttention(
            num_heads, size, dropout=dropout, sent_length=sent_length
        )
        self.combine_layer = nn.Linear(size + size // 4, size)
        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(
        self,
        context: torch.Tensor,
        box: torch.Tensor,
        mask: torch.Tensor,
        xy_pdf_score: torch.Tensor | None,
    ) -> torch.Tensor:
        context_norm = self.layer_norm(context)
        box_norm = self.box_norm(box)
        h = self.src_src_att(box_norm, context_norm, box_norm, mask, xy_pdf_score)
        return self.feed_forward(self.dropout(h) + context_norm)


class RefineEncoder(nn.Module):
    """Original refinement encoder."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        box_dim: int,
        sent_length: int = 128,
    ) -> None:
        super().__init__()
        self.aug_size = max(1, hidden_size // 4)
        self.box_embedding = nn.Linear(box_dim, self.aug_size)
        self.layer = TransformerRefineLayer(
            size=hidden_size,
            ff_size=hidden_size * 4,
            num_heads=num_heads,
            dropout=dropout,
            sent_length=sent_length,
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.emb_dropout = nn.Dropout(p=dropout)
        self.box_dim = box_dim
        self._output_size = hidden_size
        self._hidden_size = hidden_size
        self.blank_box = torch.Tensor([2.0, 2.0, 2.0, 2.0])

    def forward(
        self,
        context: torch.Tensor,
        input_box: torch.Tensor,
        mask: torch.Tensor,
        xy_pdf_score: torch.Tensor | None,
    ) -> torch.Tensor:
        box = input_box.clone()
        box[:, :, : self.box_dim][~mask.squeeze(1)] = self.blank_box[: self.box_dim].to(
            box.device
        )
        box_embed = self.emb_dropout(self.box_embedding(box[:, :, : self.box_dim]))
        return self.layer_norm(self.layer(context, box_embed, mask, xy_pdf_score))


class PDFDecoder(nn.Module):
    """Original LT-Net PDF decoder."""

    def __init__(
        self,
        *,
        box_dim: int = 4,
        hidden_size: int = 256,
        num_layers: int = 2,
        attn_heads: int = 2,
        dropout: float = 0.1,
        config: LayoutTransformerConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.schedule_sample = config.decoder_schedule_sample
        self.global_feature = config.decoder_global_feature
        self.aug_size = max(1, hidden_size // 4)
        self.box_embedding = nn.Linear(box_dim, self.aug_size)
        output_input = 2 * hidden_size + self.aug_size
        if not self.global_feature:
            output_input = hidden_size + self.aug_size
        self.output_Layer = nn.Linear(output_input, hidden_size)
        self.latent_transformer = nn.Linear(hidden_size, hidden_size - self.aug_size)
        self.decoder = CustomTransformerDecoder(
            hidden_size=hidden_size,
            hidden_bb_size=self.aug_size,
            ff_size=hidden_size * 4,
            num_layers=num_layers,
            num_heads=attn_heads,
            dropout=dropout,
            emb_dropout=dropout,
        )
        if config.decoder_head_type.upper() == "GMM":
            self.box_predictor = GMMHead(
                hidden_size,
                condition=True,
                x_softmax=config.refine_x_softmax,
                greedy=config.decoder_greedy,
                config=config,
            )
        else:
            self.box_predictor = DecoderLinearHead(hidden_size, 4)

    def random_sample(
        self, output_box: torch.Tensor, pred_box: torch.Tensor, sample_num: int
    ) -> torch.Tensor:
        length = torch.arange(output_box.size(1))
        index = torch.Tensor(random.sample(list(enumerate(length)), sample_num))[
            :, 0
        ].long()
        mask = torch.zeros(
            output_box.size(), dtype=torch.bool, device=output_box.device
        )
        mask[:, index] = 1
        output_box[mask] = pred_box[mask]
        return output_box

    def forward(
        self,
        output_box: torch.Tensor,
        output_context: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        trg_mask: torch.Tensor,
        src: torch.Tensor,
        class_embeds: torch.Tensor,
        epoch: int = 0,
        is_train: bool = True,
        global_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        _ = (src, class_embeds)
        output_box_c = output_box.clone()
        if global_mask is None:
            global_mask = torch.ones(
                encoder_output.shape[:2], dtype=torch.bool, device=encoder_output.device
            )
        if is_train:
            global_feature = encoder_output.clone()
            global_feature[~global_mask] = float("-inf")
            global_feature = torch.max(global_feature, dim=1).values
            global_feature = global_feature.unsqueeze(1).repeat(
                1, encoder_output.size(1), 1
            )
            pair_count = min(
                output_box_c[:, 2::2, :].size(1), output_box_c[:, 1::2, :].size(1)
            )
            output_box_c[:, 2 : 2 + 2 * pair_count : 2, :] = output_box_c[
                :, 1 : 1 + 2 * pair_count : 2, :
            ]
        else:
            global_feature = output_context.clone()
            global_feature[~global_mask[:, : output_context.size(1)]] = float("-inf")
            global_feature = torch.max(global_feature, dim=1).values
            global_feature = global_feature.unsqueeze(1).repeat(
                1, encoder_output.size(1), 1
            )
            if (output_box_c.size(1) - 1) % 2 == 0 and output_box_c.size(1) > 1:
                pair_count = min(
                    output_box_c[:, 2::2, :].size(1),
                    output_box_c[:, 1::2, :].size(1),
                )
                output_box_c[:, 2 : 2 + 2 * pair_count : 2, :] = output_box_c[
                    :, 1 : 1 + 2 * pair_count : 2, :
                ]
            elif (output_box_c.size(1) - 1) % 2 != 0 and output_box_c.size(1) > 1:
                output_box_c[:, 2::2, :] = output_box_c[:, 1:-1:2, :]
        output_box_embed = self.box_embedding(output_box_c)
        decoder_output = self.decoder(
            trg_embed_0=output_box_embed,
            trg_embed_1=encoder_output[:, 1:, :],
            encoder_output=encoder_output,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=output_box_embed.size(1),
            hidden=None,
            trg_mask=trg_mask,
        )
        trg_input = torch.cat((encoder_output[:, :-1, :], output_box_embed), dim=-1)
        decoder_output = torch.cat(
            (trg_input[:, 0, :].unsqueeze(1), decoder_output), dim=1
        )
        if self.global_feature:
            decoder_output = torch.cat((decoder_output, global_feature), dim=-1)
        box_predictor_input = self.output_Layer(decoder_output)
        sample_wh, sample_xy, wh_gmm, xy_gmm, xy_pdf = self.box_predictor(
            box_predictor_input, generator=generator
        )
        if is_train and self.schedule_sample:
            pred_box = torch.cat((sample_xy, sample_wh), dim=-1)
            pred_box = pred_box[:, :-1]
            pred_box[:, 2::2, :] = pred_box[:, 1::2, :]
            sample_pred_num = int(
                pred_box.size(1) * (1.0 - ((epoch + 1) / 50.0) ** 1.2)
            )
            if sample_pred_num >= pred_box.size(1) / 3.0:
                sample_pred_num = int(pred_box.size(1) / 3.0)
            mix_output_box = self.random_sample(output_box_c, pred_box, sample_pred_num)
            mix_output_box_embed = self.box_embedding(mix_output_box)
            decoder_output = self.decoder(
                trg_embed_0=mix_output_box_embed,
                trg_embed_1=encoder_output[:, 1:, :],
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=mix_output_box_embed.size(1),
                hidden=None,
                trg_mask=trg_mask,
            )
            trg_input = torch.cat(
                (encoder_output[:, :-1, :], mix_output_box_embed), dim=-1
            )
            decoder_output = torch.cat(
                (trg_input[:, 0, :].unsqueeze(1), decoder_output), dim=1
            )
            if self.global_feature:
                decoder_output = torch.cat((decoder_output, global_feature), dim=-1)
            box_predictor_input = self.output_Layer(decoder_output)
            sample_wh, sample_xy, wh_gmm, xy_gmm, xy_pdf = self.box_predictor(
                box_predictor_input, generator=generator
            )
        return box_predictor_input, sample_wh, sample_xy, wh_gmm, xy_gmm, xy_pdf


class BBoxHead(nn.Module):
    """Original LT-Net bbox head."""

    def __init__(self, config: LayoutTransformerConfig) -> None:
        super().__init__()
        self.pad_index = 0
        self.bos_index = 1
        self.eos_index = 2
        self.box_dim = 4
        self.cfg = _cfg(config)
        self.Decoder = PDFDecoder(
            hidden_size=config.hidden_size,
            num_layers=2,
            attn_heads=2,
            dropout=config.dropout,
            config=config,
        )
        self.refine_module = config.refine
        if config.refine:
            self.refine_encoder = RefineEncoder(
                hidden_size=config.hidden_size,
                num_heads=1,
                dropout=config.dropout,
                box_dim=self.box_dim,
                sent_length=max(1, config.max_sequence_length // 2),
            )
            if config.refine_head_type.title() == "Linear":
                self.refine_box_head = LinearHead(
                    config.hidden_size,
                    self.box_dim,
                    4,
                    max(1, config.hidden_size // 4),
                )
            elif config.refine_head_type.upper() == "GMM":
                self.refine_box_head = GMMHead(
                    config.hidden_size,
                    condition=True,
                    x_softmax=False,
                    greedy=False,
                    config=config,
                )

    def forward(
        self,
        epoch: int,
        encoder_output: torch.Tensor,
        mask: torch.Tensor,
        src: torch.Tensor,
        class_embeds: torch.Tensor,
        output_box: torch.Tensor,
        trg_mask: torch.Tensor,
        global_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        (
            decoder_output,
            coarse_wh,
            coarse_xy,
            coarse_wh_gmm,
            coarse_xy_gmm,
            xy_pdf_score,
        ) = self.Decoder(
            output_box,
            encoder_output[:, :-1, :],
            encoder_output,
            mask,
            trg_mask,
            src,
            class_embeds,
            epoch,
            global_mask=global_mask,
            generator=generator,
        )
        coarse_box = torch.cat((coarse_xy, coarse_wh), dim=-1)
        coarse_gmm = (
            torch.cat((coarse_xy_gmm, coarse_wh_gmm), dim=-1)
            if coarse_xy_gmm is not None and coarse_wh_gmm is not None
            else None
        )
        if not self.refine_module:
            return coarse_box, coarse_gmm, None, None
        if xy_pdf_score is not None:
            refine_context = self.refine_encoder(
                decoder_output[:, 1::2],
                coarse_box[:, 1::2],
                mask[:, :, 1::2],
                xy_pdf_score.detach()[:, 1::2],
            )
        else:
            refine_context = self.refine_encoder(
                decoder_output[:, 1::2], coarse_box[:, 1::2], mask[:, :, 1::2], None
            )
        refine_wh, refine_xy, refine_wh_gmm, refine_xy_gmm, _ = self.refine_box_head(
            refine_context, coarse_box[:, 1::2]
        )
        refine_box = torch.cat((refine_xy, refine_wh), dim=-1)
        all_refine_box = torch.zeros(coarse_box.size(), device=coarse_box.device)
        all_refine_box[:, 1::2] += refine_box
        refine_gmm = (
            torch.cat((refine_xy_gmm, refine_wh_gmm), dim=-1)
            if refine_xy_gmm is not None and refine_wh_gmm is not None
            else None
        )
        return coarse_box, coarse_gmm, all_refine_box, refine_gmm

    def inference(
        self,
        encoder_output: torch.Tensor,
        mask: torch.Tensor,
        src: torch.Tensor,
        class_embeds: torch.Tensor,
        global_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        (
            decoder_output,
            coarse_wh,
            coarse_xy,
            coarse_wh_gmm,
            coarse_xy_gmm,
            xy_pdf_score,
        ) = greedy_pdf(
            src_mask=mask,
            bos_index=self.bos_index,
            eos_index=self.eos_index,
            max_output_length=128,
            decoder=self.Decoder,
            encoder_output=encoder_output,
            encoder_hidden=None,
            class_embeds=class_embeds,
            src=src,
            global_mask=global_mask,
            generator=generator,
        )
        coarse_box = torch.cat((coarse_xy, coarse_wh), dim=-1)
        coarse_gmm = (
            torch.cat((coarse_xy_gmm, coarse_wh_gmm), dim=-1)
            if coarse_xy_gmm is not None and coarse_wh_gmm is not None
            else None
        )
        if not self.refine_module:
            return coarse_box, coarse_gmm, None, None
        if xy_pdf_score is not None:
            refine_context = self.refine_encoder(
                decoder_output[:, 1::2],
                coarse_box[:, 1::2],
                mask[:, :, 1::2],
                xy_pdf_score.detach()[:, 1::2],
            )
        else:
            refine_context = self.refine_encoder(
                decoder_output[:, 1::2], coarse_box[:, 1::2], mask[:, :, 1::2], None
            )
        refine_wh, refine_xy, refine_wh_gmm, refine_xy_gmm, _ = self.refine_box_head(
            refine_context, coarse_box[:, 1::2]
        )
        refine_box = torch.cat((refine_xy, refine_wh), dim=-1)
        all_refine_box = torch.zeros(coarse_box.size(), device=coarse_box.device)
        all_refine_box[:, 1::2] += refine_box
        refine_gmm = (
            torch.cat((refine_xy_gmm, refine_wh_gmm), dim=-1)
            if refine_xy_gmm is not None and refine_wh_gmm is not None
            else None
        )
        return coarse_box, coarse_gmm, all_refine_box, refine_gmm


def greedy_pdf(
    *,
    src_mask: torch.Tensor,
    bos_index: int,
    eos_index: int,
    max_output_length: int,
    decoder: PDFDecoder,
    encoder_output: torch.Tensor,
    encoder_hidden: torch.Tensor | None,
    class_embeds: torch.Tensor,
    src: torch.Tensor,
    global_mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    _ = (bos_index, eos_index, encoder_hidden)
    gmm_comp_num = 5
    batch_size = src_mask.size(0)
    ys_0 = encoder_output.new_full([batch_size, 1, 4], 2.0, dtype=torch.float)
    ys_1 = encoder_output
    ys_wh_gmm: torch.Tensor | None = encoder_output.new_full(
        [batch_size, 1, gmm_comp_num * 6], 0.0, dtype=torch.float
    )
    ys_xy_gmm: torch.Tensor | None = encoder_output.new_full(
        [batch_size, 1, gmm_comp_num * 6], 0.0, dtype=torch.float
    )
    ys_xy_pdf: torch.Tensor | None = encoder_output.new_full(
        [batch_size, 1], 0.0001, dtype=torch.float
    )
    trg_mask = src_mask.new_ones([1, 1, 1])
    decoder_output = encoder_output
    for step in range(min(max_output_length - 1, encoder_output.size(1) - 1)):
        with torch.no_grad():
            decoder_output, sample_wh, sample_xy, wh_gmm, xy_gmm, xy_pdf = decoder(
                ys_0,
                ys_1,
                encoder_output[:, : step + 2, :],
                src_mask,
                trg_mask,
                src[:, : step + 2, :],
                class_embeds[:, : step + 2, :],
                is_train=False,
                global_mask=global_mask,
                generator=generator,
            )
            box_output = torch.cat((sample_xy, sample_wh), dim=-1)
            next_ys_0 = box_output[:, -1].unsqueeze(1).data
            ys_0 = torch.cat([ys_0, next_ys_0], dim=1)
            if wh_gmm is not None and xy_gmm is not None:
                if ys_wh_gmm is None or ys_xy_gmm is None:
                    raise ValueError("GMM history tensors must be initialized")
                ys_wh_gmm = torch.cat([ys_wh_gmm, wh_gmm[:, -1].unsqueeze(1)], dim=1)
                ys_xy_gmm = torch.cat([ys_xy_gmm, xy_gmm[:, -1].unsqueeze(1)], dim=1)
                if xy_pdf is not None and ys_xy_pdf is not None:
                    ys_xy_pdf = torch.cat(
                        [ys_xy_pdf, xy_pdf[:, -1].unsqueeze(1)], dim=1
                    )
                else:
                    ys_xy_pdf = None
            else:
                ys_wh_gmm = None
                ys_xy_gmm = None
                ys_xy_pdf = None
    return (
        decoder_output,
        ys_0[:, :, 2:],
        ys_0[:, :, :2],
        ys_wh_gmm,
        ys_xy_gmm,
        ys_xy_pdf,
    )

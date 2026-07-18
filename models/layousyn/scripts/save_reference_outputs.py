"""Generate LayouSyn vendor reference tensors for parity tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _public_xywh_from_vendor_xyxy(sample: torch.Tensor) -> torch.Tensor:
    sample = ((sample.clamp(-1, 1) + 1.0) / 2.0).float()
    left, top, right, bottom = sample.unbind(dim=-1)
    fixed = torch.stack(
        (
            torch.minimum(left, right),
            torch.minimum(top, bottom),
            torch.maximum(left, right),
            torch.maximum(top, bottom),
        ),
        dim=-1,
    )
    x1, y1, x2, y2 = fixed.unbind(dim=-1)
    return torch.stack(
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1),
        dim=-1,
    ).clamp(0, 1)


def save_reference_outputs(
    *,
    vendor_root: str | Path,
    ckpt: str | Path,
    ckpt_config: str | Path,
    output_dir: str | Path,
    seed: int = 0,
    caption: str,
    concepts: list[str],
    cfg_scale: float = 2.0,
    aspect_ratio: float = 1.0,
    num_sampling_steps: str = "40",
    sampling_type: str = "ddim",
) -> None:
    """Run the original implementation and save tensor references.

    The output directory is intentionally outside git. It contains real
    T5/SentenceTransformer embeddings, first-step denoiser/scheduler references,
    full-sample output, and metadata needed to regenerate the artifacts.
    """
    sys.path.insert(0, str(Path(vendor_root).resolve()))

    from sentence_transformers import SentenceTransformer

    from layousyn.config import Config
    from layousyn.diffusion import create_diffusion
    from layousyn.diffusion import gaussian_diffusion as gd
    from layousyn.evaluation.common import encode_captions, encode_labels, load_model
    from layousyn.model.t5_google import T5EmbedderGoogle

    if sampling_type != "ddim":
        raise ValueError("LayouSyn parity references currently use DDIM sampling")
    if len(concepts) == 0:
        raise ValueError("At least one --concept is required")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config.from_json(str(ckpt_config))
    model = load_model(str(ckpt), config, device=device)
    diffusion = create_diffusion(
        num_sampling_steps,
        alpha_scale=config.scale,
        noise_schedule=config.noise_schedule,
        diffusion_steps=config.diffusion_steps,
    )
    label_encoder = SentenceTransformer(
        f"sentence-transformers/sentence-t5-{config.t5_size}",
        device=device,
    )
    caption_encoder = T5EmbedderGoogle(
        dir_or_name=f"t5-v1_1-{config.t5_size}",
        device=device,
        model_max_length=config.max_y_len,
        is_torch_compile=False,
    )

    torch.manual_seed(seed)
    captions = [caption]
    labels_set = [concepts]
    batch_size = len(captions)
    initial = torch.randn(
        batch_size, config.max_in_len, config.in_channel, device=device
    )
    concept_embeds, concept_padding_mask = encode_labels(
        labels_set, label_encoder, config, device
    )
    caption_embeds, caption_padding_mask = encode_captions(
        captions, caption_encoder, batch_size=1
    )
    caption_embeds = caption_embeds.to(device)
    caption_padding_mask = caption_padding_mask.to(device)
    aspect = torch.tensor([aspect_ratio] * batch_size, device=device).float()

    initial_cfg = torch.cat([initial, initial], 0)
    concept_embeds_cfg = torch.cat([concept_embeds, concept_embeds], 0)
    concept_padding_mask_cfg = torch.cat(
        [concept_padding_mask, concept_padding_mask], 0
    )
    y_null = model.y_embedder.y_embedding.to(device).repeat(batch_size, 1, 1)
    y_mask_null = model.y_embedder.y_padding_mask.to(device).repeat(batch_size, 1)
    caption_embeds_cfg = torch.cat([caption_embeds, y_null], 0)
    caption_padding_mask_cfg = torch.cat([caption_padding_mask, y_mask_null], 0)
    aspect_cfg = torch.cat([aspect, aspect], 0)
    model_kwargs = {
        "ar": aspect_cfg,
        "x_enc": concept_embeds_cfg,
        "y": caption_embeds_cfg,
        "y_padding_mask": caption_padding_mask_cfg,
        "cfg_scale": cfg_scale,
    }

    scheduler_timestep = torch.full(
        (initial_cfg.shape[0],), diffusion.num_timesteps - 1, device=device
    )
    model_timestep = torch.full(
        (initial_cfg.shape[0],),
        diffusion.timestep_map[diffusion.num_timesteps - 1],
        device=device,
    )
    denoiser_logits = model.forward_with_cfg(
        initial_cfg,
        concept_padding_mask_cfg,
        model_timestep,
        **model_kwargs,
    )
    first_step = diffusion.ddim_sample(
        model.forward_with_cfg,
        initial_cfg,
        concept_padding_mask_cfg,
        scheduler_timestep,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        eta=0.0,
    )
    full_cfg = diffusion.ddim_sample_loop(
        model.forward_with_cfg,
        initial_cfg.shape,
        noise=initial_cfg,
        x_padding_mask=concept_padding_mask_cfg,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=False,
        return_samples=False,
        device=device,
        eta=0.0,
    )
    full = full_cfg[:batch_size].clamp(-1.0, 1.0)
    original_betas = gd.get_named_beta_schedule(
        config.noise_schedule,
        config.diffusion_steps,
        alpha_scale=config.scale,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "ckpt": str(ckpt),
        "ckpt_config": str(ckpt_config),
        "vendor_root": str(vendor_root),
        "seed": seed,
        "caption": caption,
        "concepts": concepts,
        "cfg_scale": cfg_scale,
        "aspect_ratio": aspect_ratio,
        "num_sampling_steps": int(num_sampling_steps),
        "sampling_type": sampling_type,
        "diffusion_steps": config.diffusion_steps,
        "noise_schedule": config.noise_schedule,
        "alpha_scale": config.scale,
        "layout_type": config.layout_type,
        "caption_model_name": f"google/t5-v1_1-{config.t5_size}",
        "concept_model_name": f"sentence-transformers/sentence-t5-{config.t5_size}",
        "device": device,
        "torch_initial_seed": torch.initial_seed(),
    }
    (out / "inputs.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    torch.save(
        {
            "initial_sample": initial_cfg.detach().cpu(),
            "concept_padding_mask": concept_padding_mask_cfg.detach().cpu(),
            "concept_embeds": concept_embeds_cfg.detach().cpu(),
            "caption_embeds": caption_embeds_cfg.detach().cpu(),
            "caption_padding_mask": caption_padding_mask_cfg.detach().cpu(),
            "aspect_ratio": aspect_cfg.detach().cpu(),
            "model_timestep": model_timestep.detach().cpu(),
            "scheduler_timestep": scheduler_timestep.detach().cpu(),
            "pipeline_concept_embeds": concept_embeds.detach().cpu(),
            "pipeline_caption_embeds": caption_embeds.detach().cpu(),
            "pipeline_caption_padding_mask": caption_padding_mask.detach().cpu(),
            "pipeline_aspect_ratio": aspect.detach().cpu(),
        },
        out / "inputs.pt",
    )
    torch.save(
        {
            "original_betas": torch.from_numpy(original_betas),
            "respaced_betas": torch.from_numpy(diffusion.betas),
            "timestep_map": torch.tensor(diffusion.timestep_map, dtype=torch.long),
            "denoiser_logits": denoiser_logits.detach().cpu(),
            "first_prev_sample": first_step["sample"].detach().cpu(),
            "first_pred_xstart": first_step["pred_xstart"].detach().cpu(),
            "full_sample": full.detach().cpu(),
            "public_bbox": _public_xywh_from_vendor_xyxy(full.detach().cpu()),
        },
        out / "reference.pt",
    )


def parse_args() -> argparse.Namespace:
    """Parse reference-generation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor-root", type=Path, default=Path("vendor/lay-your-scene")
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--ckpt-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--concept", action="append", dest="concepts", default=[])
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--aspect-ratio", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", default="40")
    parser.add_argument("--sampling-type", default="ddim")
    return parser.parse_args()


def main() -> None:
    """Generate reference tensors."""
    args = parse_args()
    save_reference_outputs(
        vendor_root=args.vendor_root,
        ckpt=args.ckpt,
        ckpt_config=args.ckpt_config,
        output_dir=args.output_dir,
        seed=args.seed,
        caption=args.caption,
        concepts=args.concepts,
        cfg_scale=args.cfg_scale,
        aspect_ratio=args.aspect_ratio,
        num_sampling_steps=args.num_sampling_steps,
        sampling_type=args.sampling_type,
    )


if __name__ == "__main__":
    main()

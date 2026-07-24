# Reproducing LayoutVAE

Workflow order: prepare local assets, generate original-module reference goldens, run parity, convert checkpoints, and run a `from_pretrained` smoke test.

## Setup

```bash
uv sync --package layoutvae --extra vendor
```

## Download

The LayoutVAE files are supplied by the repository submodule. Initialize it before running the conversion workflow. If this worktree has an empty submodule checkout, point `--source-root` and `LAYOUTVAE_SOURCE_ROOT` at a read-only initialized checkout such as `/root/ghq/github.com/creative-graphic-design/design-generators/vendor/layout-generation-baselines/LayoutVAE`.

```bash
git submodule update --init vendor/layout-generation-baselines
```

## Reference

Generate reference goldens under `.cache/layoutvae/reference`. The script loads the released `countvae.h5` and `bboxvae.h5` modules through the original source shim, fixes count latents, count Poisson samples, box latents, and box output noise, then writes the original-module forward outputs to `fixed_forward.pt`.

```bash
uv run --package layoutvae python models/layoutvae/scripts/save_reference_outputs.py \
  --source-root vendor/layout-generation-baselines/LayoutVAE \
  --output-dir .cache/layoutvae/reference
```

## Parity

Run the gated parity test with required assets enabled.

```bash
LAYOUTVAE_REFERENCE_DIR=.cache/layoutvae/reference \
LAYOUTVAE_CONVERTED_DIR=.cache/layoutvae/converted/layoutvae-publaynet \
LAYOUTVAE_SOURCE_ROOT=vendor/layout-generation-baselines/LayoutVAE \
PARITY_REQUIRE=1 \
uv run --package layoutvae pytest models/layoutvae/tests/vendor_parity -m vendor_parity
```

The parity suite verifies state-dict conversion for 74 tensors and fixed-latent forward agreement against the original modules. The observed fixed-latent maximum absolute difference is `0.0` for count outputs and `5.364418e-7` for box outputs, below the `1e-6` tolerance. The larger seeded raw-box drift observed on the stochastic end-to-end path is therefore attributed to RNG consumption order across the PyTorch distribution calls, not to the converted weights or deterministic tensor path.

## Convert

Convert the released checkpoint files into standard `config.json`, `model.safetensors`, `preprocessor_config.json`, and `README.md` artifacts.

```bash
uv run --package layoutvae python models/layoutvae/scripts/convert_original_checkpoint.py \
  --output-dir .cache/layoutvae/converted/layoutvae-publaynet
```

## from_pretrained Smoke

```bash
uv run --package layoutvae python -c 'from layoutvae import LayoutVAEPipeline; pipe = LayoutVAEPipeline.from_pretrained(".cache/layoutvae/converted/layoutvae-publaynet"); out = pipe(labels=[["text", "figure"]], seed=0); print(out.bbox.shape, out.labels.shape, out.mask.shape)'
```

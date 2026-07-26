# LayoutDM Training

This guide covers package-local LightningCLI training configs, trained-checkpoint conversion, and the staged S0-S5 reproduction protocol for LayoutDM. It follows the repository [training reproduction protocol](docs/training-reproduction.md).

Run commands from the repository root. Training data, logs, generated checkpoints, converted local pipelines, and evaluation artifacts stay under `.cache/layout-dm`.

## Install

```bash
uv sync --package layout-dm --extra training
```

Install the `vendor` extra only when rerunning staged parity against `vendor/layout-dm`.

```bash
uv sync --package layout-dm --extra training --extra vendor
```

### GPU Environment

The currently verified GPU setup is Tesla V100 with driver `575.57.08`, which
is compatible with CUDA 12.9-era wheels. The repository default torch build is
`cu130` (CUDA 13.0), and GPU initialization on that verified V100 setup fails
with `driver too old (found 12090)`. Keep the repository torch pins and lockfile
at their defaults, but before launching GPU training in this environment,
install a driver-compatible CUDA 12 torch wheel such as `cu126` into the
execution environment. This is a local runtime change only; do not commit
lockfile or `pyproject.toml` changes for the temporary torch replacement.

## Data

The training datamodule reads approved Hugging Face datasets directly:

| Dataset | Source | Config |
| --- | --- | --- |
| RICO25 | `creative-graphic-design/Rico` | `ui-screenshots-and-hierarchies-with-semantic-annotations` |
| PubLayNet | `creative-graphic-design/PubLayNet` | default |

The `smoke.yaml` config uses a synthetic local dataset and does not download RICO25 or PubLayNet.

## Configs

Training configs live under `models/layout-dm/configs/training`.

| Config | Dataset | Seed mode | Purpose |
| --- | --- | --- | --- |
| `layoutdm_rico25.yaml` | RICO25 | `default` | Regular LightningCLI training. |
| `layoutdm_publaynet.yaml` | PubLayNet | `default` | Regular LightningCLI training. |
| `layoutdm_rico25_deterministic.yaml` | RICO25 | `deterministic` | Deterministic short run for parity/debug checks. |
| `layoutdm_publaynet_deterministic.yaml` | PubLayNet | `deterministic` | Deterministic short run for parity/debug checks. |
| `smoke.yaml` | PubLayNet synthetic | `deterministic` | CPU smoke config for CLI wiring. |

## Validation Stages

| Stage | Scope | Purpose |
| --- | --- | --- |
| S0 | Static config and initialized state parity | Confirms package-local topology, parameter state, optimizer groups, scheduler defaults, and dataset encoding. |
| S1 | Fixed-batch pre-optimizer trace parity | Confirms timestep sampling, `q_sample`, denoiser output, posterior KL, auxiliary loss, and total loss before optimizer mutation. |
| S2 | One optimizer-step parity | Confirms gradients, clipped gradients, optimizer state, post-step params, and learning rate. |
| S3 | Short deterministic multi-batch run | Confirms loader, scheduler, clipping, and checkpoint wiring. |
| S4 | Deterministic loader stream | Confirms sample order, transforms, masks, padding, dataset-local class ids, and validation stream. |
| S5 | Full-run statistical comparison | Compares full RICO25 and PubLayNet learning behavior against the original-code checkpoints under the original evaluation protocol. |

## Reproduction Results

LayoutDM package-local training has S0-S4 scaffolding in place and S5 full-run statistical comparison is PENDING. RICO25 and PubLayNet training-seed n=3 results must be filled after package and original-code runs finish under the vendor evaluation protocol.

| Dataset | System | Stat scope | FID | Alignment | Overlap | mIoU | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| RICO25 | vendor | training-seed n=3 | TBD | TBD | TBD | TBD | S5 pending |
| RICO25 | ours | training-seed n=3 | TBD | TBD | TBD | TBD | S5 pending |
| PubLayNet | vendor | training-seed n=3 | TBD | TBD | TBD | TBD | S5 pending |
| PubLayNet | ours | training-seed n=3 | TBD | TBD | TBD | TBD | S5 pending |

Evidence locations:

```text
.cache/layout-dm/training-runs/
.cache/layout-dm/full-run/
.cache/layout-dm/converted/
```

## Reproducing These Results

Run the local CI training checks.

```bash
uv run --package layout-dm --extra training --with pytest pytest \
  models/layout-dm/tests -m "not vendor_parity and not integration" -q
CUDA_VISIBLE_DEVICES="" uv run --package layout-dm --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-dm/configs/training/smoke.yaml
```

Run the staged vendor parity checks after the submodule and local assets are available.

```bash
git submodule update --init vendor/layout-dm
CUDA_VISIBLE_DEVICES=<gpu-index> PARITY_REQUIRE=1 \
  uv run --package layout-dm --extra training --extra vendor --with pytest pytest \
  models/layout-dm/tests/vendor_parity/test_layout_dm_training_parity.py \
  -m "vendor_parity and training" -rs
```

Start a regular RICO25 package-local training run.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layout-dm --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-dm/configs/training/layoutdm_rico25.yaml \
  --trainer.accelerator=gpu --trainer.devices=1 \
  --trainer.default_root_dir=.cache/layout-dm/training-runs/rico25
```

Start a regular PubLayNet package-local training run.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layout-dm --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-dm/configs/training/layoutdm_publaynet.yaml \
  --trainer.accelerator=gpu --trainer.devices=1 \
  --trainer.default_root_dir=.cache/layout-dm/training-runs/publaynet
```

Convert a trained package checkpoint to a local Diffusers pipeline directory.

```bash
uv run --package layout-dm --extra convert \
  python models/layout-dm/scripts/convert_original_checkpoint.py \
  --dataset rico25 \
  --starter-dir .cache/layout-dm/starter/layoutdm_starter \
  --output-dir .cache/layout-dm/converted/layoutdm-rico25
```

Smoke-test local loading.

```bash
uv run --package layout-dm python models/layout-dm/scripts/smoke_from_pretrained.py \
  --model-dir .cache/layout-dm/converted/layoutdm-rico25
```

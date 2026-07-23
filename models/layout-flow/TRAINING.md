# LayoutFlow Training

This guide covers package-local LightningCLI training configs and the S0-S2 training-parity rerun for LayoutFlow.

Run commands from the repository root. Training data and generated checkpoints stay under `.cache/layout-flow`.

## Install

```bash
uv sync --package layout-flow --extra training
```

Install the `vendor` extra only when rerunning S0-S2 parity against `vendor/layout-flow`.

```bash
uv sync --package layout-flow --extra training --extra vendor
```

## Configs

Training configs live under `models/layout-flow/configs/training`.

| Config | Dataset | Seed mode | Purpose |
| --- | --- | --- | --- |
| `layoutflow_rico25.yaml` | RICO25 | `default` | Regular LightningCLI training. |
| `layoutflow_publaynet.yaml` | PubLayNet | `default` | Regular LightningCLI training. |
| `layoutflow_rico25_deterministic.yaml` | RICO25 | `deterministic` | Deterministic short run for parity/debug checks. |
| `layoutflow_publaynet_deterministic.yaml` | PubLayNet | `deterministic` | Deterministic short run for parity/debug checks. |
| `smoke.yaml` | PubLayNet | `deterministic` | CPU smoke config for CLI wiring. |

`default` preserves the LayoutFlow training seed policy used by regular runs. `deterministic` applies the `traingen-parity` determinism controls for fixed-batch trace and optimizer-step checks.

## Launch Training

Start a regular RICO25 training run.

```bash
uv run --package layout-flow --extra training \
  python -m layout_flow.training.cli fit \
  --config models/layout-flow/configs/training/layoutflow_rico25.yaml
```

Start a deterministic short run on one selected GPU. Set `CUBLAS_WORKSPACE_CONFIG` before the process starts.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --package layout-flow --extra training \
  python -m layout_flow.training.cli fit \
  --config models/layout-flow/configs/training/layoutflow_rico25_deterministic.yaml \
  --trainer.limit_train_batches=2
```

Run the CPU smoke config when checking CLI wiring without training data coverage claims.

```bash
uv run --package layout-flow --extra training \
  python -m layout_flow.training.cli fit \
  --config models/layout-flow/configs/training/smoke.yaml
```

## S0-S2 Parity Rerun

S0-S2 parity instantiates the vendor training module from `vendor/layout-flow`, copies its synthetic initialized state into the package-local LightningModule, and compares static state, fixed-batch pre-optimizer traces, and one optimizer step through `traingen-parity`.

```bash
git submodule update --init vendor/layout-flow
```

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> PARITY_REQUIRE=1 \
  uv run --package layout-flow --extra training --extra vendor pytest \
  models/layout-flow/tests/vendor_parity/test_layout_flow_training_parity.py \
  -m "vendor_parity and training" -rs
```

## Convert A Trained Checkpoint

After training, convert a Lightning checkpoint to a local `diffusers` pipeline directory. The converter accepts both original `model.*` keys and package-local training `model.backbone.*` keys.

```bash
uv run --package layout-flow python models/layout-flow/scripts/convert_original_checkpoint.py \
  --dataset rico25 \
  --checkpoint .cache/layout-flow/training-runs/rico25/checkpoints/<ckpt>.ckpt \
  --output-dir .cache/layout-flow/converted-trained/rico25
```

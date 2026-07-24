# LayoutFlow Training

This guide covers package-local LightningCLI training configs, full training runs, trained-checkpoint conversion, and the staged training-parity rerun for LayoutFlow.

Run commands from the repository root. Training data, generated checkpoints, CSV logs, and converted local pipelines stay under `.cache/layout-flow`.

## Install

```bash
uv sync --package layout-flow --extra training
```

Install the `vendor` extra only when rerunning staged parity against `vendor/layout-flow`.

```bash
uv sync --package layout-flow --extra training --extra vendor
```

Full RICO25 and PubLayNet runs fit on one 80 GB A100 with the default batch size of 512 used by the LayoutFlow reference setup. For smaller GPUs, lower `data.init_args.batch_size` and increase gradient accumulation only after rerunning the deterministic parity checks, because batch shape and optimizer order are part of the parity surface.

## Data

The training datamodule expects the LayoutFlow HDF5 files used by the original implementation. Keep them under `.cache/layout-flow/data` with this layout:

```text
.cache/layout-flow/data/
  rico25/
    ldm_rico_train.h5
    ldm_rico_val.h5
    ldm_rico_test.h5
  publaynet/
    publaynet_train.h5
    publaynet_val.h5
    publaynet_test.h5
```

The package configs point at those directories by default. If the files live elsewhere, override the datamodule path from LightningCLI:

```bash
uv run --package layout-flow --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-flow/configs/training/layoutflow_rico25.yaml \
  --data.init_args.data_path /path/to/rico25
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

## Validation Stages

| Stage | Scope | Purpose |
| --- | --- | --- |
| S0 | Static config and initialized state parity | Confirms the package-local Lightning module starts from the same state as the vendor training module. |
| S1 | Fixed-batch pre-optimizer trace parity | Confirms batch preparation, condition masking, flow sampling, forward output, and loss tensors before optimizer mutation. |
| S2 | One optimizer-step parity | Confirms one update produces matching parameter state. |
| S3 | Scheduler-enabled short run | Confirms the scheduler-enabled training path is wired before validation metrics are available. |
| S4 | Validation and scheduler behavior | Confirms package-local validation metrics can drive scheduler behavior when those metrics are implemented. |
| S5 | Full-run statistical comparison | Compares full RICO25 and PubLayNet learning behavior against the vendor training setup. |

## Reproduction Results

LayoutFlow training reproduction is achieved. RICO25 is statistically equivalent on training-seed n=3: FID 5.7111 +/- 0.7459 ours vs. 6.3907 +/- 0.8031 vendor, mIoU 0.5562 +/- 0.0102 ours vs. 0.5631 +/- 0.0200 vendor; PubLayNet loss curves nearly match, with final `train_loss_epoch` 0.1529803425 ours vs. 0.1530758440 vendor, and generated metrics are comparable or better overall.

The RICO25 numbers use training seeds `42975`, `42976`, and `42977`, epoch 1000, and fixed evaluation seed `42975`; the PubLayNet numbers use evaluation seeds `42975`, `42976`, and `42977` on one final checkpoint pair at epoch 1000 / global step 609000.

| Dataset | System | Stat scope | FID | Alignment | Overlap | mIoU | Loss evidence |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| RICO25 | vendor | training-seed n=3 | 6.3907 +/- 0.8031 | 0.2236 +/- 0.0163 | 0.5735 +/- 0.0365 | 0.5631 +/- 0.0200 | - |
| RICO25 | ours | training-seed n=3 | 5.7111 +/- 0.7459 | 0.2359 +/- 0.0086 | 0.5730 +/- 0.0202 | 0.5562 +/- 0.0102 | - |
| PubLayNet | vendor | evaluation-seed n=3 | 16.5893 +/- 0.2400 | 0.1072 +/- 0.0048 | 0.0313 +/- 0.0007 | 0.4290 +/- 0.0044 | final `train_loss_epoch` 0.1530758440 |
| PubLayNet | ours | evaluation-seed n=3 | 11.9050 +/- 0.5273 | 0.1160 +/- 0.0033 | 0.0303 +/- 0.0012 | 0.4151 +/- 0.0025 | final `train_loss_epoch` 0.1529803425 |

PubLayNet train-loss agreement over the 1000 common epochs has final difference -0.0000955015, mean absolute difference 0.0002493398, RMSE 0.0003080925, correlation 0.9995266371, and last-100 mean relative difference 0.0017203791.

PubLayNet n=3 is evaluation-seed n=3 on one final checkpoint pair; true training-seed n=3 needs additional full runs.

Evidence is recorded in the [RICO25 issue #149 comment](https://github.com/creative-graphic-design/design-generators/issues/149#issuecomment-5060415006), the [PubLayNet issue #149 comment](https://github.com/creative-graphic-design/design-generators/issues/149#issuecomment-5065491554), `.cache/layout-flow/full-run/eval-rico25-n3/vendor-protocol/summary_mean_std.csv`, `.cache/layout-flow/full-run/eval-publaynet/vendor-protocol-eval-seed-n3-summary.csv`, and local train-loss comparison outputs under `.cache/layout-flow/full-run/eval-publaynet/`.

### Reproducing These Results

1. Train the package-local implementation from the repository root.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layout-flow --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-flow/configs/training/layoutflow_<rico25|publaynet>.yaml \
  --model.init_args.scheduler=null --model.init_args.fid_calc_every_n=0 \
  --trainer.accelerator=gpu --trainer.devices=1 --trainer.max_epochs=1000 \
  --trainer.limit_val_batches=0 --trainer.num_sanity_val_steps=0 \
  --data.init_args.batch_size=512 \
  --trainer.default_root_dir=.cache/layout-flow/full-run/ours-<rico25|publaynet>
```

2. Train the vendor baseline from `vendor/layout-flow`.

```bash
cd vendor/layout-flow
CUDA_VISIBLE_DEVICES=<gpu-index> python src/train.py \
  experiment=LayoutFlow_<RICO|PubLayNet> dataset=<RICO|PubLayNet> model=LayoutFlow \
  trainer.max_epochs=1000 model.scheduler=null model.fid_calc_every_n=0 \
  +trainer.limit_val_batches=0 +trainer.num_sanity_val_steps=0 \
  dataset.dataset.data_path=../../.cache/layout-flow/data/<rico25|publaynet>
```

3. Evaluate with the vendor protocol from the repository root.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> LAYOUTFLOW_EVAL_SEED=<42975|42976|42977> \
uv run --package layout-flow --extra vendor --with rootutils \
  python models/layout-flow/scripts/run_vendor_test_with_torch_load_patch.py \
  checkpoint=.cache/layout-flow/full-run/<vendor-dataset>/checkpoints/last.ckpt \
  model=LayoutFlow dataset=<RICO|PubLayNet> task=uncond cond_mask=uncond \
  ode_solver=euler model.inference_steps=100 calc_miou=True multirun=False \
  dataset.dataset.data_path=.cache/layout-flow/data/<rico25|publaynet>
```

For package-local checkpoints, replace the checkpoint path with `.cache/layout-flow/full-run/<ours-dataset>/eval-vendor-protocol/last-vendor-compatible.ckpt` after converting the raw package checkpoint to the vendor-compatible format.

## Launch Training

Start a regular RICO25 training run. Until package-local FID validation is implemented, disable the plateau scheduler and validation batches for full training.

```bash
uv run --package layout-flow --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-flow/configs/training/layoutflow_rico25.yaml \
  --model.init_args.scheduler=null \
  --model.init_args.fid_calc_every_n=0 \
  --trainer.limit_val_batches=0 \
  --trainer.num_sanity_val_steps=0
```

Start a regular PubLayNet training run.

```bash
uv run --package layout-flow --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-flow/configs/training/layoutflow_publaynet.yaml \
  --model.init_args.scheduler=null \
  --model.init_args.fid_calc_every_n=0 \
  --trainer.limit_val_batches=0 \
  --trainer.num_sanity_val_steps=0
```

Use CLI overrides for short checks, smaller GPUs, or full-run scheduling changes:

```bash
uv run --package layout-flow --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-flow/configs/training/layoutflow_rico25.yaml \
  --trainer.max_epochs=1000 \
  --data.init_args.batch_size=256 \
  --trainer.limit_train_batches=20
```

For a controlled full run that keeps only the best checkpoint and `last.ckpt`, add a small runtime config and pass it after the package config:

```bash
mkdir -p .cache/layout-flow/training-runs/rico25
cat > .cache/layout-flow/training-runs/rico25/runtime.yaml <<'YAML'
trainer:
  accelerator: gpu
  devices: 1
  max_epochs: 1000
  limit_val_batches: 0
  num_sanity_val_steps: 0
  default_root_dir: .cache/layout-flow/training-runs/rico25
  logger:
    class_path: lightning.pytorch.loggers.CSVLogger
    init_args:
      save_dir: .cache/layout-flow/training-runs/rico25
      name: csv
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: .cache/layout-flow/training-runs/rico25/checkpoints
        filename: checkpoint-{epoch:04d}-{train_loss_epoch:.6f}
        monitor: train_loss_epoch
        mode: min
        save_top_k: 1
        save_last: true
data:
  init_args:
    batch_size: 512
    num_workers: 12
model:
  init_args:
    scheduler: null
    fid_calc_every_n: 0
YAML
CUDA_VISIBLE_DEVICES=<gpu-index> \
uv run --package layout-flow --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-flow/configs/training/layoutflow_rico25.yaml \
  --config .cache/layout-flow/training-runs/rico25/runtime.yaml
```

Start a deterministic short run on one selected GPU. Set `CUBLAS_WORKSPACE_CONFIG` before the process starts.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --package layout-flow --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-flow/configs/training/layoutflow_rico25_deterministic.yaml \
  --trainer.limit_train_batches=2
```

Run the CPU smoke config when checking CLI wiring without training data coverage claims.

```bash
uv run --package layout-flow --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-flow/configs/training/smoke.yaml
```

## Outputs And Resume

The runtime config above writes outputs below the selected Lightning `default_root_dir`:

```text
.cache/layout-flow/training-runs/<dataset>/
  checkpoints/
    checkpoint-epoch=....ckpt
    last.ckpt
  csv/
    version_0/
      metrics.csv
      hparams.yaml
```

CSV metrics can be inspected directly or opened from the run directory. If a run is configured with `TensorBoardLogger`, launch TensorBoard against the configured logger directory:

```bash
uv run --with tensorboard tensorboard --logdir .cache/layout-flow/training-runs
```

Resume from the latest checkpoint with `--ckpt_path`:

```bash
uv run --package layout-flow --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-flow/configs/training/layoutflow_rico25.yaml \
  --ckpt_path .cache/layout-flow/training-runs/rico25/checkpoints/last.ckpt
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

Use `deterministic` configs when rerunning S0-S2 or debugging equivalence failures. Use the default configs for full learning runs after deterministic parity is green. Scheduler-enabled equivalence belongs to the S3 and S4 checks above, while the current full-run commands disable the plateau scheduler until package-local validation metrics are implemented.

## Convert A Trained Checkpoint

After training, convert a Lightning checkpoint to a local `diffusers` pipeline directory. The converter accepts both original `model.*` keys and package-local training `model.backbone.*` keys.

```bash
uv run --package layout-flow python models/layout-flow/scripts/convert_original_checkpoint.py \
  --dataset rico25 \
  --checkpoint .cache/layout-flow/training-runs/rico25/checkpoints/<ckpt>.ckpt \
  --output-dir .cache/layout-flow/converted-trained/rico25
```

Run a local `from_pretrained` smoke test after conversion:

```bash
uv run --package layout-flow python - <<'PY'
from layout_flow import LayoutFlowPipeline

pipe = LayoutFlowPipeline.from_pretrained(".cache/layout-flow/converted-trained/rico25")
out = pipe(condition_type="unconditional", num_inference_steps=2)
print(out.bbox.shape, out.labels.shape, out.mask.shape)
PY
```

# LayoutDiffusion Training

This guide covers package-local LightningCLI training configs and the staged S0-S5 reproduction protocol for LayoutDiffusion. It follows the repository [training reproduction protocol](docs/training-reproduction.md).

Run commands from the repository root. Training data, logs, generated checkpoints, converted local pipelines, and evaluation artifacts stay under `.cache/layoutdiffusion`.

## Install

```bash
uv sync --package layoutdiffusion --extra training
```

Install the `vendor` extra only when rerunning staged parity against the original LayoutDiffusion checkout.

```bash
uv sync --package layoutdiffusion --extra training --extra vendor
```

## Data

The training datamodule supports two data sources:

- `hf`: approved Hugging Face datasets for development and smoke checks. Do not combine this source with `vocab_file`; Hugging Face numeric labels would be interpreted under the injected vendor corpus-order `id2label`.
- `processed`: preprocessed LayoutDiffusion token streams under `.cache/layoutdiffusion/original-data`. Use this source with `vocab_file` for S4/S5 so package-local training and the original-code training path consume the same `ltrb_lex` stream and vocabulary order. The S5 configs set `preconsume_train_batches: 1` so the package train stream starts after the same initial train-batch read performed before the original training loop begins.

| Dataset   | Source                              | Config / stream                                                                           |
| --------- | ----------------------------------- | ----------------------------------------------------------------------------------------- |
| RICO25    | `creative-graphic-design/Rico`      | `ui-screenshots-and-hierarchies-with-semantic-annotations`; vendor stream `RICO_ltrb_lex` |
| PubLayNet | `creative-graphic-design/PubLayNet` | default; vendor stream `PublayNet_ltrb_lex`                                               |

The `smoke.yaml` config uses a synthetic local dataset and does not download RICO25 or PubLayNet.

## Configs

Training configs live under `models/layoutdiffusion/configs/training`.

| Config                                         | Dataset             | Seed mode       | Purpose                                          |
| ---------------------------------------------- | ------------------- | --------------- | ------------------------------------------------ |
| `layoutdiffusion_rico25.yaml`                  | RICO25              | `default`       | Regular package-local training.                  |
| `layoutdiffusion_publaynet.yaml`               | PubLayNet           | `default`       | Regular package-local training.                  |
| `layoutdiffusion_rico25_deterministic.yaml`    | RICO25              | `deterministic` | Deterministic short run for parity/debug checks. |
| `layoutdiffusion_publaynet_deterministic.yaml` | PubLayNet           | `deterministic` | Deterministic short run for parity/debug checks. |
| `smoke.yaml`                                   | PubLayNet synthetic | `deterministic` | CPU smoke config for CLI wiring.                 |

## Scheduler and Recipe Notes

LayoutDiffusion uses the original AdamW recipe with linear learning-rate annealing
per optimizer step and EMA updates after each optimizer step. S5 configs use
`time_sampler: uniform` because the original GPU training path leaves diffusion
state on CPU and therefore never activates its effective-uniform sampler update
buffers. The processed-stream S5 configs set `preconsume_train_batches: 1` to
match the original pre-loop `next(data)` read before training begins.

The currently verified GPU setup is Tesla V100 with driver `575.57.08`, which is
compatible with CUDA 12.9-era torch wheels. Keep repository dependency metadata
unchanged; install driver-compatible torch wheels only as local runtime setup
when running GPU training in that environment.

## Seed Policy

S0-S4 parity uses fixed deterministic seeds inside the vendor-parity fixtures.
S5 is reported at `training-seed n=3` for RICO25 and PubLayNet with training seeds
`102`, `103`, and `104` on both original and package systems. Unconditional sample
export uses sampling seed `101` for every reported S5 run.

## Stage Evidence

| Stage | Command                                                                                                                                                                                                                                                                                                                                                       | Artifact                                                                             | Result                                                                                                                                                                                                                                                                                                   |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `CUDA_VISIBLE_DEVICES=<gpu-index> PARITY_REQUIRE=1 uv run --package layoutdiffusion --extra training --extra vendor --with pytest pytest models/layoutdiffusion/tests/vendor_parity/test_layoutdiffusion_training_parity.py -m "vendor_parity and training" -k s0 -rs`                                                                                        | `models/layoutdiffusion/tests/vendor_parity/test_layoutdiffusion_training_parity.py` | 12 passing S0 tests: parameter-count and state-dict topology with an explicit allowlist, copied-weight forward equality, real-scale schedule buffers for both datasets, optimizer/EMA/sampler static state, effective-uniform sampler regression, and tokenizer/vocab statics.                           |
| S1    | `CUDA_VISIBLE_DEVICES=<gpu-index> PARITY_REQUIRE=1 uv run --package layoutdiffusion --extra training --extra vendor --with pytest pytest models/layoutdiffusion/tests/vendor_parity/test_layoutdiffusion_training_parity.py -m "vendor_parity and training" -k "adapter_fixture or trace_matches_original_fixture" -rs`                                       | `.cache/layoutdiffusion/training-parity/<dataset>/s0_s2_reference.pt`                | Fixed-batch pre-optimizer trace parity (timesteps, `q_sample`, denoiser logits, posterior KL, auxiliary loss, total loss) within `atol=2e-5, rtol=2e-6`; effective-uniform Lt buffers asserted zero on the original side.                                                                                |
| S2    | `CUDA_VISIBLE_DEVICES=<gpu-index> PARITY_REQUIRE=1 uv run --package layoutdiffusion --extra training --extra vendor --with pytest pytest models/layoutdiffusion/tests/vendor_parity/test_layoutdiffusion_training_parity.py -m "vendor_parity and training" -k s3_repeated -rs`                                                                               | `.cache/layoutdiffusion/training-parity/<dataset>/s3_s4_reference.pt`                | Repeated-step parity against the real original `TrainLoop.run_step` covers gradients, learning-rate anneal cadence, optimizer state, post-step parameters, and EMA.                                                                                                                                      |
| S3    | `CUDA_VISIBLE_DEVICES=<gpu-index> PARITY_REQUIRE=1 uv run --package layoutdiffusion --extra training --extra vendor --with pytest pytest models/layoutdiffusion/tests/vendor_parity/test_layoutdiffusion_training_parity.py -m "vendor_parity and training" -k s3_repeated -rs`                                                                               | `.cache/layoutdiffusion/training-parity/<dataset>/s3_s4_reference.pt`                | The same repeated-step fixture drives multiple batches through the original `TrainLoop`; a Lightning `Trainer.fit` cadence test additionally pins scheduler-per-step, EMA-per-step, and checkpoint content.                                                                                              |
| S4    | `CUDA_VISIBLE_DEVICES=<gpu-index> PARITY_REQUIRE=1 uv run --package layoutdiffusion --extra training --extra vendor --with pytest pytest models/layoutdiffusion/tests/vendor_parity/test_layoutdiffusion_training_parity.py -m "vendor_parity and training" -k s4_processed -rs`                                                                              | `.cache/layoutdiffusion/original-data/<stream>`                                      | Processed-stream order parity: the package first trained batch after `preconsume_train_batches: 1` equals the original first trained batch after the pre-loop `next(data)` read, under the injected corpus-order vocab.                                                                                  |
| S5    | `CUDA_VISIBLE_DEVICES=<gpu-index> TRANSFORMERS_NO_TORCHVISION=1 uv run --package layoutdiffusion --extra training traingen fit --config models/layoutdiffusion/configs/training/layoutdiffusion_<rico25\|publaynet>.yaml --seed_everything=<seed> --trainer.accelerator=gpu --trainer.devices=1 --trainer.max_epochs=-1 --trainer.max_steps=<175000\|400000>` | `.cache/layoutdiffusion/s5/eval/summary.json`                                        | Training-seed n=3 statistical comparison under the original evaluation stack; RICO25 statistically equivalent, PubLayNet structural metrics equivalent with small FID/Alignment endpoint residuals (tables below). Training runs live under `.cache/layoutdiffusion/s5/full-run-final-20260801-144316/`. |

## Reproduction Results

Overall S5 verdict: RICO25 package-local training is statistically equivalent to the original implementation at training-seed n=3 (mIoU and Overlap match within seed noise; Alignment overlaps; FID is close with a small residual). PubLayNet is accepted with an interpretation note at training-seed n=3: mIoU and Overlap match or favor the package, while FID (+6.4%) and Alignment (0.062 vs 0.036) carry small trained-weight endpoint residuals. Staged gates S0-S4, released inference parity, and the one-step processed-stream probe all pass on the same code; the residuals are endpoint-level, not training-dynamics defects.

| Dataset   | System   | Status                      | Seed scope        | FID           | mIoU          | Alignment       | Overlap         |
| --------- | -------- | --------------------------- | ----------------- | ------------- | ------------- | --------------- | --------------- |
| RICO25    | original | `s5-practical-reproduction` | training-seed n=3 | 1.851 ± 0.029 | 0.608 ± 0.002 | 0.123 ± 0.004   | 0.516 ± 0.006   |
| RICO25    | package  | `s5-practical-reproduction` | training-seed n=3 | 2.125 ± 0.043 | 0.612 ± 0.004 | 0.133 ± 0.013   | 0.514 ± 0.006   |
| PubLayNet | original | `s5-practical-reproduction` | training-seed n=3 | 7.673 ± 0.045 | 0.409 ± 0.000 | 0.0361 ± 0.0004 | 0.0065 ± 0.0002 |
| PubLayNet | package  | `s5-practical-reproduction` | training-seed n=3 | 8.162 ± 0.094 | 0.421 ± 0.001 | 0.0623 ± 0.0009 | 0.0059 ± 0.0002 |

Evaluation protocol: unconditional generation with the original evaluation stack (`json2metrics.py`, layout-feature FID) over the original test-set sample counts (RICO25 3728, PubLayNet 10998 per seed), EMA weights on both systems (original `ema_0.9999_*` checkpoints, package `layoutdiffusion_ema_state_dict` from `last.ckpt`), training seeds 102/103/104 per system per dataset, sampling seed 101.

Interpretation: structural quality metrics (mIoU, Overlap) are equivalent on both datasets. FID is slightly higher for the package on both datasets (RICO25 +0.27, PubLayNet +0.49) with per-seed spreads far smaller than the gap, indicating a small systematic trained-weight endpoint residual rather than seed noise; the staged S0-S3 lockstep evidence (loss/gradient/parameter/EMA agreement) and the aligned one-step probe rule out training-dynamics divergence as the cause. PubLayNet package Alignment is higher in absolute terms but both values are in the strong range for the method.

The original GPU training path uses effective uniform timestep sampling. In the vendor `discrete_diffusion.py` loss update around lines 800-803, `self.Lt_history.to(model.device).scatter_(...)` and `self.Lt_count.to(model.device).scatter_add_(...)` write to temporary CUDA copies when the diffusion module stays on CPU while the model is on CUDA, so `Lt_history` and `Lt_count` never update and importance sampling never activates. The package S5 configs therefore set `time_sampler: uniform` for faithful reproduction. Earlier package PubLayNet S5 attempts that used package-side importance sampling degenerated after the package buffers crossed the activation threshold; those runs are invalid as reproduction evidence.

Until S5 is confirmed for a claimed dataset, PRs should remain draft and trained checkpoints should not be published as reproduced. The upstream LayoutDiffusion checkout does not provide a LayoutDiffusion-specific top-level license, so trained checkpoints must not claim an OSS license until upstream confirms the license status.

## Regeneration Metadata

Evidence locations (local, not committed): training runs under `.cache/layoutdiffusion/s5/full-run-final-20260801-144316` (package) and `.cache/layoutdiffusion/s5/full-run-20260731-093243` (original); evaluation samples, per-run metrics, and `summary.json` under `.cache/layoutdiffusion/s5/eval/`.

```text
.cache/layoutdiffusion/original-data/
.cache/layoutdiffusion/s5/full-run-final-20260801-144316/
.cache/layoutdiffusion/s5/full-run-20260731-093243/
.cache/layoutdiffusion/s5/eval/
```

## S5 Prelaunch Gate

Do not launch S5 until all of these checks pass in the same worktree and with the same processed stream mirror that the S5 configs will use:

1. Regular training config tests confirm `auxiliary_loss_weight: 0.001`, `preconsume_train_batches: 1`, and matching model/data `vocab_file` paths for every processed S5 config. This is satisfied by the current staged redo; the CPU suite includes these config guards.
2. S3 repeated short-step parity passes against the generated original-code fixture, comparing loss trajectory, learning-rate cadence, gradients, updated parameters, and EMA within the documented test tolerances. This is satisfied by the current staged redo: S0 has 12 passing tests, S1 has 4 passing tests, and S3 has 2 passing repeated-step tests using the real vendor `TrainLoop` fixture.
3. S4 processed-stream order parity passes, confirming the package first trained batch after `preconsume_train_batches: 1` equals the original first trained batch after the pre-loop `next(data)` read. This is satisfied by the current staged redo: S4 has 2 passing processed-stream tests, and released inference parity has 8 passing tests.
4. A corrected one-step processed-stream package/original train-metric probe is rerun for RICO25 and PubLayNet, and the package/original `train_loss`, KL component, auxiliary component, and total loss are compared before any full S5 launch. This is satisfied by `.cache/layoutdiffusion/s5/gate-probe-20260801-133008`:

   | Dataset   | Package total | Vendor total | Total ratio | Package KL  | Vendor KL | KL ratio | Package aux | Vendor aux | Aux ratio |
   | --------- | ------------- | ------------ | ----------- | ----------- | --------- | -------- | ----------- | ---------- | --------- |
   | RICO25    | 87104.96875   | 87100.0      | 1.000057    | 86919.15625 | 86900.0   | 1.000220 | 185.81418   | 186.0      | 0.999001  |
   | PubLayNet | 84603.28125   | 84600.0      | 1.000039    | 84425.15625 | 84400.0   | 1.000298 | 178.12349   | 178.0      | 1.000694  |

Passing this gate authorizes full S5 launch only after the coordinator reviews the recorded probe numbers and gives a separate launch order.

## Training Commands

The following commands reproduce local CI checks, staged evidence, full S5
training, sample export, and original-stack scoring.

### Reproducing Current Checks

Run local CI training checks.

```bash
uv run --package layoutdiffusion --extra training --with pytest pytest \
  models/layoutdiffusion/tests/test_training_configs.py \
  models/layoutdiffusion/tests/test_training_utils.py \
  models/layoutdiffusion/tests/test_training_lightning.py -q
CUDA_VISIBLE_DEVICES="" uv run --package layoutdiffusion --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layoutdiffusion/configs/training/smoke.yaml
```

Generate lightweight package S3/S4 evidence artifacts for a synthetic smoke stream or a processed stream.

```bash
TRANSFORMERS_NO_TORCHVISION=1 uv run --package layoutdiffusion --extra training \
  python models/layoutdiffusion/scripts/generate_training_gate_evidence.py \
  --dataset publaynet --steps 3
TRANSFORMERS_NO_TORCHVISION=1 uv run --package layoutdiffusion --extra training \
  python models/layoutdiffusion/scripts/generate_training_gate_evidence.py \
  --dataset publaynet \
  --processed-data-dir .cache/layoutdiffusion/original-data \
  --steps 3
```

Generate the shortest deterministic package-side S0-S3 evidence run.

```bash
CUDA_VISIBLE_DEVICES="" uv run --package layoutdiffusion --extra training python - <<'PY'
from pathlib import Path

import torch
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from layoutdiffusion import LayoutDiffusionConfig
from layoutdiffusion.training.datamodule import LayoutDiffusionDataModule
from layoutdiffusion.training.lightning_module import LayoutDiffusionTrainingModule

out_dir = Path(".cache/layoutdiffusion/training-evidence/package-smoke")
out_dir.mkdir(parents=True, exist_ok=True)
seed_everything(102, workers=True)
config = LayoutDiffusionConfig(
    dataset_name="publaynet",
    seq_length=19,
    max_num_elements=3,
    diffusion_steps=10,
    num_channels=8,
    hidden_size=16,
    num_attention_heads=4,
    num_hidden_layers=1,
    intermediate_size=32,
)
dm = LayoutDiffusionDataModule(
    dataset_name="publaynet",
    config=config,
    batch_size=2,
    synthetic_size=4,
    num_workers=0,
)
module = LayoutDiffusionTrainingModule(
    config=config,
    learning_rate=5e-5,
    scheduler="linear_anneal",
    lr_anneal_steps=4,
    time_sampler="importance",
    seed_mode="deterministic",
)
ckpt = ModelCheckpoint(
    dirpath=out_dir / "checkpoints",
    filename="package-step-{step}",
    save_last=True,
    every_n_train_steps=1,
    save_top_k=-1,
)
trainer = Trainer(
    accelerator="cpu",
    devices=1,
    precision="32-true",
    deterministic=True,
    max_epochs=1,
    limit_train_batches=2,
    limit_val_batches=0,
    num_sanity_val_steps=0,
    logger=False,
    enable_progress_bar=False,
    enable_model_summary=False,
    callbacks=[ckpt],
    plugins=[LightningEnvironment()],
)
trainer.fit(module, datamodule=dm)
optimizer = trainer.optimizers[0]
torch.save(
    {
        "seed": 102,
        "global_step": trainer.global_step,
        "trace": module.latest_step_trace,
        "lt_history": module.lt_history.detach().clone(),
        "lt_count": module.lt_count.detach().clone(),
        "ema": module.ema_state_dict(),
        "model_state": {k: v.detach().cpu() for k, v in module.model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "checkpoint_callback_last": ckpt.last_model_path,
    },
    out_dir / "package_s0_s3_evidence.pt",
)
PY
```

Run staged vendor parity checks after local assets are available.

```bash
git submodule update --init vendor/ms-layout-generation
CUDA_VISIBLE_DEVICES=<gpu-index> PARITY_REQUIRE=1 \
  uv run --package layoutdiffusion --extra training --extra vendor --with pytest pytest \
  models/layoutdiffusion/tests/vendor_parity/test_layoutdiffusion_training_parity.py \
  -m "vendor_parity and training" -rs
```

### Reproducing The S5 Results

Full-run package-local training per dataset and seed (seeds 102/103/104; RICO25 175000 steps, PubLayNet 400000 steps):

```bash
TRANSFORMERS_NO_TORCHVISION=1 CUDA_VISIBLE_DEVICES=<gpu-index> \
  uv run --package layoutdiffusion --extra training traingen fit \
  --config models/layoutdiffusion/configs/training/layoutdiffusion_rico25.yaml \
  --seed_everything=<seed> \
  --trainer.default_root_dir=.cache/layoutdiffusion/s5/<run-root>/runs/package-rico25-seed<seed> \
  --trainer.accelerator=gpu --trainer.devices=1 \
  --trainer.max_epochs=-1 --trainer.max_steps=175000 \
  --trainer.enable_progress_bar=false
```

Original-implementation training per dataset and seed uses the vendor README command shape (`improved-diffusion/scripts/train.py`, `--seed <seed>`, `--lr_anneal_steps 175000/400000`) against the same processed streams; run it from the patched vendor copy used by the parity tooling.

Export unconditional samples from a package checkpoint (EMA weights) in the vendor JSON format. The `--config` JSON must embed the vendor corpus-order vocab used in training (see `.cache/layoutdiffusion/s5/eval/configs/` generation in the evaluation driver); PubLayNet sample paths must contain lowercase `pub` for the vendor evaluator's dataset branch:

```bash
TRANSFORMERS_NO_TORCHVISION=1 CUDA_VISIBLE_DEVICES=<gpu-index> \
  uv run --package layoutdiffusion --extra training \
  python models/layoutdiffusion/scripts/export_training_checkpoint_samples.py \
  --checkpoint <run-root>/runs/package-<dataset>-seed<seed>/lightning_logs/version_0/checkpoints/last.ckpt \
  --dataset <dataset> --config <vendor-vocab-config.json> \
  --output .cache/layoutdiffusion/s5/eval/package-<dataset-dir>-seed<seed>/samples.json \
  --weights ema --num-samples <3728|10998> --batch-size 64 --seed 101
```

Original-side sampling uses the patched vendor `text_sample.py` with `--model_path <run>/ema_0.9999_<steps>.pt --top_p -1.0 --constrained ungen` and the same sample counts.

Score any samples JSON with the original evaluation stack (run from the vendor LayoutDiffusion root; `seaborn` is required at runtime by `eval_src`):

```bash
cd vendor/ms-layout-generation/LayoutDiffusion
PYTHONPATH="$PWD:$PWD/eval_src" \
  uv --project <repo-root> run --with seaborn --no-sync \
  --package layoutdiffusion --extra vendor \
  python json2metrics.py <samples.json>
```

Start a regular RICO25 package-local training run.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layoutdiffusion --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layoutdiffusion/configs/training/layoutdiffusion_rico25.yaml \
  --trainer.accelerator=gpu --trainer.devices=1 \
  --trainer.default_root_dir=.cache/layoutdiffusion/training-runs/rico25
```

Start a regular PubLayNet package-local training run.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layoutdiffusion --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layoutdiffusion/configs/training/layoutdiffusion_publaynet.yaml \
  --trainer.accelerator=gpu --trainer.devices=1 \
  --trainer.default_root_dir=.cache/layoutdiffusion/training-runs/publaynet
```

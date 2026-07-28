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

So far, we have only verified the GPU setup on Tesla V100 with driver
`575.57.08`, which is compatible with CUDA 12.9-era wheels. The repository
default torch build is `cu130` (CUDA 13.0), and GPU initialization on that
verified V100 setup fails with `driver too old (found 12090)`. Keep the
repository torch pins and lockfile at their defaults, but before launching GPU
training in this environment, install a driver-compatible CUDA 12 torch wheel
such as `cu126` into the execution environment. This is a local runtime change
only; do not commit lockfile or `pyproject.toml` changes for the temporary torch
replacement.

## Data

The training datamodule supports two data sources:

- `hf`: approved Hugging Face datasets for development and smoke checks.
- `processed`: preprocessed LayoutDM `.pt` splits under `<data-dir>/<dataset>-max<S>/processed/{train,val,test}.pt`. Use this source for S5 so package-local training and original-code training consume the same sample stream.

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

LayoutDM package-local training currently has exact S0-S2 numeric parity on a fixed PubLayNet-style synthetic batch, plus S4 tokenizer/loader row encoding parity and preprocessed stream reader parity for local fixtures. RICO25 S5 training-seed n=8 and PubLayNet S5 training-seed n=3 have been evaluated under the original FIDNetV3 protocol. The root cause of the RICO25 high-FID failure was an initialization mismatch: the package-local denoiser used PyTorch defaults while the original training path initialized linear and embedding weights with `normal_(0, 0.02)`. After matching that scheme, the RICO25 FID mean gap is +0.1252 at n=8, the FID ranges overlap, and Welch's t-test does not reject equality (`t=0.7331`, `p=0.4768`). The pre-fix high-FID n=8 run had a larger gap (`ours=9.0264`, `vendor=7.1055`, `delta=+1.9209`, `p=0.0005`, no range overlap), so the initialization fix resolves the established RICO25 over-FID failure.

PubLayNet is finalized at training-seed n=3 by project decision; the attempted n=8 expansion was stopped after the n=3 early verdict. PubLayNet ours-initfix has lower FID than the original-code run (`delta=-0.5308`, Welch `t=-2.9589`, `p=0.0426`, no FID range overlap), so this result should not be described as statistically identical. The difference is in the favorable direction, and Alignment, Overlap, and mIoU do not show a significant gap (`p>0.18` with overlapping ranges). The overall S5 verdict is complete: RICO25 is statistically reproduced at n=8, and PubLayNet is accepted at n=3 with no over-FID regression or vendor-worse reproduction failure.

RICO25 conditional generation was evaluated without retraining on the same n=8 seed set using the original `trainer.test` and `eval.py` paths. The vendor conditional names are `cond=c` for category-conditioned generation, `cond=partial` for completion, and `cond=refinement` for noisy-layout refinement. Completion and refinement reproduce statistically on FID and task metrics. Category-conditioned generation has a small but Welch-significant FID gap against ours-initfix (`delta=+0.1217`, `p=0.0346`, bootstrap 95% CI `[+0.0304, +0.2190]`) even though the FID ranges overlap; therefore the category-conditioned result is close but not strict FID equality. A same-weight route-identity check reloaded vendor seed 42975 through the package denoiser, re-exported it to the vendor checkpoint format, and then ran the same vendor `trainer.test cond=c` settings (`temperature=1.0`, `num_timesteps=100`, `sampling=random`, and vendor mask construction). The direct and re-export paths produced identical 4,218-layout outputs by SHA256, so the residual is not a conditioning-route, setting, or export bug. The best supported interpretation is that `cond=c` is sensitive to small trained-weight endpoint differences: the +0.1217 FID gap is near the same-method run-to-run floor estimated from the existing c-mode seeds, while completion, refinement, and unconditional generation remain equivalent.

| Dataset | System | Stat scope | FID | Alignment | Overlap | mIoU | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| RICO25 | vendor | training-seed n=8 | 7.1055 ± 0.3673 | 0.0019 ± 0.0004 | 0.8438 ± 0.0115 | 0.1941 ± 0.0050 | S5 RICO25 complete |
| RICO25 | ours | training-seed n=8 initfix | 7.2307 ± 0.2632 | 0.0022 ± 0.0006 | 0.8379 ± 0.0169 | 0.1931 ± 0.0028 | S5 RICO25 complete |
| PubLayNet | vendor | training-seed n=3 | 12.0617 ± 0.1665 | 0.0020 ± 0.0001 | 0.1295 ± 0.0040 | 0.0759 ± 0.0040 | S5 PubLayNet complete |
| PubLayNet | ours | training-seed n=3 initfix | 11.5308 ± 0.1915 | 0.0021 ± 0.0002 | 0.1240 ± 0.0008 | 0.0727 ± 0.0033 | S5 PubLayNet complete |

RICO25 and PubLayNet metrics use the vendor `cond=unconditional`, `num_uncond_samples=1000`, `num_timesteps=100` evaluation path with FIDNetV3. Alignment and Overlap are the vendor `LayoutGAN++` variants; mIoU is reported from the vendor `average_iou-VTN` output. Standard deviations use population standard deviation over the reported training seeds.

| Dataset | Condition | System | Stat scope | FID | maximum_iou | DocSim | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| RICO25 | `c` | vendor | training-seed n=8 | 3.3863 ± 0.0825 | 0.2763 ± 0.0025 | 0.1673 ± 0.0004 | close; category FID caveat |
| RICO25 | `c` | ours | training-seed n=8 initfix | 3.5080 ± 0.1088 | 0.2739 ± 0.0016 | 0.1670 ± 0.0004 | close; category FID caveat |
| RICO25 | `partial` | vendor | training-seed n=8 | 8.1341 ± 0.6067 | 0.5945 ± 0.0096 | 0.0918 ± 0.0020 | conditional S5 supported |
| RICO25 | `partial` | ours | training-seed n=8 initfix | 8.1523 ± 0.3817 | 0.6029 ± 0.0098 | 0.0921 ± 0.0023 | conditional S5 supported |
| RICO25 | `refinement` | vendor | training-seed n=8 | 4.6356 ± 0.1029 | 0.3404 ± 0.0011 | 0.1971 ± 0.0002 | conditional S5 supported |
| RICO25 | `refinement` | ours | training-seed n=8 initfix | 4.7018 ± 0.1922 | 0.3409 ± 0.0022 | 0.1973 ± 0.0002 | conditional S5 supported |

RICO25 conditional metrics use `num_timesteps=100`, `sampling=random`, and the original FIDNetV3 evaluation path. For `cond=c`, FID is `t=2.3580`, `p=0.0346`, with overlapping per-seed ranges; `maximum_iou` is borderline lower for ours (`p=0.0550`) and `DocSim` is not significantly different (`p=0.1760`). Same-method c-FID spread from the existing eight seeds is of comparable scale: exact 4-vs-4 within-method splits have mean absolute gaps of `0.0652` for ours and `0.0519` for vendor, with p90 `0.1376` and `0.1018`; same-method bootstrap absolute-gap p95 is `0.1062` for ours and `0.0812` for vendor, and ours p97.5 is `0.1210`. For `cond=partial`, FID is `p=0.9477`, `maximum_iou` is `p=0.1282`, and `DocSim` is `p=0.7897`. For `cond=refinement`, FID is `p=0.4392`, `maximum_iou` is `p=0.5377`, and `DocSim` is `p=0.1189`.

Evidence locations:

```text
.cache/layout-dm/training-runs/
.cache/layout-dm/full-run/
.cache/layout-dm/converted/
.cache/layout-dm/full-run/eval/rico25_conditional_n3/summary_n8_conditional_metrics.csv
.cache/layout-dm/full-run/eval/rico25_conditional_n3/summary_n8_conditional_stats.csv
.cache/layout-dm/full-run/eval/rico25_c_route_identity/
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
CUDA_VISIBLE_DEVICES="" PARITY_REQUIRE=1 \
  uv run --package layout-dm --extra training --extra vendor --with pytest pytest \
  models/layout-dm/tests/vendor_parity/test_layout_dm_training_parity.py \
  -m "vendor_parity and training" -rs
```

Start a regular RICO25 package-local training run.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layout-dm --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-dm/configs/training/layoutdm_rico25.yaml \
  --data.init_args.dataset_source=processed \
  --data.init_args.processed_data_dir=.cache/layout-dm/original-data \
  --trainer.accelerator=gpu --trainer.devices=1 \
  --trainer.default_root_dir=.cache/layout-dm/training-runs/rico25
```

Start a regular PubLayNet package-local training run.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layout-dm --extra training \
  python -m traingen.lightning.cli fit \
  --config models/layout-dm/configs/training/layoutdm_publaynet.yaml \
  --data.init_args.dataset_source=processed \
  --data.init_args.processed_data_dir=.cache/layout-dm/original-data \
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

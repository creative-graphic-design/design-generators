# Training CGB-DM

CGB-DM training is reproducible enough to support CGL practical parity with the
package implementation under the reference architecture, reference dataset
encoding, and raw-internal S5 evaluation protocol. PKU PosterLayout is not a
practical-parity claim: the seed-variance matrix shows no-underlay collapse in
both the original implementation and the package, so the failure is not
package-exclusive. The observed collapse rates are 3/4 original runs and 4/5
package runs; that sample is enough to reject a package-only explanation but
too small to support a stable frequency comparison. This matches the
trajectory-sensitivity risk tracked in
[issue #148](https://github.com/creative-graphic-design/design-generators/issues/148).

Run commands from the repository root. Generated checkpoints, sample tensors,
metric summaries, converted local pipelines, and downloaded assets stay outside
git under `.cache/cgb-dm/`.

## Install

Install the package-local training dependencies.

```bash
uv sync --package cgb-dm --extra training
```

Install the `vendor` extra only when rerunning original-code parity checks or
the original-code S5 evaluator.

```bash
uv sync --package cgb-dm --extra training --extra vendor
```

## Data

PKU PosterLayout and CGL use the original CGB-DM asset structure. The original
asset zip remains authoritative for parity because it includes the image,
saliency, saliency-box, and CSV ordering needed to mirror the upstream training
loop.

| Dataset | Source | Config or path |
| --- | --- | --- |
| PKU PosterLayout CGB-DM | `creative-graphic-design/PKU-PosterLayout` plus original CGB-DM assets | `.cache/cgb-dm/datasets/pku/split`; validation path `val/inpaint`; source-order manifest `.cache/cgb-dm/reference/pku_posterlayout_train_manifest.json` |
| CGL CGB-DM | `creative-graphic-design/CGL-Dataset` plus original CGB-DM assets | `.cache/cgb-dm/datasets/cgl/split`; validation path `val/inpaint` |

## Configs

Training configs live under `models/cgb-dm/configs/training`.

| Config | Dataset | Seed mode | Purpose |
| --- | --- | --- | --- |
| `smoke.yaml` | synthetic/local smoke | deterministic smoke | Local configuration and Lightning startup smoke check. |
| `cgb_dm_pku_posterlayout.yaml` | PKU PosterLayout CGB-DM | default | Reference-compatible PKU full training. |
| `cgb_dm_pku_posterlayout_deterministic.yaml` | PKU PosterLayout CGB-DM | deterministic | Deterministic PKU short-run and parity diagnostics. |
| `cgb_dm_cgl.yaml` | CGL CGB-DM | default | Reference-compatible CGL full training. |
| `cgb_dm_cgl_deterministic.yaml` | CGL CGB-DM | deterministic | Deterministic CGL short-run and parity diagnostics. |

## Scheduler and Recipe Notes

The reproducible package run uses `CGBDMTransformerModel` with 47.9M
parameters, matching the reference `LayoutModel` architecture. PKU reference
encoding uses the internal layout vocabulary `0=padding/invalid` and
`1..3=layout classes`, and full PKU runs use the captured source-order manifest
for the original PKU training split.

The recorded recipe uses Adam with `lr=1e-4`, betas `(0.9, 0.999)`, `eps=1e-8`,
no weight decay, `CosineAnnealingLR(T_max=500)`, and gradient clipping at
`1.0`. The S3 launch metadata additionally records explicit
`--model.init_args.optimizer.*` overrides for those Adam settings.

S5 evaluation uses the validation split with raw internal `argmax` class ids
and raw generated boxes passed to the original metric formulas. PKU S5 uses
1,000 samples per evaluation seed. CGL S5 uses 6,055 samples per evaluation
seed.

## Seed Policy

PKU PosterLayout uses evaluation seeds 1, 2, and 3 for the package and
reference S5 comparisons. The seed-variance matrix covers original training
seeds 42, 43, 44, and 45 plus package training seeds 42, 43, 44, 45, and 46.
That matrix supports a recipe-instability verdict, not a stable collapse-rate
frequency estimate.

CGL uses evaluation-seed n=3 for the package and reference S5 comparisons. The
CGL result is a practical reproduction claim under the raw-internal S5 protocol.

## Validation Stages

| Stage | Scope | Purpose |
| --- | --- | --- |
| S0 | Static config and initialized state parity | Confirm PKU source-order manifest replay and reference-compatible topology inputs. |
| S1 | Fixed-batch pre-optimizer trace parity | Confirm fixed-batch forward and training trace agreement. |
| S2 | One optimizer-step parity | Confirm gradients, Adam state, and post-step parameters within documented tolerances. |
| S3 | Short deterministic multi-batch run | Confirm LightningCLI launch metadata, GPU, config, startup verification, and optimizer overrides. |
| S4 | Deterministic loader stream | Regenerate PKU source-order metadata for deterministic loader/order replay. |
| S5 | Full-run statistical comparison | Document PKU recipe instability and CGL practical parity under the raw-internal evaluator. |

## Stage Evidence

The rows below point to existing local evidence only. Generated checkpoints,
sample tensors, and full-run metric summaries stay outside git under `.cache/`.

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz pytest models/cgb-dm/tests/vendor_parity/test_cgb_dm_training_parity.py -m vendor_parity -k s0 -rs` | `.cache/cgb-dm/reference/pku_posterlayout_train_manifest.json` | PKU source-order manifest replay matches the original loader rows. |
| S1 | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz pytest models/cgb-dm/tests/vendor_parity/test_cgb_dm_training_parity.py -m vendor_parity -k s1 -rs` | `.cache/cgb-dm/reference/metadata.json` | Fixed-batch forward and training trace checks pass locally. |
| S2 | `PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz pytest models/cgb-dm/tests/vendor_parity/test_cgb_dm_training_parity.py -m vendor_parity -k s2 -rs` | `.cache/cgb-dm/reference/metadata.json` | One optimizer step matches gradients, Adam state, and post-step parameters within documented tolerances. |
| S3 | `CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra training --with tensorboard --with jsonargparse[signatures]>=4.27.7 python -m traingen.lightning.cli fit --config models/cgb-dm/configs/training/cgb_dm_pku_posterlayout.yaml --seed_everything 1 --trainer.accelerator gpu --trainer.devices 1 --trainer.default_root_dir .cache/cgb-dm/full-run/ours-pku/pku_full_ours_20260724_013039` | `.cache/cgb-dm/full-run/ours-pku/pku_full_ours_20260724_013039/run_metadata.json` | Full LightningCLI launch metadata records the package training command, GPU, config, and startup verification; the recorded full run additionally carried explicit `--model.init_args.optimizer.*` overrides for the Adam settings summarized above. |
| S4 | `CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm python models/cgb-dm/scripts/generate_reference_outputs.py --dataset pku_posterlayout --data-root .cache/cgb-dm/datasets/pku/split --manifest-output .cache/cgb-dm/reference/pku_posterlayout_train_manifest.json` | `.cache/cgb-dm/reference/pku_posterlayout_train_manifest.json` | Regenerated PKU source-order metadata is available for deterministic loader/order replay. |
| S5 | `CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz python models/cgb-dm/scripts/evaluate_full_run.py --backend ours --repo-root "$PWD" --data-root .cache/cgb-dm/datasets/pku/split --checkpoint .cache/cgb-dm/full-run/ours-pku-fixed/pku_full_ours_archfixed_20260724_122952/lightning_logs/version_0/checkpoints/epoch=499-step=121000.ckpt --output-dir .cache/cgb-dm/full-run/s5-eval-ours-pku-val-archfixed --gpu 0 --seeds 1 2 3` | `.cache/cgb-dm/full-run/s5-eval-ours-pku-val-archfixed/summary.json` | PKU S5 verdict is no-underlay collapse that is not package-exclusive; CGL S5 practical parity is recorded separately in `.cache/cgb-dm/full-run/s5-eval-cgl-comparison.json`. |

## Reproduction Results

CGB-DM has practical reproduction for CGL under the raw-internal S5 evaluation
protocol. PKU PosterLayout remains documented recipe instability: the
seed-variance matrix shows no-underlay collapse in both implementations, with
3/4 original runs and 4/5 package runs collapsed/no-underlay.

| Dataset | System | Status | Seed scope | Primary metrics | Loss evidence | Artifact summary |
| --- | --- | --- | --- | --- | --- | --- |
| PKU PosterLayout CGB-DM | original implementation | `recipe-unstable (documented)` | training-seed n=4; evaluation-seed n=3 per row | seed42 standalone mean `val=1.000000 +/- 0.000000`, `ove=0.002286 +/- 0.000156`, `undl=0.996406 +/- 0.001368`, `unds=0.972385 +/- 0.000736`, `occ=0.127496 +/- 0.000878`, `rea=0.015695 +/- 0.000349`; 3/4 original runs collapsed/no-underlay | Reference full training log emitted `val=1.000000`, `ove=0.002727`, `undl=0.996477`, `unds=0.978788`, `occ=0.127215`, and `rea=0.015321` after epoch 500. | `.cache/cgb-dm/full-run/s5-eval-vendor-pku-val-fast/summary.json`; `.cache/cgb-dm/full-run/s5-evaluation-pku-seed-variance/` |
| PKU PosterLayout CGB-DM | package | `recipe-unstable (documented)` | training-seed n=5; evaluation-seed n=3 per row | seed42 standalone mean `val=1.000000 +/- 0.000000`, `ove=0.003293 +/- 0.000801`, `undl=0.999345 +/- 0.000284`, `unds=0.991428 +/- 0.001467`, `occ=0.116661 +/- 0.000648`, `rea=0.014180 +/- 0.000295`; 4/5 package runs collapsed/no-underlay | Training loss trajectories for collapsed and non-collapsed runs follow the same broad decay and do not expose a clear collapse boundary on their own. | `.cache/cgb-dm/full-run/s5-eval-ours-pku-val-archfixed/summary.json`; `.cache/cgb-dm/full-run/s5-evaluation-pku-seed-variance/` |
| CGL CGB-DM | original implementation | `s5-practical-reproduction` | evaluation-seed n=3 | `val=0.999097 +/- 0.000109`, `ove=0.001795 +/- 0.000044`, `undl=0.997452 +/- 0.000849`, `unds=0.983453 +/- 0.001646`, `occ=0.115873 +/- 0.000318`, `rea=0.005768 +/- 0.000118` | Reference full training log emitted `val=0.998943`, `ove=0.002324`, `undl=0.996198`, `unds=0.982091`, `occ=0.115683`, and `rea=0.005327` after epoch 500. | `.cache/cgb-dm/full-run/s5-eval-vendor-cgl-val/`; `.cache/cgb-dm/full-run/s5-eval-cgl-comparison.json` |
| CGL CGB-DM | package | `s5-practical-reproduction` | evaluation-seed n=3 | `val=0.999213 +/- 0.000044`, `ove=0.001790 +/- 0.000203`, `undl=0.996399 +/- 0.001292`, `unds=0.987553 +/- 0.002680`, `occ=0.116357 +/- 0.000279`, `rea=0.005971 +/- 0.000115` | The package and reference runs are statistically equivalent under the same raw-internal S5 protocol. | `.cache/cgb-dm/full-run/s5-eval-ours-cgl-val/`; `.cache/cgb-dm/full-run/s5-eval-cgl-comparison.json` |

PKU seed-variance matrix:

| Run | System | Nonpad/sample | Underlay total | Underlay/sample | Samples with underlay | `unds` | Label |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| original_seed42 | original implementation | 3.742000 | 2028 | 0.676000 | 1756 | 0.972385 | non-collapsed |
| original_seed43 | original implementation | 0.863000 | 0 | 0.000000 | 0 | NaN | collapsed/no-underlay |
| original_seed44 | original implementation | 0.896667 | 0 | 0.000000 | 0 | NaN | collapsed/no-underlay |
| original_seed45 | original implementation | 0.873333 | 0 | 0.000000 | 0 | NaN | collapsed/no-underlay |
| package_seed42 | package | 3.364667 | 1865 | 0.621667 | 1644 | 0.991428 | non-collapsed |
| package_seed43 | package | 0.816000 | 0 | 0.000000 | 0 | NaN | collapsed/no-underlay |
| package_seed44 | package | 0.849667 | 0 | 0.000000 | 0 | NaN | collapsed/no-underlay |
| package_seed45 | package | 0.941667 | 0 | 0.000000 | 0 | NaN | collapsed/no-underlay |
| package_seed46 | package | 0.877000 | 0 | 0.000000 | 0 | NaN | collapsed/no-underlay |

The difference between 75% and 80% is not meaningful at this sample size; the
defensible conclusion is that no-underlay collapse is shared by both
implementations and should be treated as recipe/trajectory instability rather
than a package-exclusive regression. `undl`/`unds` reported as `NaN` means the
metric is undefined because no underlay was generated.

Training-trajectory notes:

- Original seed42 has saved late checkpoints at epochs 400, 450, and 500. A
  seed-1 probe generated underlay at all three points: `underlay_total=493` at
  epoch 400, `674` at epoch 450, and `647` at epoch 500.
- The newly trained collapsed runs were sampled from live checkpoints around
  epochs 25-30. Those early snapshots had only rare underlay class ids:
  original seed44 `7`, original seed45 `21`, package seed45 `42`, and package
  seed46 `1` underlay element across 1,000 samples. Their final S5 outputs have
  zero underlay across 3,000 samples per run.
- Dense intermediate checkpoints are not available for the collapsed runs, so
  the exact epoch where underlay generation disappears cannot be localized from
  the current artifacts.

## Regeneration Metadata

Record generated evidence under `.cache/cgb-dm/`. The PKU seed-variance matrix
was regenerated from existing dumps using evaluation seeds 1, 2, and 3 for each
row. The matrix is a separate derived analysis artifact, not the direct output
of the S5 `evaluate_full_run.py` command in the Stage Evidence table.

```text
.cache/cgb-dm/reference/pku_posterlayout_train_manifest.json
.cache/cgb-dm/reference/metadata.json
.cache/cgb-dm/full-run/ours-pku/pku_full_ours_20260724_013039/run_metadata.json
.cache/cgb-dm/full-run/ours-pku-fixed/pku_full_ours_archfixed_20260724_122952/lightning_logs/version_0/checkpoints/epoch=499-step=121000.ckpt
.cache/cgb-dm/full-run/vendor-pku/checkpoints/pku_full_vendor_20260723_224914/Epoch500_cgbdm_weights.pth
.cache/cgb-dm/full-run/s5-eval-ours-pku-val-archfixed/summary.json
.cache/cgb-dm/full-run/s5-eval-vendor-pku-val-fast/summary.json
.cache/cgb-dm/full-run/s5-evaluation-pku-seed-variance/pku-seed-variance-matrix.json
.cache/cgb-dm/full-run/s5-evaluation-pku-seed-variance/pku-seed-variance-matrix.md
.cache/cgb-dm/full-run/s5-evaluation-pku-seed-variance/vendor-seed43/summary.json
.cache/cgb-dm/full-run/s5-evaluation-pku-seed-variance/vendor-seed44/summary.json
.cache/cgb-dm/full-run/s5-evaluation-pku-seed-variance/vendor-seed45/summary.json
.cache/cgb-dm/full-run/s5-evaluation-pku-seed-variance/seed43-package-replicate/summary.json
.cache/cgb-dm/full-run/s5-evaluation-pku-seed-variance/seed44-package-replicate/summary.json
.cache/cgb-dm/full-run/s5-evaluation-pku-seed-variance/seed45-package-replicate/summary.json
.cache/cgb-dm/full-run/s5-evaluation-pku-seed-variance/seed46-package-replicate/summary.json
.cache/cgb-dm/full-run/ours-cgl/cgl_full_ours_archfixed_20260725_012723/lightning_logs/version_0/checkpoints/epoch=499-step=189500.ckpt
.cache/cgb-dm/full-run/vendor-cgl/cgl_full_vendor_20260725_012722/checkpoints/cgl_full_vendor_20260725_012722/Epoch500_cgbdm_weights.pth
.cache/cgb-dm/full-run/s5-eval-ours-cgl-val/
.cache/cgb-dm/full-run/s5-eval-vendor-cgl-val/
.cache/cgb-dm/full-run/s5-eval-cgl-comparison.json
```

## Training Commands

Download the original assets.

```bash
uv run --package cgb-dm python models/cgb-dm/scripts/download_original_assets.py
```

Generate the PKU source-order manifest before starting a full training run.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm \
  python models/cgb-dm/scripts/generate_reference_outputs.py \
  --dataset pku_posterlayout \
  --data-root .cache/cgb-dm/datasets/pku/split \
  --manifest-output .cache/cgb-dm/reference/pku_posterlayout_train_manifest.json
```

Run the staged vendor parity checks after the submodule and local assets are
available.

```bash
PARITY_REQUIRE=1 \
CGB_DM_DATA_ROOT=.cache/cgb-dm/datasets/pku/split \
CGB_DM_VENDOR_ORDER_MANIFEST=.cache/cgb-dm/reference/pku_posterlayout_train_manifest.json \
CUDA_VISIBLE_DEVICES=<gpu-index> \
uv run --package cgb-dm --extra vendor --with pytz \
  pytest models/cgb-dm/tests/vendor_parity -m vendor_parity
```

Train the package model with the reference-compatible PKU config.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra training \
  traingen fit \
  --config models/cgb-dm/configs/training/cgb_dm_pku_posterlayout.yaml \
  --trainer.default_root_dir .cache/cgb-dm/full-run/ours-pku
```

Re-run the package PKU checkpoint comparison.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz \
  python models/cgb-dm/scripts/evaluate_full_run.py \
  --backend ours \
  --repo-root "$PWD" \
  --data-root .cache/cgb-dm/datasets/pku/split \
  --checkpoint .cache/cgb-dm/full-run/ours-pku-fixed/pku_full_ours_archfixed_20260724_122952/lightning_logs/version_0/checkpoints/epoch=499-step=121000.ckpt \
  --output-dir .cache/cgb-dm/full-run/s5-eval-ours-pku-val-archfixed \
  --gpu 0 \
  --seeds 1 2 3
```

Re-run the reference PKU checkpoint comparison.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz \
  python models/cgb-dm/scripts/evaluate_full_run.py \
  --backend reference \
  --repo-root "$PWD" \
  --data-root .cache/cgb-dm/datasets/pku/split \
  --checkpoint .cache/cgb-dm/full-run/vendor-pku/checkpoints/pku_full_vendor_20260723_224914/Epoch500_cgbdm_weights.pth \
  --output-dir .cache/cgb-dm/full-run/s5-eval-vendor-pku-val-fast \
  --gpu 0 \
  --seeds 1 2 3
```

Re-run the package CGL checkpoint comparison.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz \
  python models/cgb-dm/scripts/evaluate_full_run.py \
  --dataset cgl \
  --backend ours \
  --repo-root "$PWD" \
  --data-root .cache/cgb-dm/datasets/cgl/split \
  --checkpoint .cache/cgb-dm/full-run/ours-cgl/cgl_full_ours_archfixed_20260725_012723/lightning_logs/version_0/checkpoints/epoch=499-step=189500.ckpt \
  --output-dir .cache/cgb-dm/full-run/s5-eval-ours-cgl-val \
  --gpu 0 \
  --seeds 1 2 3
```

Re-run the reference CGL checkpoint comparison.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package cgb-dm --extra vendor --with pytz \
  python models/cgb-dm/scripts/evaluate_full_run.py \
  --dataset cgl \
  --backend reference \
  --repo-root "$PWD" \
  --data-root .cache/cgb-dm/datasets/cgl/split \
  --checkpoint .cache/cgb-dm/full-run/vendor-cgl/cgl_full_vendor_20260725_012722/checkpoints/cgl_full_vendor_20260725_012722/Epoch500_cgbdm_weights.pth \
  --output-dir .cache/cgb-dm/full-run/s5-eval-vendor-cgl-val \
  --gpu 0 \
  --seeds 1 2 3
```

Convert and smoke-test a local checkpoint directory.

```bash
uv run --package cgb-dm python models/cgb-dm/scripts/convert_training_checkpoint.py \
  --checkpoint .cache/cgb-dm/checkpoints/example.ckpt \
  --output-dir .cache/cgb-dm/converted/pku
uv run --package cgb-dm python models/cgb-dm/scripts/smoke_from_pretrained.py \
  --path .cache/cgb-dm/converted/pku
```

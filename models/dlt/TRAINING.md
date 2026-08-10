# Training DLT

DLT training uses the shared class-path-driven LightningCLI entry point. The
package does not define `dlt.training.cli`. PubLayNet and RICO13 have accepted
S5 practical reproduction with stochastic residuals disclosed; Magazine remains
gated until polygon and train-only handling is amended.

Run commands from the repository root. Generated checkpoints, result JSON,
converted local pipelines, and downloaded assets stay outside git under
`.cache/dlt/`.

## Install

Install the package-local training dependencies.

```bash
uv sync --package dlt --extra training
```

Install the `vendor` extra only when rerunning original-code parity checks or
S5 evaluators.

```bash
uv sync --package dlt --extra training --extra vendor
```

## Data

| Dataset | Source | Config or path |
| --- | --- | --- |
| PubLayNet | `creative-graphic-design/PubLayNet` | PubLayNet HDF5 training data under `.cache/dlt/`; validation uses `all` conditioning with batch size 64. |
| RICO13 | `creative-graphic-design/Rico` with `name="ui-screenshots-and-hierarchies-with-semantic-annotations"` and the DLT label mapping | RICO13 valid-box-filtered data; the aligned preparation kept `train=19204` and `val=1129` records for both vendor and package paths. |
| Magazine | `creative-graphic-design/magazine` | Gated until polygon and train-only handling is amended. |

## Configs

Training configs live under `models/dlt/configs/training`.

| Config | Dataset | Seed mode | Purpose |
| --- | --- | --- | --- |
| `smoke.yaml` | synthetic/local smoke | deterministic smoke | Local CPU LightningCLI class-path, optimizer, scheduler, clipping, and synthetic loader smoke. |
| `dlt_publaynet.yaml` | PubLayNet | default | Full PubLayNet package training. |
| `dlt_publaynet_deterministic.yaml` | PubLayNet | deterministic | Deterministic PubLayNet diagnostics. |
| `dlt_rico13.yaml` | RICO13 | default | Full RICO13 package training. |
| `dlt_rico13_deterministic.yaml` | RICO13 | deterministic | Deterministic RICO13 diagnostics. |
| `dlt_magazine.yaml` | Magazine | default | Gated Magazine recipe, pending polygon/train-only handling. |

## Scheduler and Recipe Notes

All class-path training configs use AdamW, per-step warmup-cosine scheduling,
and `gradient_clip_val=1.0`. The PubLayNet configs pin
`num_warmup_steps=100000` and `num_training_steps=1994400`, matching the
evaluated S5 checkpoint's `global_step=1994400` and scheduler
`last_epoch=1994400`. RICO13 uses the vendor warmup of `10000` steps and
Magazine uses the vendor warmup of `2000` steps; their total training steps are
resolved by Lightning from the active datamodule.

PubLayNet and RICO13 package training pairs have been evaluated. Magazine
remains gated until polygon/train-only handling is amended.

## Seed Policy

PubLayNet S5 compares a reference PubLayNet checkpoint trained from scratch
with the vendor implementation using seed `42` against an independently trained
package checkpoint using the same PubLayNet validation split, `all`
conditioning, and evaluation seeds `42`, `43`, and `44`. The final
seed-variance control compares vendor samples `vendor42`, `vendor43`, and
`vendor44` against package `lr-step` seeds `42`, `45`, and `46`.

RICO13 S5 compares the vendor seed-42 `checkpoint-799` against the package
seed-42 `final-epoch799.ckpt` on the RICO13 validation split, `all`
conditioning, and sampling seeds `42`, `43`, and `44`.

Magazine has no S5 run yet.

## Validation Stages

| Stage | Scope | Purpose |
| --- | --- | --- |
| S0 | Static config and initialized state parity | Confirm package-vs-vendor topology and initialized state dicts for tiny test and full PubLayNet configurations. |
| S1 | Fixed-batch pre-optimizer trace parity | Confirm prepared inputs, noise/timestep sampling, model predictions, masked losses, and total loss. |
| S2 | One optimizer-step parity | Confirm loss definition parity and real-batch PubLayNet diagnostic deltas. |
| S3 | Short deterministic multi-batch run | Exercise LightningCLI class-path wiring, AdamW, per-step warmup-cosine scheduling, clipping, synthetic data loading, and one train batch. |
| S4 | Deterministic loader stream | Verify HDF5 loading, padding/filtering, shuffling, scheduler stepping, and trace adapter coverage. |
| S5 | Full-run statistical comparison | Accept PubLayNet and RICO13 practical reproduction; keep Magazine gated. |

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `CUDA_VISIBLE_DEVICES="" PARITY_REQUIRE=1 uv run --package dlt --extra training --extra vendor pytest models/dlt/tests/vendor_parity/test_dlt_training_parity.py -m "vendor_parity and training" -rs` | `models/dlt/tests/vendor_parity/test_dlt_training_parity.py` | Static package-vs-vendor topology and initialized state dicts match exactly for the tiny test and full PubLayNet configurations. |
| S1 | `CUDA_VISIBLE_DEVICES="" PARITY_REQUIRE=1 uv run --package dlt --extra training --extra vendor pytest models/dlt/tests/vendor_parity/test_dlt_training_parity.py -m "vendor_parity and training" -rs` | `models/dlt/tests/vendor_parity/test_dlt_training_parity.py` | Fixed-batch pre-optimizer trace parity covers `box`, `box_cond`, `cat`, `mask_box`, `mask_cat`, `noise`, `t`, noised boxes/categories, model predictions, masked L2, masked CE, and total loss. |
| S2 | `CUDA_VISIBLE_DEVICES="" PARITY_REQUIRE=1 uv run --package dlt --extra training --extra vendor pytest models/dlt/tests/vendor_parity/test_dlt_training_parity.py -m "vendor_parity and training" -rs` | `models/dlt/tests/vendor_parity/test_dlt_training_parity.py` | One training-step loss definition is bit-identical; the S5 matched-batch diagnostic later confirmed `max_abs_ours_loss_def_delta=0.0`, `max_abs_l2_def_delta=0.0`, and `max_abs_ce_def_delta=0.0` on real PubLayNet batches. |
| S3 | `CUDA_VISIBLE_DEVICES="" uv run --package dlt --extra training python -m traingen.lightning.cli fit --config models/dlt/configs/training/smoke.yaml` | `models/dlt/configs/training/smoke.yaml` | Deterministic CPU short run exercises LightningCLI class-path wiring, AdamW, per-step warmup-cosine scheduling, gradient clipping, synthetic data loading, and one train batch without checkpoint artifacts. |
| S4 | `uv run --package dlt --extra training pytest models/dlt/tests/test_dlt_training.py -m training -q` | `models/dlt/tests/test_dlt_training.py` | HDF5 data loading, padding/filtering, per-access element shuffling, reference epoch-sampling RNG consumption, scheduler stepping, and trace adapter coverage are verified on local fixtures; the full PubLayNet data audit is recorded in `.cache/dlt/full-run/dlt_training_repro_primary_diagnosis.json`. |
| S5 | `CUDA_VISIBLE_DEVICES=<gpu-id> uv run --package dlt --extra training --extra vendor python .cache/dlt/full-run/scripts/evaluate_s5_rico13.py --ours-checkpoint .cache/dlt/full-run/ours-rico13/checkpoints/final-epoch799.ckpt --output .cache/dlt/full-run/s5-evaluation-rico13/vendor-vs-package.json --seeds 42 43 44 --device cuda` | `.cache/dlt/full-run/s5-evaluation-rico13/vendor-vs-package.json` | PubLayNet and RICO13 practical reproduction are accepted. PubLayNet cross residuals fall inside within-implementation seed variation, and RICO13 residuals are small, sign-reversing across seeds, and inside the PubLayNet seed-variance reference ranges. |

## Reproduction Results

PubLayNet and RICO13 S5 are accepted as practical reproduction, not bit-level
training parity. PubLayNet cross residuals fall inside within-implementation
seed variation for every reported metric. RICO13 residuals are small,
sign-reversing across seeds, and inside the PubLayNet seed-variance reference
ranges. Magazine remains not yet run.

| Dataset | System | Status | Seed scope | Primary metrics | Loss evidence | Artifact summary |
| --- | --- | --- | --- | --- | --- | --- |
| PubLayNet | vendor seed-42 reference vs package `final-epoch799.ckpt` | `s5-practical-reproduction` | evaluation-seed n=3 plus seed-variance controls | vendor FID `2.3806 +/- 0.0546`; package FID `2.4858 +/- 0.0528`; delta `+0.1051`; overlap delta `+0.0024`; alignment delta `+0.0003`; IoU delta `+0.0000` | vendor loss `1.8069 +/- 0.0228`; package loss `1.8068 +/- 0.0223`; delta `-0.0001`; matched-batch loss definition deltas are `0.0` for total, L2, and CE. | `.cache/dlt/full-run/s5-evaluation-reference-callback-seed42/results.json`; `.cache/dlt/full-run/s5-evaluation-seed-variance/final-verdict.json`; `.cache/dlt/full-run/dlt_training_repro_primary_diagnosis.json` |
| RICO13 | vendor seed-42 checkpoint vs package seed-42 `final-epoch799.ckpt` | `s5-practical-reproduction` | evaluation-seed n=3 | vendor FID `3.4915 +/- 0.1143`; package FID `3.4812 +/- 0.1450`; delta `-0.0103`; overlap delta `-0.0167`; alignment delta `-0.0001`; IoU delta `-0.0225` | vendor loss `1.8105 +/- 0.0444`; package loss `1.8056 +/- 0.0399`; delta `-0.0049`; per-seed signs reverse for FID, alignment, and loss. | `.cache/dlt/full-run/s5-evaluation-rico13/vendor-vs-package.json`; `.cache/dlt/full-run/s5-evaluation-rico13/vendor-vs-package-summary.md` |
| Magazine | not run | `not-yet-run (#60)` | not yet run | not yet run | not yet run | Pending vendor support and polygon/train-only handling. |

PubLayNet's validation loss mean is lower by `-0.04%` for seed `42`
(`1.8391` to `1.8383`). Across seeds `42`, `43`, and `44`, FID is higher by
`+4.4%` (`2.3806` to `2.4858`), `overlap_pred` is higher by `+10.1%`
(`0.0241` to `0.0266`), and `alignment_pred` is higher by `+2.6%` (`0.0102`
to `0.0105`). These offsets are reported as the residual of an independent
from-scratch run, not as evidence of a remaining sampling or loss
implementation bug.

### Seed-Variance Control Experiment

The final seed-variance control compares three within-implementation seed
samples against the cross-implementation residual. Vendor samples are
`vendor42`, `vendor43`, and `vendor44`; package samples are `lr-step` seeds
`42`, `45`, and `46`; the cross residual is the reference-callback seed-42
package checkpoint against the vendor seed-42 from-scratch reference. Bit
parity is impossible for this train-ourselves path because the two
implementations do not share a full RNG trajectory.

| Metric | Vendor within-seed variation | Package within-seed variation | Cross residual | Verdict |
| --- | ---: | ---: | ---: | --- |
| `overlap_pred` | `+8.86%` to `+19.55%` | `+0.76%` to `+19.30%` | `+10.10%` | inside both ranges |
| `FID` | `+1.64%` to `+3.89%` | `+17.20%` to `+46.03%` | `+4.40%` | inside package range |
| `alignment_pred` | `-1.86%` to `+4.24%` | `-1.76%` to `+1.05%` | `+2.60%` | inside vendor range |
| `loss_mean` | `-0.04%` to `+0.04%` | `-0.03%` to `-0.01%` | `-0.04%` | inside vendor range |

Conclusion: cross residuals fall inside within-implementation seed variation
for every reported metric, so the remaining package-vs-reference differences
are not statistically distinguishable from same-implementation stochastic
training variation. This establishes practical reproduction, not bit-level
training parity.

PubLayNet per-seed S5 rows:

| Seed | Vendor loss | Ours loss | Loss delta | Vendor FID | Ours FID | FID delta | Vendor overlap | Ours overlap | Overlap delta | Vendor align | Ours align | Align delta | Vendor IoU | Ours IoU | IoU delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `1.8391` | `1.8383` | `-0.0008` | `2.3065` | `2.5502` | `+0.2438` | `0.0245` | `0.0264` | `+0.0020` | `0.0102` | `0.0104` | `+0.0002` | `0.0035` | `0.0036` | `+0.0001` |
| 43 | `1.7927` | `1.7927` | `-0.0000` | `2.4362` | `2.4863` | `+0.0501` | `0.0242` | `0.0269` | `+0.0026` | `0.0102` | `0.0105` | `+0.0003` | `0.0035` | `0.0033` | `-0.0002` |
| 44 | `1.7889` | `1.7896` | `+0.0006` | `2.3993` | `2.4208` | `+0.0216` | `0.0237` | `0.0264` | `+0.0027` | `0.0103` | `0.0105` | `+0.0002` | `0.0034` | `0.0035` | `+0.0001` |

The per-epoch train-loss rows are instantaneous stochastic training-step logs,
not synchronized epoch means. The same-batch train-loss distributions are
close: reference mean `0.8764726606`, package mean `0.8752792128`, and mean
delta `-0.0011934477`, while the per-batch standard deviation is about `0.175`.
The visible sparse train-loss curve difference is therefore a single-batch
logging artifact.

RICO13 residuals are inside the previously recorded PubLayNet seed-variance
reference ranges: `overlap_pred` is within the about `+/-20%` relative range,
FID is within the tens-of-percent package seed variation range, and
`alignment_pred` is within the about `+/-4%` range. Loading the vendor seed-42
weights through the package checkpoint format and running the same S5 evaluator
produced zero deltas for all reported metrics on all three seeds, which clears
the package sampling and evaluator paths.

The PubLayNet HDF5 training data also matches the JSON source by image id for
normalized LTWH boxes, category ids, filtering, and dataset order. The only
observed data-order difference is that the reference data loader applies one
construction-time element shuffle and one per-access shuffle, while
`H5DLTDataset` applies the per-access shuffle. The composition of two uniform
permutations is still a uniform permutation, so this is distribution-equivalent
to the package loader and only prevents bit-level RNG-trajectory parity.

## Regeneration Metadata

The paths below are non-committed local evidence locations and rerun metadata
needed to regenerate the recorded PubLayNet and RICO13 results.

```text
.cache/dlt/reference/
.cache/dlt/converted/
.cache/dlt/full-run/dlt_training_repro_primary_diagnosis.json
.cache/dlt/full-run/ours-publaynet-reference-callback-seed42/checkpoints/final-epoch799.ckpt
.cache/dlt/full-run/ours-publaynet-reference-callback-seed42/csv/csv/version_0/metrics.csv
.cache/dlt/full-run/vendor-publaynet/checkpoints/checkpoint-799
.cache/dlt/full-run/s5-evaluation-reference-callback-seed42/results.json
.cache/dlt/full-run/s5-evaluation-seed-variance/vendor42-vendor43-vendor44-pairwise.json
.cache/dlt/full-run/s5-evaluation-seed-variance/ours-lr-step-seed42-seed45-seed46-pairwise.json
.cache/dlt/full-run/s5-evaluation-seed-variance/ours-lr-step-seed46-vs-vendor42.json
.cache/dlt/full-run/s5-evaluation-seed-variance/final-verdict.json
.cache/dlt/full-run/s5-evaluation-seed-variance/final-verdict.md
.cache/dlt/full-run/s5-evaluation-rico13/vendor-vs-package.json
.cache/dlt/full-run/s5-evaluation-rico13/vendor-vs-package-summary.md
.cache/dlt/full-run/ours-rico13/checkpoints/final-epoch799.ckpt
.cache/dlt/full-run/vendor-rico13/checkpoints/checkpoint-799
```

```text
training_config_sha256:
  models/dlt/configs/training/dlt_publaynet.yaml:
    18bfdfb67fa98eae777a5c6db4383b80cd7bcaf5912e9030f3411f7bed1fbbf8
s5_scripts_sha256:
  .cache/dlt/full-run/scripts/run_s5_publaynet_lr_step.py:
    fd552c8a9261b2295b1838925eaa88900f0544bb598203a75b352314abbc7409
  .cache/dlt/full-run/scripts/evaluate_s5_publaynet.py:
    19815064bab013c5e07b2c03704b8687c30318135075603c8a8d9b6f528c39f8
seeds: [42, 43, 44]
condition: all
batch_size: 64
reference_checkpoint: .cache/dlt/full-run/vendor-publaynet/checkpoints/checkpoint-799
package_checkpoint: .cache/dlt/full-run/ours-publaynet-reference-callback-seed42/checkpoints/final-epoch799.ckpt
package_loss_curve: .cache/dlt/full-run/ours-publaynet-reference-callback-seed42/csv/csv/version_0/metrics.csv
result_json: .cache/dlt/full-run/s5-evaluation-reference-callback-seed42/results.json
diagnosis_json: .cache/dlt/full-run/dlt_training_repro_primary_diagnosis.json
```

```text
run_dir: .cache/dlt/full-run/s5-evaluation-rico13
dataset: rico13
data_prep_valid_box_filter:
  vendor: {train: 19204, val: 1129}
  package: {train: 19204, val: 1129}
vendor_seed: 42
package_seed: 42
sampling_seeds: [42, 43, 44]
condition: all
batch_size: 64
fid_checkpoint: .cache/dlt/fid/layoutnet_rico.pth.tar
reference_checkpoint: .cache/dlt/full-run/vendor-rico13/checkpoints/checkpoint-799
package_checkpoint: .cache/dlt/full-run/ours-rico13/checkpoints/final-epoch799.ckpt
result_json: .cache/dlt/full-run/s5-evaluation-rico13/vendor-vs-package.json
summary_markdown: .cache/dlt/full-run/s5-evaluation-rico13/vendor-vs-package-summary.md
```

## Training Commands

Run the smoke training config.

```bash
uv run --package dlt --extra training \
  traingen fit \
  --config models/dlt/configs/training/smoke.yaml
```

Train PubLayNet.

```bash
uv run --package dlt --extra training \
  traingen fit \
  --config models/dlt/configs/training/dlt_publaynet.yaml
```

Train RICO13.

```bash
uv run --package dlt --extra training \
  traingen fit \
  --config models/dlt/configs/training/dlt_rico13.yaml
```

Run vendor parity checks.

```bash
PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES=<gpu-id> uv run --package dlt --extra vendor \
  pytest models/dlt/tests/vendor_parity -m vendor_parity
```

Regenerate the PubLayNet vendor reference metadata.

```bash
CUDA_VISIBLE_DEVICES=<gpu-id> uv run --package dlt --extra vendor \
  python models/dlt/scripts/generate_vendor_reference.py \
  --config vendor/dlt/dlt/configs/remote/dlt_publaynet_config.py \
  --workdir dlt-publaynet \
  --epoch 799 \
  --condition all \
  --output-metadata .cache/dlt/reference/publaynet-all.json
```

Re-run the accepted PubLayNet S5 evaluation.

```bash
CUDA_VISIBLE_DEVICES=<gpu-id> uv run --package dlt --extra training --extra vendor \
  python .cache/dlt/full-run/scripts/run_s5_publaynet_lr_step.py \
  --ours-checkpoint .cache/dlt/full-run/ours-publaynet-reference-callback-seed42/checkpoints/final-epoch799.ckpt \
  --ours-curve .cache/dlt/full-run/ours-publaynet-reference-callback-seed42/csv/csv/version_0/metrics.csv \
  --output .cache/dlt/full-run/s5-evaluation-reference-callback-seed42/results.json \
  --seeds 42 43 44 \
  --device cuda
```

Re-run the RICO13 S5 evaluation.

```bash
CUDA_VISIBLE_DEVICES=<gpu-id> uv run --package dlt --extra training --extra vendor \
  python .cache/dlt/full-run/scripts/evaluate_s5_rico13.py \
  --ours-checkpoint .cache/dlt/full-run/ours-rico13/checkpoints/final-epoch799.ckpt \
  --output .cache/dlt/full-run/s5-evaluation-rico13/vendor-vs-package.json \
  --seeds 42 43 44 \
  --device cuda
```

Convert and smoke-test a local checkpoint directory.

```bash
uv run --package dlt \
  python models/dlt/scripts/convert_original_checkpoint.py \
  --dataset publaynet \
  --checkpoint-dir .cache/dlt/original/checkpoint-799 \
  --output-dir .cache/dlt/converted/publaynet
uv run --package dlt \
  python models/dlt/scripts/smoke_from_pretrained.py \
  --path .cache/dlt/converted/publaynet
```

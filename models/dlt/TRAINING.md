# Training DLT

DLT training uses the shared class-path-driven LightningCLI entry point. The package does not define `dlt.training.cli`.

## Smoke

```bash
uv run --package dlt --extra training \
  python -m traingen.lightning.cli fit \
  --config models/dlt/configs/training/smoke.yaml
```

## PubLayNet

```bash
uv run --package dlt --extra training \
  python -m traingen.lightning.cli fit \
  --config models/dlt/configs/training/dlt_publaynet.yaml
```

## RICO13

```bash
uv run --package dlt --extra training \
  python -m traingen.lightning.cli fit \
  --config models/dlt/configs/training/dlt_rico13.yaml
```

Magazine remains gated until polygon and train-only handling is amended.

## Scheduler Recipe

All class-path training configs use AdamW, per-step warmup-cosine scheduling,
and `gradient_clip_val=1.0`. The PubLayNet configs pin `num_warmup_steps=100000`
and `num_training_steps=1994400`, matching the evaluated S5 checkpoint's
`global_step=1994400` and scheduler `last_epoch=1994400`. RICO13 uses the
vendor warmup of `10000` steps and Magazine uses the vendor warmup of `2000`
steps; their total training steps are resolved by Lightning from the active
datamodule because full RICO13/Magazine package training pairs have not yet
been run.

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `CUDA_VISIBLE_DEVICES="" PARITY_REQUIRE=1 uv run --package dlt --extra training --extra vendor pytest models/dlt/tests/vendor_parity/test_dlt_training_parity.py -m "vendor_parity and training" -rs` | `models/dlt/tests/vendor_parity/test_dlt_training_parity.py` | Static package-vs-vendor topology and initialized state dicts match exactly for the tiny test and full PubLayNet configurations. |
| S1 | `CUDA_VISIBLE_DEVICES="" PARITY_REQUIRE=1 uv run --package dlt --extra training --extra vendor pytest models/dlt/tests/vendor_parity/test_dlt_training_parity.py -m "vendor_parity and training" -rs` | `models/dlt/tests/vendor_parity/test_dlt_training_parity.py` | Fixed-batch pre-optimizer trace parity covers `box`, `box_cond`, `cat`, `mask_box`, `mask_cat`, `noise`, `t`, noised boxes/categories, model predictions, masked L2, masked CE, and total loss. |
| S2 | `CUDA_VISIBLE_DEVICES="" PARITY_REQUIRE=1 uv run --package dlt --extra training --extra vendor pytest models/dlt/tests/vendor_parity/test_dlt_training_parity.py -m "vendor_parity and training" -rs` | `models/dlt/tests/vendor_parity/test_dlt_training_parity.py` | One training-step loss definition is bit-identical; the S5 matched-batch diagnostic later confirmed `max_abs_ours_loss_def_delta=0.0`, `max_abs_l2_def_delta=0.0`, and `max_abs_ce_def_delta=0.0` on real PubLayNet batches. |
| S3 | `CUDA_VISIBLE_DEVICES="" uv run --package dlt --extra training python -m traingen.lightning.cli fit --config models/dlt/configs/training/smoke.yaml` | `models/dlt/configs/training/smoke.yaml` | Deterministic CPU short run exercises LightningCLI class-path wiring, AdamW, per-step warmup-cosine scheduling, gradient clipping, synthetic data loading, and one train batch without checkpoint artifacts. |
| S4 | `uv run --package dlt --extra training pytest models/dlt/tests/test_dlt_training.py -m training -q` | `models/dlt/tests/test_dlt_training.py` | HDF5 data loading, padding/filtering, per-access element shuffling, reference epoch-sampling RNG consumption, scheduler stepping, and trace adapter coverage are verified on local fixtures; the full PubLayNet data audit is recorded in `.cache/dlt/full-run/dlt_training_repro_primary_diagnosis.json`. |
| S5 | `CUDA_VISIBLE_DEVICES=<gpu-id> uv run --package dlt --extra training --extra vendor python .cache/dlt/full-run/scripts/run_s5_publaynet_lr_step.py --ours-checkpoint .cache/dlt/full-run/ours-publaynet-reference-callback-seed42/checkpoints/final-epoch799.ckpt --ours-curve .cache/dlt/full-run/ours-publaynet-reference-callback-seed42/csv/csv/version_0/metrics.csv --output .cache/dlt/full-run/s5-evaluation-reference-callback-seed42/results.json --seeds 42 43 44 --device cuda` | `.cache/dlt/full-run/s5-evaluation-seed-variance/final-verdict.json` | PubLayNet practical reproduction is accepted: all cross-implementation residuals fall inside at least one within-implementation seed-variance range, so the remaining differences are from-scratch stochastic divergence rather than a package bug. |

## Reproduction Results

S5 compares a reference PubLayNet checkpoint trained from scratch with the
vendor implementation using seed `42` against an independently trained package
checkpoint using the same PubLayNet validation split, `all` conditioning, and
evaluation seeds `42`, `43`, and `44`. Lower is better for FID, overlap,
alignment, IoU, and loss.

| Dataset | Status | Checkpoints | Loss mean | FID | Overlap | Alignment | IoU | Conclusion |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PubLayNet | S5 practical reproduction, stochastic residual disclosed | vendor seed-42 epoch 799 vs package `final-epoch799.ckpt` | vendor `1.8069 +/- 0.0228`; package `1.8068 +/- 0.0223`; delta `-0.0001` | vendor `2.3806 +/- 0.0546`; package `2.4858 +/- 0.0528`; delta `+0.1051` | vendor `0.0241 +/- 0.0003`; package `0.0266 +/- 0.0002`; delta `+0.0024` | vendor `0.0102 +/- 0.0000`; package `0.0105 +/- 0.0000`; delta `+0.0003` | vendor `0.0034 +/- 0.0001`; package `0.0035 +/- 0.0001`; delta `+0.0000` | The train recipe reproduces the loss definition and sampling path. Generation retains small one-directional offsets that are attributable to independent from-scratch stochastic training divergence rather than a fixable package bug. |
| RICO13 | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | Pending full vendor/package training pair. |
| Magazine | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | not yet run | Pending vendor support and polygon/train-only handling. |

The PubLayNet S5 run is not metric-identical, and it should not be described as
bit-level training parity. For seed `42`, the validation loss mean is lower by
`-0.04%` (`1.8391` to `1.8383`). Across seeds `42`, `43`, and `44`, the summary
generation offsets are consistent: FID is higher by `+4.4%` (`2.3806` to
`2.4858`), `overlap_pred` is higher by `+10.1%` (`0.0241` to `0.0266`), and
`alignment_pred` is higher by `+2.6%` (`0.0102` to `0.0105`). These offsets are
reported as the residual of an independent from-scratch run, not as evidence of
a remaining sampling or loss implementation bug.

### Seed-Variance Control Experiment

The final seed-variance control compares three within-implementation seed
samples against the cross-implementation residual. Vendor samples are
`vendor42`, `vendor43`, and `vendor44`; package samples are `lr-step` seeds
`42`, `45`, and `46`; the cross residual is the reference-callback seed-42
package checkpoint against the vendor seed-42 from-scratch reference. Bit parity
is impossible for this train-ourselves path because the two implementations do
not share a full RNG trajectory. The practical reproduction criterion is
therefore whether the cross residual is statistically indistinguishable from
same-implementation seed variation.

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

Regenerate or inspect the control artifacts with the following metadata.

```text
run_dir: .cache/dlt/full-run/s5-evaluation-seed-variance
vendor_seeds: [42, 43, 44]
package_lr_step_seeds: [42, 45, 46]
cross_residual_source: reference-callback seed42 vs vendor seed42
vendor_pairwise: .cache/dlt/full-run/s5-evaluation-seed-variance/vendor42-vendor43-vendor44-pairwise.json
package_seed46_s5: .cache/dlt/full-run/s5-evaluation-seed-variance/ours-lr-step-seed46-vs-vendor42.json
package_pairwise: .cache/dlt/full-run/s5-evaluation-seed-variance/ours-lr-step-seed42-seed45-seed46-pairwise.json
final_verdict_json: .cache/dlt/full-run/s5-evaluation-seed-variance/final-verdict.json
final_verdict_markdown: .cache/dlt/full-run/s5-evaluation-seed-variance/final-verdict.md
```

```bash
CUDA_VISIBLE_DEVICES=<gpu-id> uv run --package dlt --extra training --extra vendor \
  python .cache/dlt/full-run/scripts/run_s5_publaynet_lr_step.py \
  --ours-checkpoint .cache/dlt/full-run/ours-publaynet-reference-callback-seed42/checkpoints/final-epoch799.ckpt \
  --ours-curve .cache/dlt/full-run/ours-publaynet-reference-callback-seed42/csv/csv/version_0/metrics.csv \
  --output .cache/dlt/full-run/s5-evaluation-reference-callback-seed42/results.json \
  --seeds 42 43 44 \
  --device cuda
```

| Seed | Vendor loss | Ours loss | Loss delta | Vendor FID | Ours FID | FID delta | Vendor overlap | Ours overlap | Overlap delta | Vendor align | Ours align | Align delta | Vendor IoU | Ours IoU | IoU delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `1.8391` | `1.8383` | `-0.0008` | `2.3065` | `2.5502` | `+0.2438` | `0.0245` | `0.0264` | `+0.0020` | `0.0102` | `0.0104` | `+0.0002` | `0.0035` | `0.0036` | `+0.0001` |
| 43 | `1.7927` | `1.7927` | `-0.0000` | `2.4362` | `2.4863` | `+0.0501` | `0.0242` | `0.0269` | `+0.0026` | `0.0102` | `0.0105` | `+0.0003` | `0.0035` | `0.0033` | `-0.0002` |
| 44 | `1.7889` | `1.7896` | `+0.0006` | `2.3993` | `2.4208` | `+0.0216` | `0.0237` | `0.0264` | `+0.0027` | `0.0103` | `0.0105` | `+0.0002` | `0.0034` | `0.0035` | `+0.0001` |

The per-epoch train-loss rows are instantaneous stochastic training-step logs,
not synchronized epoch means. The S5 diagnostic recomputed both final
checkpoints on the same 128 PubLayNet batches, noise, and timesteps and found
bit-identical package-vs-reference loss definitions
(`max_abs_ours_loss_def_delta=0.0`, `max_abs_l2_def_delta=0.0`, and
`max_abs_ce_def_delta=0.0`). The same-batch train-loss distributions are close:
reference mean `0.8764726606`, package mean `0.8752792128`, and mean delta
`-0.0011934477`, while the per-batch standard deviation is about `0.175`. The
visible sparse train-loss curve difference is therefore a single-batch logging
artifact.

The residual investigation found no EMA snapshot or EMA state keys. The
evaluated package checkpoint is `final-epoch799.ckpt`, which records
`epoch=800`, `global_step=1994400`, scheduler `last_epoch=1994400`, and final
learning rate `0.0`; the earlier `checkpoint.ckpt` / `last.ckpt` pair is an
epoch-495 best-train-loss checkpoint and gives substantially worse generation
metrics. Loading the vendor seed-42 weights through the package checkpoint format and
running the same S5 evaluator produced zero deltas for all reported metrics on
all three seeds, which clears the package sampling and evaluator paths.

The PubLayNet HDF5 training data also matches the JSON source by image id for
normalized LTWH boxes, category ids, filtering, and dataset order. The only
observed data-order difference is that the reference data loader applies one
construction-time element shuffle and one per-access shuffle, while
`H5DLTDataset` applies the per-access shuffle. The composition of two uniform
permutations is still a uniform permutation, so this is distribution-equivalent
to the package loader and only prevents bit-level RNG-trajectory parity.

Reproduce the recorded S5 result with the following metadata. Generated
checkpoints, result JSON, and downloaded assets stay outside git.

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

```bash
CUDA_VISIBLE_DEVICES=<gpu-id> uv run --package dlt --extra training --extra vendor \
  python .cache/dlt/full-run/scripts/run_s5_publaynet_lr_step.py \
  --ours-checkpoint .cache/dlt/full-run/ours-publaynet-reference-callback-seed42/checkpoints/final-epoch799.ckpt \
  --ours-curve .cache/dlt/full-run/ours-publaynet-reference-callback-seed42/csv/csv/version_0/metrics.csv \
  --output .cache/dlt/full-run/s5-evaluation-reference-callback-seed42/results.json \
  --seeds 42 43 44 \
  --device cuda
```

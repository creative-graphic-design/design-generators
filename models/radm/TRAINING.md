---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - RADM
---

# RADM Training

## Result at a glance

The CGL S5 endpoint comparison is complete for paired training seeds 1, 2, and 3, but the maintainer verdict is pending: package and original terminal checkpoints were evaluated through the same released metric chain, and the reported distributions differ for `R_ove`, `R_und`, and `R_com` while `R_occ` overlaps; this document makes no equivalence claim.

The proposed verdict is `recipe-unstable (documented)`: the package completed the approved terminal training recipe and the comparison is reproducible and fully recorded, but the three-seed sample does not support claiming practical reproduction of the original metric distribution.

S0 topology evidence records that text encoding fields are derived from the selected mapper; class-mapping and derived-output guards passed.

Verdict: [VERDICT PENDING MAINTAINER]

## Install

Run commands from the repository root and keep generated data, logs, checkpoints, converted pipelines, and evaluation artifacts under `.cache/radm/`.

```bash
uv sync --package radm --extra training
```

Install the `vendor` extra only for original-code parity checks.

```bash
uv sync --package radm --extra training --extra vendor
```

## Data

The approved CGL archive is materialized under `.cache/radm/data/cgl/` with source annotations, images, text features, and text-feature tensors; the training and evaluation runs used the source five-label order and the package's recorded four-output-class configuration.

| Dataset | Source                                                                                                                                      | Config or path                                                              |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| CGL     | `creative-graphic-design/CGL-Dataset` approved through [issue 261](https://github.com/creative-graphic-design/design-generators/issues/261) | `.cache/radm/data/cgl/`                                                     |
| CGL-v2  | approved source distribution                                                                                                                | `.cache/radm/data/cgl/` and `models/radm/configs/training/radm_cgl_v2.yaml` |

The CGL test annotation file contains 1,035 entries whose first 500 boxes contain one distinct tuple, `[1, 2, 3, 4]`, and all 1,035 test boxes use that tuple; the package validation stream loads this file through `RADMDataModule.val_dataloader()` and the original `layout_val` registration points to the same file, so validation loss is a placeholder-target diagnostic and is not a checkpoint-quality criterion.

## Configs

| Config                       | Dataset        | Seed mode               | Purpose                                                           |
| ---------------------------- | -------------- | ----------------------- | ----------------------------------------------------------------- |
| `effective_radm_config.yaml` | CGL and CGL-v2 | captured static state   | Effective model, optimizer, scheduler, sampler, and input record. |
| `radm_cgl.yaml`              | CGL            | package training recipe | Full CGL training configuration.                                  |
| `radm_cgl_v2.yaml`           | CGL-v2         | package default         | Future CGL-v2 training configuration; not included in S5.         |
| `radm_s0_deterministic.yaml` | CGL fixture    | parity seed             | S0 and initialized-state wiring.                                  |
| `radm_smoke.yaml`            | CGL fixture    | deterministic           | Package-local smoke and configuration checks.                     |

## Scheduler and Recipe Notes

The effective recipe uses AdamW with learning rate `2.5e-5`, weight decay `1e-4`, batch size 16, full-model gradient clipping at 1.0, warmup factor 0.01 for 1,000 optimizer steps, milestones 150,000 and 220,000, maximum 250,000 optimizer steps, 1,000 diffusion timesteps, `SNR_SCALE=2.0`, `SAMPLE_STEP=1`, 100 proposals, six repeated heads, and FP32 arithmetic.

The S5 package launcher set `trainer.deterministic=false` to match the original runtime's non-deterministic CUDA algorithm policy; this is an S5-only launcher setting and does not change the S0-S4 parity configuration or model, optimizer, data, or schedule.

Scheduler cadence is applicable and was checked at optimizer-step cadence in S2 and the training runs; the timestep sampler is applicable and covered by the S1-S3 traces; EMA is not applicable because this loop has no EMA state; AMP is not applicable because the claimed runs use FP32; multi-worker loader rules are not applicable because `num_workers=0` was used.

The amended lockstep rule for discrete assignment models is recorded in [the issue 261 amendment](https://github.com/creative-graphic-design/design-generators/issues/261#issuecomment-5318097501): initial state, RNG, batch order, step-1 forward/loss, and step-1 gradient absolute tolerance are the pre-S5 gate, while multi-seed distribution comparison is the endpoint because ULP-scale backward accumulation can cross dynamic-matching boundaries after the amplification horizon.

The validation-loss diagnostic found dummy test boxes, so the earlier monitored best-validation records are not quality evidence; S5 uses terminal checkpoints selected by exact global-step identity instead.

## Seed Policy

S0-S4 parity fixtures use parity seed 261 and are not training-seed or evaluation-seed results. S5 uses paired training seeds 1, 2, and 3 for original and package systems, and evaluation uses one inference step for each corresponding checkpoint with evaluator seeds 1, 2, and 3 as recorded by the evaluation chain.

## Validation Stages

| Stage | Scope                                           | Purpose                                                                                                                    |
| ----- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity      | Topology, state coverage, optimizer, scheduler, sampler, class mapping, and initial-state agreement.                       |
| S1    | Fixed-batch pre-optimizer trace parity          | Inputs, RNG-dependent values, outputs, losses, and total loss.                                                             |
| S2    | One optimizer-step parity                       | Gradients, clipping, optimizer state, post-step parameters, scheduler state, and learning rate.                            |
| S3    | Deterministic multi-batch and production wiring | Natural trajectory diagnostic, synchronized bounded diagnostic, loader behavior, logging, scheduler, and checkpoint hooks. |
| S4    | Deterministic loader stream                     | Sample order, transforms, masks, padding, class ids, text features, and validation-stream behavior.                        |
| S5    | Full-run statistical comparison                 | Three paired training seeds, terminal checkpoints, original evaluator chain, per-seed hashes, and distribution analysis.   |

## Stage Evidence

| Stage | Command                                                                                                                                                                                                                                                                                                                                                                                                | Artifact                                                                  | Result                                                                                                                                                                                                                          |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `CUDA_VISIBLE_DEVICES=<gpu> PARITY_REQUIRE=1 RADM_REFERENCE_DEVICE=cuda:0 .cache/radm/reference-env/bin/python -m pytest models/radm/tests/test_training_s0_contracts.py models/radm/tests/vendor_parity/test_s0_reference_adapter.py models/radm/tests/vendor_parity/test_s0_radm_topology.py -m 'not integration' -q -rs`                                                                            | `models/radm/configs/training/effective_radm_config.yaml`                 | Accepted; 23 passed, 0 skipped; vendor revision `413f87a45760ceac5635b6a08c8047f86478acf5`; config SHA-256 `308139a77dc29df6d91d909565b48a026cf0b3f8a965ac9c315d7bb61291002f`.                                                  |
| S1    | `CUDA_VISIBLE_DEVICES=<gpu> PARITY_REQUIRE=1 RADM_S1_EVIDENCE_PATH=.cache/radm/supported-v100/s3-wiring-20260814/s1_fixed_batch_trace.json .cache/radm/reference-env/bin/python -m pytest models/radm/tests/vendor_parity/test_s1_radm_training.py::test_s1_radm_fixed_batch_pre_optimizer_parity -s -q -rs`                                                                                           | `.cache/radm/supported-v100/s3-wiring-20260814/s1_fixed_batch_trace.json` | Accepted; first divergence `none`, max absolute error `6.103515625e-05`, max relative error `7.620204911518158e-08`; SHA-256 `8f27ccd99b90fac3f1ed3e922308df7c8013197c7b05799e32f2055a7c66b63a`.                                |
| S2    | `CUDA_VISIBLE_DEVICES=<gpu> PARITY_REQUIRE=1 RADM_S1_EVIDENCE_PATH=.cache/radm/supported-v100/s3-wiring-20260814/s1_fixed_batch_trace.json RADM_S2_EVIDENCE_PATH=.cache/radm/supported-v100/s3-wiring-20260814/s2_one_step_trace.json .cache/radm/reference-env/bin/python -m pytest models/radm/tests/vendor_parity/test_s1_radm_training.py::test_s2_radm_one_optimizer_step_parity -s -q -rs`       | `.cache/radm/supported-v100/s3-wiring-20260814/s2_one_step_trace.json`    | Accepted under the recorded S2 tolerance; first divergence `none`, max absolute error `1.9073486328125e-06`; SHA-256 `0af1cbd069e43f4e0005f8d728e569dd1b77e6800a2ea23bc1e750d974518181`.                                        |
| S3    | `CUDA_VISIBLE_DEVICES=<gpu> PARITY_REQUIRE=1 .cache/radm/reference-env/bin/python -m pytest models/radm/tests/vendor_parity/test_s3_radm_wiring.py -s -q -rs` plus the recorded synchronized and natural traces                                                                                                                                                                                        | `.cache/radm/s3/two-layer-20260814-rerun.json`                            | Synchronized diagnostic passed; natural trajectory drift was recorded at step 2; production wiring passed with two optimizer steps and exit code 0; SHA-256 `b4498680c80c04eb193e7166e79553fa8d9ab2e93120b039e4ba291f9b60677d`. |
| S4    | `CUDA_VISIBLE_DEVICES=<gpu> PARITY_REQUIRE=1 RADM_S4_DATA_ROOT=.cache/radm/data/cgl RADM_S4_EVIDENCE_PATH=.cache/radm/s4/run-012-image-box-resampling.json .cache/radm/reference-env/bin/python -m pytest models/radm/tests/vendor_parity/test_s4_radm_loader_stream.py -s -q -rs`                                                                                                                     | `.cache/radm/s4/run-012-image-box-resampling.json`                        | Accepted; first divergence `none` for loader order, labels, fallback, images, normalized boxes, masks, and text features; SHA-256 `304892b296c24f7bec86e92ffcc4b6ab380ab62bfc9a7ce70dc414b3a9c59415`.                           |
| S5    | `CUDA_VISIBLE_DEVICES=<gpu> PARITY_REQUIRE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 .cache/radm/reference-env/bin/python -u -m traingen.lightning.cli fit --config models/radm/configs/training/radm_cgl.yaml --seed_everything=<seed> --trainer.deterministic=false --trainer.max_steps=250000 --trainer.limit_val_batches=0` and the source evaluator command in [Comparison Scope](#comparison-scope) | `.cache/radm/full-run/cgl/manifest.json`                                  | Complete for CGL training-seed n=3 per system; manifest SHA-256 `b14de5657eb82b0b4643def5f0e7b4119e60f78a4889fa63bfccf55a675b9db4`; maintainer verdict pending.                                                                 |

## Reproduction Results

The terminal-vs-terminal CGL comparison is complete for three original and three package training seeds, with full-precision values retained in `.cache/radm/full-run/cgl/comparison.json`; statuses remain `blocked (maintainer verdict pending)` until the maintainer decides whether the observed three-seed distribution supports the proposed recipe-unstable conclusion.

| Dataset | System   | Status                                       | Seed scope        | Primary metrics                      | Loss evidence                                                                | Artifact summary                                       |
| ------- | -------- | -------------------------------------------- | ----------------- | ------------------------------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------ |
| CGL     | original | `blocked (maintainer verdict pending)`       | training-seed n=3 | See terminal comparison table below. | 300-step amended preflight passed; this is not a full-run loss-parity claim. | `.cache/radm/full-run/cgl/original-seed-{1,2,3}-eval/` |
| CGL     | package  | `blocked (maintainer verdict pending)`       | training-seed n=3 | See terminal comparison table below. | 300-step amended preflight passed; this is not a full-run loss-parity claim. | `.cache/radm/full-run/cgl/package-seed-{1,2,3}-eval/`  |
| CGL-v2  | original | `blocked (CGL-v2 outside approved S5 scope)` | not run           | not measured                         | outside approved scope                                                       | `.cache/radm/data/cgl/`                                |
| CGL-v2  | package  | `blocked (CGL-v2 outside approved S5 scope)` | not run           | not measured                         | outside approved scope                                                       | `models/radm/configs/training/radm_cgl_v2.yaml`        |

### Comparison Scope

| Dataset | System | Evaluator                                                                                                                                                                  | Test split                          | Checkpoint-selection rule                                                                                                                                                                                                      | Sample count                                                                           |
| ------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| CGL     | both   | Released source `metrics.py` chain invoked through source `train_net.py --eval-only --resume` semantics; package predictions are passed through the same metric functions. | CGL test split                      | Original: matching terminal `model_final.pth` at source `final_iter=249999/max_iter=250000`; package: `terminal-step-250000.ckpt` guarded by global step, training seed, config path, config SHA-256, and checkpoint identity. | 1,035 layouts per evaluation seed; one inference step; class threshold 0.25; NMS 0.15. |
| CGL-v2  | both   | Not run; outside approved S5 scope.                                                                                                                                        | Not run; outside approved S5 scope. | Not run; outside approved S5 scope.                                                                                                                                                                                            | Not run; outside approved S5 scope.                                                    |

The exact original evaluator invocation was `CUDA_VISIBLE_DEVICES=<gpu> PARITY_REQUIRE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 .cache/radm/reference-env/bin/python -u -c 'import sys; sys.path.insert(0, "vendor/radm"); import pdb; pdb.set_trace = lambda: None; from PIL import Image; Image.LINEAR = Image.BILINEAR; import runpy; runpy.run_path("vendor/radm/train_net.py", run_name="__main__")' --num-gpus 1 --config-file vendor/radm/configs/radm.yaml --eval-only --resume MODEL.WEIGHTS .cache/radm/s5/cgl/original-seed-<seed>/model_final.pth DATASETS.DATASET_PATH .cache/radm/data/cgl DATASETS.TEXT_FEATURE_PATH .cache/radm/data/cgl/text_features OUTPUT_DIR .cache/radm/full-run/cgl/original-seed-<seed>-eval`; the package driver consumed the guarded terminal checkpoint and wrote the same vendor metric outputs.

`R_shm` and `R_sub` are unavailable because the released `vendor/radm/metrics.py` does not implement them. The evaluator deviations are process-local `PIL.Image.LINEAR = PIL.Image.BILINEAR` and a process-local no-op for the unconditional `pdb.set_trace()` at `vendor/radm/train_net.py:248`; vendor files were not modified.

### Terminal comparison

The table retains full parsed precision and reports arithmetic mean ± sample standard deviation; reader-facing rounding is not applied elsewhere in this document.

| Metric  | Original mean ± SD                            | Package mean ± SD                           | Original range                               | Package range                               | Ranges overlap | Welch two-sided p      | Exact permutation p | Standardized difference (Cohen d, package − original) |
| ------- | --------------------------------------------- | ------------------------------------------- | -------------------------------------------- | ------------------------------------------- | -------------- | ---------------------- | ------------------- | ----------------------------------------------------- |
| `R_ove` | 0.0860741497809693 ± 0.0018684487379379618    | 0.04287902026715906 ± 0.001805857711955914  | [0.08397312277475291, 0.08754935708632096]   | [0.04090649830509365, 0.04445093465056203]  | no             | 0.00000875327923411577 | 0.1                 | -23.508580027141292                                   |
| `R_ali` | 0.0061818007616413345 ± 0.0005137019247577399 | 0.016518116846165398 ± 0.005042923865792919 | [0.005604343830867542, 0.006587991081797301] | [0.012587615098168175, 0.02220418055110927] | no             | 0.0695454122574498     | 0.1                 | 2.883744075571025                                     |
| `R_und` | 0.9763129462304504 ± 0.0022676962396016814    | 0.8975106515078382 ± 0.005937465973311059   | [0.9748172696010776, 0.9789221442095661]     | [0.8928365004569013, 0.9041914277767966]    | no             | 0.0005696806254884086  | 0.1                 | -17.53415969610047                                    |
| `R_occ` | 0.9990338164251208 ± 0.0                      | 0.9993558776167472 ± 0.0005578263470431064  | [0.9990338164251208, 0.9990338164251208]     | [0.9990338164251208, 1.0]                   | yes            | 0.4226497308104187     | 1.0                 | 0.8164965809276322                                    |
| `R_com` | 7.881973584493001 ± 0.14170089554296408       | 14.90238889058431 ± 0.7762038899255534      | [7.725506782531738, 8.001652717590332]       | [14.026309967041016, 15.504308700561523]    | no             | 0.0031771020688053948  | 0.1                 | 12.582969434187925                                    |

The analysis script is `.cache/radm/full-run/cgl/analyze_comparison.py`; it runs both-direction range containment and overlap, two-sided unequal-variance Welch tests, and the exact two-sided relabeling test over all `C(6,3)=20` assignments using `Fraction(str(v))`. The Welch and permutation p-values are exploratory because [issue 261](https://github.com/creative-graphic-design/design-generators/issues/261) prespecifies no test or threshold; they cover only these seeds and do not establish equivalence or non-equivalence, and methods emit undefined values when a zero-variance denominator makes a statistic undefined.

No released full RADM checkpoint was available for a reference-only row; `R-50.pkl` is backbone initialization only and is excluded.

## Per-seed ledger

The original runs were launched from package commit `e93df3609d875d77a7326f3e0ba149ac8d755df4` with vendor submodule revision `413f87a45760ceac5635b6a08c8047f86478acf5`; the package terminal retrains were launched from commit `673ae81a8531cd01ea6957fd3246a086cb8b23dc`, reconstructed exactly from the branch reflog entry at 2026-09-25 14:56:23 JST and the three launch timestamps because the original per-run manifests omitted the commit. The 300-step lockstep probe passed the amended preflight: initial state, RNG, and batch order were exact for 300 records, step-1 forward/loss was exact apart from `1.1920928955078125e-07` GIoU error, and step-1 maximum gradient absolute error was `8.106231689453125e-06` within S2 `atol=5e-5`; the natural trajectory's documented ULP-seeded discrete jumps at steps 7 and 9 are why this is not a 300-step loss-equality claim.

| Seed | Original checkpoint SHA-256                                        | Package terminal checkpoint SHA-256                                | Original evaluator output SHA-256                                  | Package metric record SHA-256                                      | Source package commit                                                                   | Vendor revision                            |
| ---- | ------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------- | ------------------------------------------ |
| 1    | `55aca7804d6e3472bbfb913867865f556481872de9fb5db63ac9ed46fa2d523d` | `70ca285adad1f880d72b6f67e2dd18377a177f83be7194e679cd94e39754100a` | `830e66c30c82d36c46a890387135ad51548274c0f9fb7e4871d4dea80af6ae3d` | `7785d9a4e9988bf31ae0022b23215630edc50a19fb578a1f1b7e48edc03679fc` | `e93df3609d875d77a7326f3e0ba149ac8d755df4` / `673ae81a8531cd01ea6957fd3246a086cb8b23dc` | `413f87a45760ceac5635b6a08c8047f86478acf5` |
| 2    | `b1e54b8a0e4bc48c92d93a8182e5f3f99c50b23bd9610112f54b58633e9a3fb9` | `c3e99b199d048eace58f6e0a2cadd568623739955044e61a9054eef0b0a1f918` | `5ab124d9f38935134889d2d75a2f593fd67e1b57f6c09b3b8b437e556d9c0990` | `864fd1bff45797779115c237c4cd21fb3fff52d6378e677c955fd315d67a8476` | `e93df3609d875d77a7326f3e0ba149ac8d755df4` / `673ae81a8531cd01ea6957fd3246a086cb8b23dc` | `413f87a45760ceac5635b6a08c8047f86478acf5` |
| 3    | `69a0a22e0d972541ffefa6557b67ea49f939edaab40f7cce8d5225bf13ccd334` | `4ef19c0454ea122a2a4a22efb0ed01e9d64d3209c8cff5ef4246d4ae7a6bbe74` | `ab7763980728a9545ced01ef1903a230d49af6e9d70d419ffd17a968b9f169df` | `54cbcb24982901a8b15f43917a0875306ec788fd6aab8d8d8e5125f8602ca76e` | `e93df3609d875d77a7326f3e0ba149ac8d755df4` / `673ae81a8531cd01ea6957fd3246a086cb8b23dc` | `413f87a45760ceac5635b6a08c8047f86478acf5` |

The original evaluator output column hashes each seed's source `coco_instances_results.json`, and the package metric record column hashes each seed's package `vendor-metrics.json`; the authoritative raw prediction and evaluation output hashes are in `.cache/radm/full-run/cgl/manifest.json`.

## Regeneration Metadata

The aggregate launch and evaluation manifest is `.cache/radm/full-run/cgl/manifest.json` with SHA-256 `b14de5657eb82b0b4643def5f0e7b4119e60f78a4889fa63bfccf55a675b9db4`; the comparison is `.cache/radm/full-run/cgl/comparison.json` with SHA-256 `207d083701be70514f7daa85178c6515c80961e1002d0d83939c69dca73a4129`; package config SHA-256 is `aad0dcc6e93bae66ccefd04562bcc8a6031e343f7d4322d6dfa63cc85ba4343e`; source config SHA-256 is `66757da8054edf7a491a5147ba65acc63fe7898f6fb6a70d97f7fd4b507a0856`; vendor metric code SHA-256 is `01447490d9f196638a9a578dbf2a1a007be1e7b0296a92b8370328802616042f`; test annotation SHA-256 is `e131eec9130a97845096604d894b618afb20c24589e0e55c16298218c4d4b909`.

The S0-S4 static evidence and the full-run evidence remain cache-relative and are not committed as tensors, images, datasets, or checkpoints. Every S5 manifest records source commit, resolved config, evaluator command, checkpoint rule, artifact SHA-256 mappings, and launch time; the package training manifests were augmented after the run with the exact reflog-based provenance reconstruction described above.

## Training Commands

Run the staged parity checks.

```bash
PARITY_REQUIRE=1 uv run --package radm --extra training --extra vendor pytest models/radm/tests/test_training_s0_contracts.py models/radm/tests/vendor_parity/test_s0_reference_adapter.py models/radm/tests/vendor_parity/test_s0_radm_topology.py -m 'not integration' -q -rs
```

Run a package training seed with the S5 terminal-checkpoint policy.

```bash
CUDA_VISIBLE_DEVICES=<gpu> PARITY_REQUIRE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 uv run --package radm --extra training traingen fit --config models/radm/configs/training/radm_cgl.yaml --seed_everything=<seed> --trainer.deterministic=false --trainer.max_steps=250000 --trainer.limit_val_batches=0
```

The terminal evaluator must reject any checkpoint whose global step is not 250,000 or whose seed, resolved config path, config SHA-256, or identity metadata does not match the launch manifest; evaluation is not permitted before this guard passes.

## References and deviations

The design and amendments are recorded in [issue 261](https://github.com/creative-graphic-design/design-generators/issues/261), [the amended lockstep comment](https://github.com/creative-graphic-design/design-generators/issues/261#issuecomment-5318097501), and [the later validation-target diagnostic](https://github.com/creative-graphic-design/design-generators/issues/261#issuecomment-5933760418).

The campaign-local guard repair accepted the supervisor's `training_seeds` and `jobs` manifest shape and the per-run `package_config` identity while retaining exact checkpoint, seed, and config checks; it did not weaken fail-closed behavior. The aggregate comparison records this deviation and the process-local evaluator aliases, while vendor source files remain unchanged.

The S5 claim applies only to CGL, training seeds 1, 2, and 3, the 1,035-image test split, the five available vendor metrics, and the stated terminal checkpoint rules. It does not claim CGL-v2, `R_shm`, `R_sub`, equivalence, non-equivalence, or generalization to untested seeds.

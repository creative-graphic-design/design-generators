---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - RADM
---

# RADM Training

This document records package-local training checks for RADM. It is for contributors who need to reproduce the checks or decide whether a full training run may be claimed.

## Result at a glance

S0-S4 package infrastructure and the evaluation-path parity prerequisite are recorded. The CGL S5 result is `blocked (corrected training stream not retrained; S5 retrain pending)`: the retained package checkpoints came from an invalid training stream, and this PR does not retrain them. CGL-v2 is outside the approved S5 scope. No S5 reproduction or vendor-parity verification claim is made.

The regenerated diagnosis compares the six retained checkpoints with the fixed equivalent evaluator on the invalid-stream runs. The reproducible directional shift is in `R_ove`, `R_und`, and prediction count. `R_ali` ranges overlap, as do `R_occ` ranges; `R_com` differs in these runs but is not called a systematic shift. The diagnosis does not establish equivalence or non-equivalence.

The evaluation-path parity prerequisite passes for one shared original seed-1 checkpoint. It uses stated prediction and metric tolerances and reports the vendor's one out-of-bounds prediction separately from the package's zero under the stated rule. This is a path check, not full training parity.

## Install

Run commands from the repository root. Keep generated data, logs, checkpoints, converted local pipelines, and evaluation artifacts under `.cache/radm/`.

```bash
uv sync --package radm --extra training
```

Install the `vendor` extra only for original-code parity checks.

```bash
uv sync --package radm --extra training --extra vendor
```

## Data

The approved CGL archive is materialized under `.cache/radm/data/cgl/` with source annotations, images, text features, and text-feature tensors. The package and original-code runs use the source five-label order and the released four-output-class configuration.

| Dataset | Source                                                                                                                                       | Config or path                                                              |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| CGL     | `creative-graphic-design/CGL-Dataset`, approved through [issue 261](https://github.com/creative-graphic-design/design-generators/issues/261) | `.cache/radm/data/cgl/`                                                     |
| CGL-v2  | CGL-Dataset-v2 source distribution, not included in the approved S5 scope                                                                    | `.cache/radm/data/cgl/` and `models/radm/configs/training/radm_cgl_v2.yaml` |

## Configs

| Config                       | Dataset        | Seed mode               | Purpose                                                           |
| ---------------------------- | -------------- | ----------------------- | ----------------------------------------------------------------- |
| `effective_radm_config.yaml` | CGL and CGL-v2 | captured static state   | Effective model, optimizer, scheduler, sampler, and input record. |
| `radm_cgl.yaml`              | CGL            | package training recipe | Full CGL training configuration.                                  |
| `radm_cgl_v2.yaml`           | CGL-v2         | package default         | Future CGL-v2 configuration; no S5 result is claimed.             |
| `radm_s0_deterministic.yaml` | CGL fixture    | parity seed             | Initialized-state wiring.                                         |
| `radm_smoke.yaml`            | CGL fixture    | deterministic           | Package-local smoke and configuration checks.                     |

## Scheduler and Recipe Notes

The effective recipe uses AdamW with learning rate `2.5e-5`, weight decay `1e-4`, batch size `16`, full-model gradient clipping at `1.0`, warmup factor `0.01` for `1,000` optimizer steps, milestones `150,000` and `220,000`, maximum `250,000` optimizer steps, `1,000` diffusion timesteps, `SNR_SCALE=2.0`, `SAMPLE_STEP=1`, `100` proposals, six repeated heads, and FP32 arithmetic.

The scheduler is checked at optimizer-step cadence. The timestep sampler is checked in the fixed-batch and multi-batch traces. EMA is not applicable because this loop has no EMA state. AMP is not applicable because the recorded runs use FP32. Multi-worker loader rules are not applicable because the approved parity configuration uses `num_workers=0`.

The S0 topology evidence records that text encoding fields are derived from the selected mapper, with class-mapping and derived-output guards active; the retained run reports `23 passed` against vendor revision `413f87a45760ceac5635b6a08c8047f86478acf5`.

The issue 261 amendment defines the pre-S5 lockstep gate for this model: initial state, RNG and batch alignment, step-1 forward and loss surfaces, and the recorded step-1 gradient tolerance must pass. Natural long-run drift remains report-only when it is compared with self-repeat behavior. The amendment is [issue 261 comment 5318097501](https://github.com/creative-graphic-design/design-generators/issues/261#issuecomment-5318097501).

### Before any S5 launch

Before a future S5 launch, complete these checks and retain their outputs:

- Run the corrected full-stream S4 check with the vendor's seeded sampler, aspect-ratio grouping, worker count, and full-batch policy.
- Run the vendor-mapper oracle and confirm target coordinates use vendor truncation.
- Rerun evaluation-path parity on the launch weights with the vendor evaluator and the package evaluator, including prediction and metric tolerances and the out-of-bounds rule.
- Repeat the 300-step natural and synchronized diagnostics after every training-path change. The package is run-to-run deterministic, while the vendor is not; this does not establish cross-system parity.
- Decide whether periodic vendor evaluations that consume the vendor RNG stream are enabled, and record the choice before launch.
- Write the launch manifest before launching the seed queue. The manifest must identify the source commit, resolved configuration, evaluator command, checkpoint rule, artifact hashes, and launch time.
- Confirm that the launch queue stops when its source-commit gate fails in a deliberate mismatch dry run.

The current PR does not launch another S5 queue or retrain a seed.

## Seed Policy

S0-S4 fixtures use the recorded parity seed and are not training-seed or evaluation-seed results. The retained CGL diagnosis covers paired training labels `1`, `2`, and `3` per system; it is invalid-stream evidence, not an S5 result. The shared-weight evaluation-path check uses evaluation seed `1`.

## Validation Stages

| Stage | Scope                                                                                                                     | Purpose                                                                                                                                         |
| ----- | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static configuration and initialized package/reference state                                                              | Asserted gate for topology, state coverage, optimizer, scheduler, sampler, class mapping, and initial-state agreement.                          |
| S1    | One fixed batch and all recorded pre-optimizer trace surfaces                                                             | Asserted gate for prepared inputs, RNG-dependent values, outputs, loss components, and total loss.                                              |
| S2    | One optimizer step over the recorded gradients, optimizer state, parameters, and scheduler state                          | Asserted gate under the recorded gradient tolerance.                                                                                            |
| S3    | `300` natural cross-system steps plus two `300`-step package self-repeat files and one `300`-step vendor self-repeat file | The pre-S5 surfaces are an asserted gate. Natural drift, self-repeat envelopes, and the maximum-statistic classification are report-only.       |
| S4    | Mapper oracle, test-stream checks, and `3,784` full training batches containing `60,544` images                           | Mapper oracle is an asserted gate and passes; the full-stream comparison is failed/unavailable at batch `1`, so full-stream S4 is not accepted. |
| S5    | Three retained package and three retained original terminal checkpoints on the CGL test split                             | Blocked because the retained package training stream was corrected after launch and the user-approved scope does not retrain.                   |

## Stage Evidence

| Stage | Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Artifact                                                                                                               | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0    | `PARITY_REQUIRE=1 uv run --package radm --extra training --extra vendor pytest models/radm/tests/test_training_s0_contracts.py models/radm/tests/vendor_parity/test_s0_reference_adapter.py models/radm/tests/vendor_parity/test_s0_radm_topology.py -m 'not integration' -q -rs`                                                                                                                                                                                                                                                                                                   | `models/radm/configs/training/effective_radm_config.yaml`                                                              | Asserted gate accepted by the retained S0 evidence; the effective config and vendor revision are recorded in the prior stage artifact.                                                                                                                                                                                                                                                                                                                                                                                                     |
| S1    | `PARITY_REQUIRE=1 uv run --package radm --extra training --extra vendor pytest models/radm/tests/vendor_parity/test_s1_radm_training.py::test_s1_radm_fixed_batch_pre_optimizer_parity -s -q -rs`                                                                                                                                                                                                                                                                                                                                                                                   | `.cache/radm/supported-v100/s3-wiring-20260814/s1_fixed_batch_trace.json`                                              | Asserted gate over one source-generated batch (one 64x64 RGB sample, two labeled boxes, two 768-D feature rows, and six hashed fixture tensors); all recorded pre-optimizer tensor surfaces matched with no first divergence.                                                                                                                                                                                                                                                                                                              |
| S2    | `PARITY_REQUIRE=1 uv run --package radm --extra training --extra vendor pytest models/radm/tests/vendor_parity/test_s1_radm_training.py::test_s2_radm_one_optimizer_step_parity -s -q -rs`                                                                                                                                                                                                                                                                                                                                                                                          | `.cache/radm/supported-v100/s3-wiring-20260814/s2_one_step_trace.json`                                                 | Asserted gate over one optimizer step on that same batch: loss, gradients, clipped gradients, parameters, optimizer state, learning rates, and scheduler state; all recorded tensors matched under the stated S2 tolerance.                                                                                                                                                                                                                                                                                                                |
| S3    | `PARITY_REQUIRE=1 uv run --package radm --extra training --extra vendor pytest models/radm/tests/vendor_parity/test_s3_radm_wiring.py -s -q -rs`                                                                                                                                                                                                                                                                                                                                                                                                                                    | `.cache/radm/s5-preflight/run-head-s3-lockstep-300.jsonl`                                                              | Asserted pre-S5 surfaces accepted. Report-only natural cross-system source-relative total-loss drift is `11.3%` maximum at step `249`, `0.99%` at step `300`, and `2.1%` median over the last `100` steps. The self-repeat comparison is recorded below.                                                                                                                                                                                                                                                                                   |
| S4    | `PARITY_REQUIRE=1 uv run --package radm --extra training --extra vendor pytest models/radm/tests/vendor_parity/test_s4_radm_loader_stream.py::test_s4_aligned_train_mapper_fixture_matches_vendor -s -q -rs`                                                                                                                                                                                                                                                                                                                                                                        | `.cache/radm/s4/run-023-vendor-trunc.json`                                                                             | Mapper oracle passed. Retained run-022 failed at batch `0` (`max_target_abs=0.0015838146209716797`); the current-head run-024 failed separately at batch `1` (`max_target_abs=0.0011248588562011719`), so full-stream verification is unavailable. The oracle record SHA-256 is `4cc217c21e665fd052184c662fb646c2965a46c03da4799dd846256f0b4bc1e8`; run-024 SHA-256 is `2d905868f98248ca44b507f7a3dc7ebf23104b73ee37fcbc662d338888082ec7`; retained run-022 SHA-256 is `75092fbc8a3f8679919f3cbae664235265eb1efdd83682e03f589eb7ae466314`. |
| S5    | `PARITY_REQUIRE=1 UV_FROZEN=1 uv run --package radm --extra training --extra vendor python -m models.radm.tests.vendor_parity.evaluate_radm_checkpoint --checkpoint .cache/radm/s5/cgl/evaluation-check/cross-eval/seed-1/original-weights-package/model_original.ckpt --manifest .cache/radm/s5/cgl/evaluation-check/cross-eval/seed-1/original-weights-package/launch-manifest.json --seed 1 --data-root .cache/radm/data/cgl --output-dir .cache/radm/s5/cgl/evaluation-check/cross-eval/seed-1/original-weights-package-v7/package-inference --device cuda --evaluation-seed 1` | `.cache/radm/full-run/cgl/manifest.json; evaluation-path-parity: .cache/radm/full-run/cgl/evaluation-path-parity.json` | `blocked (corrected training stream not retrained; S5 retrain pending)`. The retained full-run outputs are diagnosis evidence only.                                                                                                                                                                                                                                                                                                                                                                                                        |

## Reproduction Results

The verdict for the retained CGL runs is blocked. The package stream used for those runs differed from the original at launch, the stream and mapper implementation are now corrected, and the user has chosen not to retrain in this PR. The CGL-v2 row records the scope boundary. The diagnosis tables and the 2×2 display below round values to four decimal places; the unrounded package fixed-evaluator values are listed after that table and the regenerated JSON contains all parsed values.

| Dataset | System   | Status                                                                  | Seed scope        | Primary metrics                       | Loss evidence                                                       | Artifact summary                                       |
| ------- | -------- | ----------------------------------------------------------------------- | ----------------- | ------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------ |
| CGL     | original | `blocked (corrected training stream not retrained; S5 retrain pending)` | training-seed n=3 | Diagnosis metrics below; no S5 claim. | Original terminal checkpoints paired with invalid-stream diagnosis. | `.cache/radm/full-run/cgl/original-seed-{1,2,3}-eval/` |
| CGL     | package  | `blocked (corrected training stream not retrained; S5 retrain pending)` | training-seed n=3 | Diagnosis metrics below; no S5 claim. | Package terminal checkpoints came from the pre-correction stream.   | `.cache/radm/full-run/cgl/package-seed-{1,2,3}-eval/`  |
| CGL-v2  | original | `blocked (CGL-v2 outside approved S5 scope)`                            | not run           | not measured                          | Outside approved scope.                                             | `.cache/radm/data/cgl/`                                |
| CGL-v2  | package  | `blocked (CGL-v2 outside approved S5 scope)`                            | not run           | not measured                          | Outside approved scope.                                             | `models/radm/configs/training/radm_cgl_v2.yaml`        |

### Comparison Scope

| Dataset | System   | Evaluator                                                                                  | Test split                         | Checkpoint-selection rule                                                                                             | Sample count                                                                                 |
| ------- | -------- | ------------------------------------------------------------------------------------------ | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| CGL     | original | `vendor/radm/train_net.py --eval-only --resume` with the released `vendor/radm/metrics.py` | CGL test split                     | Matching terminal `model_final.pth` from the original run                                                             | `1,035` layouts per evaluation seed; one inference step; class threshold `0.25`; NMS `0.15`. |
| CGL     | package  | Fixed equivalent evaluator with the released `vendor/radm/metrics.py`                      | CGL test split                     | Retained terminal `terminal-step-250000.ckpt`, guarded by global step, seed, config identity, and checkpoint identity | `1,035` layouts per evaluation seed; one inference step; class threshold `0.25`; NMS `0.15`. |
| CGL-v2  | both     | Not run; outside approved S5 scope                                                         | Not run; outside approved S5 scope | Not run; outside approved S5 scope                                                                                    | Not run; outside approved S5 scope                                                           |

The original evaluator command is recorded in the regenerated manifest. The package evaluator command is recorded in the regenerated evaluation-path parity artifact and its command record. Both paths use no test-time flip, fixed `800/1333` resizing, original-input-frame boxes, and evaluation seed `1` for the shared-weight check.

### Six-checkpoint diagnosis

This table was evaluated with the equivalent evaluator on the invalid-stream runs. It is not an S5 reproduction result.

| Seed | System   | Predictions | `R_ove` | `R_ali` | `R_und` | `R_occ` | `R_com` |
| ---- | -------- | ----------: | ------: | ------: | ------: | ------: | ------: |
| 1    | original |       4,635 |  0.0840 |  0.0056 |  0.9752 |  0.9990 |  7.9188 |
| 1    | package  |       5,867 |  0.0660 |  0.0073 |  0.9093 |  0.9990 |  8.3357 |
| 2    | original |       4,625 |  0.0869 |  0.0071 |  0.9783 |  0.9981 |  7.7286 |
| 2    | package  |       5,642 |  0.0740 |  0.0071 |  0.9157 |  1.0000 |  7.9484 |
| 3    | original |       4,643 |  0.0864 |  0.0062 |  0.9758 |  0.9981 |  7.5926 |
| 3    | package  |       5,784 |  0.0698 |  0.0070 |  0.9240 |  1.0000 |  8.0223 |

The aggregate means ± sample standard deviations are `R_ove` `0.0857 ± 0.0015` versus `0.0700 ± 0.0040`, `R_ali` `0.0063 ± 0.0008` versus `0.0071 ± 0.0001`, `R_und` `0.9764 ± 0.0016` versus `0.9163 ± 0.0074`, `R_occ` `0.9984 ± 0.0006` versus `0.9997 ± 0.0006`, and `R_com` `7.7467 ± 0.1638` versus `8.1021 ± 0.2057` for original versus package. The exploratory tests and full parsed values remain in `.cache/radm/full-run/cgl/comparison.json`.

The diagnosis uses both-direction range containment, two-sided unequal-variance Welch tests, and exact two-sided relabeling over all `20` assignments. The p-values are exploratory because issue 261 prespecifies no test or threshold; they cover only these three paired seeds and do not establish equivalence or non-equivalence. `R_shm` and `R_sub` are unavailable because the released vendor metric module does not implement them.

### Fixed-evaluator 2x2 check

The fixed-evaluator package cell is the retained `5,867`-prediction run, not a placeholder. Its prediction file is `.cache/radm/s5/cgl/evaluation-check/cross-eval/seed-1/package-weights-vendor/output/inference/coco_instances_results.json`.

| Weights         | Evaluator               | Predictions | `R_ove` | `R_ali` | `R_und` | `R_occ` | `R_com` |
| --------------- | ----------------------- | ----------: | ------: | ------: | ------: | ------: | ------: |
| Original seed 1 | Vendor evaluator        |       4,635 |  0.0840 |  0.0056 |  0.9752 |  0.9990 |  7.9188 |
| Original seed 1 | Fixed package evaluator |       4,635 |  0.0840 |  0.0056 |  0.9752 |  0.9990 |  7.9188 |
| Package seed 1  | Vendor evaluator        |       5,867 |  0.0660 |  0.0073 |  0.9093 |  0.9990 |  8.3357 |
| Package seed 1  | Fixed package evaluator |       5,867 |  0.0660 |  0.0073 |  0.9093 |  0.9990 |  8.3357 |

The package fixed-evaluator row is the actual `5,867`-prediction run above. Its full values are `R_ove=0.065992`, `R_ali=0.007271`, `R_und=0.909264`, `R_occ=0.999034`, and `R_com=8.335750` in the regenerated comparison artifact. The cited retained prediction file has no embedded checkpoint or evaluator manifest; the regenerated comparison records its path, count, and SHA-256. It is distinct from the retained full-run package-seed-1 output, which has `6,877` predictions and is not used for this cell.

### S3 classification

The cross-system natural total-loss drift uses the source-relative statistic `abs(package - source) / abs(source)`: it reaches `11.3%` at step `249`, is `0.99%` at step `300`, and has a `2.1%` median over the last `100` steps. The vendor self-repeat is `8.1%` maximum, `5.2%` at step `300`, and `2.6%` over the last `100` steps. The two package self-repeat artifacts have bitwise-identical rows, so the package is run-to-run deterministic. Their within-file divergence is a harness asymmetry, not package nondeterminism.

Therefore S3 is `natural: not within the vendor self-repeat envelope by the maximum statistic; same order of magnitude; synchronized exact`. This establishes that the natural cross-system trajectory does not fit the vendor's self-repeat envelope under the maximum statistic, but it does not establish a package implementation bug, exact cross-system lockstep, or S5 reproduction.

### Evaluation-path parity

The regenerated evaluation-path parity artifact is `.cache/radm/full-run/cgl/evaluation-path-parity.json` with SHA-256 `b307b17984d635aadb90ce8c876237ece5789c133cbfbe8f1f7b95bf4507ae73`. The regenerated manifest is `.cache/radm/full-run/cgl/manifest.json` with SHA-256 `73e0370aec4b3bb820b922b5de67ddd25d5685b1c122fcf22432e49902d0b2b4`.

The shared source checkpoint is `.cache/radm/s5/cgl/original-seed-1/model_final.pth` with SHA-256 `55aca7804d6e3472bbfb913867865f556481872de9fb5db63ac9ed46fa2d523d`. The package adapter checkpoint has SHA-256 `fe3be66ef92edcbb25c51051eb1c005d442bda274b68c960161d5f0cb0281f1c`. The package evaluator source is attributed to ancestor commit `ba158e016ebe8fb510d75076254d0792cb313c75`.

Prediction comparison uses exact image and category ids, exact scores, and absolute bounding-box tolerance `0.0001`. Metric comparison uses absolute tolerance `2e-9`. Both systems produced `4,635` predictions. Under the out-of-bounds tolerance `1e-6`, the vendor count is `1` and the package count is `0`: image `340` has a vendor box exceeding the width by `9.5367431640625e-06`, which is above the tolerance. The count difference is reported, not hidden by the metric pass.

## Regeneration Metadata

The regenerated artifacts are outputs of the analysis and driver scripts. The superseded aggregate files remain beside them for inspection and are not cited as current evidence. The manifest records `launched_at` as the earliest retained training launch, records each training and evaluation launch separately, and records that the aggregate was written after the retained runs. This after-the-fact write is a protocol deviation; it did not modify checkpoints, prediction files, logs, or traces.

| Artifact                                               | Current SHA-256                                                    | Superseded SHA-256 or retained evidence                                                                                                       |
| ------------------------------------------------------ | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `.cache/radm/full-run/cgl/comparison.json`             | `67d2fdc46baa90da8d2b387b348f1e462042f1abed28226e87743c4230b609ce` | `.cache/radm/full-run/cgl/comparison.json.superseded-20261004.json`                                                                           |
| `.cache/radm/full-run/cgl/manifest.json`               | `73e0370aec4b3bb820b922b5de67ddd25d5685b1c122fcf22432e49902d0b2b4` | `.cache/radm/full-run/cgl/manifest.json.superseded-20261004.json`, SHA-256 `c79c8e1ee9dfee9b1e25e8bc0043ed68033ab174a77229c274d62b49fc693ada` |
| `.cache/radm/full-run/cgl/evaluation-path-parity.json` | `b307b17984d635aadb90ce8c876237ece5789c133cbfbe8f1f7b95bf4507ae73` | `.cache/radm/full-run/cgl/evaluation-path-parity.json.superseded-20261004.json`                                                               |
| `.cache/radm/s4/run-022-full-stream.json`              | retained raw evidence                                              | SHA-256 `75092fbc8a3f8679919f3cbae664235265eb1efdd83682e03f589eb7ae466314`                                                                    |

The manifest records source config SHA-256 `66757da8054edf7a491a5147ba65acc63fe7898f6fb6a70d97f7fd4b507a0856`, package config SHA-256 `aad0dcc6e93bae66ccefd04562bcc8a6031e343f7d4322d6dfa63cc85ba4343e`, annotation SHA-256 `e131eec9130a97845096604d894b618afb20c24589e0e55c16298218c4d4b909`, and vendor metric code SHA-256 `01447490d9f196638a9a578dbf2a1a007be1e7b0296a92b8370328802616042f`. The manifest's artifact map is the source of truth for the per-seed checkpoint and prediction hashes.

## Training Commands

Run the staged parity checks.

```bash
UV_FROZEN=1 PARITY_REQUIRE=1 uv run --package radm --extra training --extra vendor pytest models/radm/tests/vendor_parity -m "vendor_parity and training" -rs
```

Run one future package training seed only after the checklist above passes and a maintainer authorizes retraining.

```bash
UV_FROZEN=1 CUDA_VISIBLE_DEVICES=<gpu> PARITY_REQUIRE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 uv run --package radm --extra training traingen fit --config models/radm/configs/training/radm_cgl.yaml --seed_everything=<seed> --trainer.deterministic=false --trainer.max_steps=250000 --trainer.limit_val_batches=0
```

The terminal evaluator must reject any checkpoint whose global step is not `250,000` or whose seed, resolved config path, config SHA-256, or identity metadata does not match its launch manifest.

## References and deviations

The design and amendments are recorded in [issue 261](https://github.com/creative-graphic-design/design-generators/issues/261), [the lockstep amendment](https://github.com/creative-graphic-design/design-generators/issues/261#issuecomment-5318097501), and [the later validation-target diagnostic](https://github.com/creative-graphic-design/design-generators/issues/261#issuecomment-5933760418).

The proposal to remove the discrete-assignment discussion from the canonical training protocol is tracked in [issue 440](https://github.com/creative-graphic-design/design-generators/issues/440). The protocol page no longer contains that section; the RADM amendment remains cited here because it defines the accepted pre-S5 gate.

The retained package runs predate the corrected sampler and mapper. The current implementation fix is carried by ancestor commit `ba158e016ebe8fb510d75076254d0792cb313c75`; the retained run manifests remain the provenance record for the invalid runs. The aggregate manifest was written after those runs and records that deviation explicitly.

This PR delivers the package, S0-S4 checks, evaluation-path parity, and an honest S5 diagnosis. It does not deliver retrained S5 checkpoints.

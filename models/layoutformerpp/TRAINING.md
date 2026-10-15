---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutFormerPP
---

# LayoutFormer++ Training

The package has passing staged S0-S4 evidence for all 12 LayoutFormer++ recipe families covered by this package. Full-run trained-checkpoint reproduction is not claimed because S5 has not been launched.

The covered recipe families are RICO25 label, RICO25 label-size, RICO25 relation, RICO25 refinement, RICO25 completion, RICO25 unconditional, PubLayNet label, PubLayNet label-size, PubLayNet relation, PubLayNet refinement, PubLayNet completion, and PubLayNet unconditional. The attempt-4 campaign used source commit `df0b6c24de6d705ca5c1e30c52d199eeb5df74db` and the normalized runtime freeze `7d05fc1e0d322bf2c6332b0dab36a0c039edae90ad4507468b107c829047c9fb`, which every attempt-4 record cites. The HEAD runner produced attempt-4; attempt-3 remains on disk as superseded evidence.

Run commands from the repository root. Keep generated evidence, logs, data, and checkpoints under `.cache/layoutformerpp/`. Keep the authoritative processed data outside repository artifacts and provide it through `LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root>`.

## Install

```bash
uv sync --package layoutformerpp --extra training
uv sync --package layoutformerpp --extra training --extra vendor
```

The vendor extra is needed for original-implementation reference checks. The package training path uses the package-local DataModule and Lightning module.

The audited runtime is recorded in `models/layoutformerpp/scripts/audit_runtime_requirements.txt` and `models/layoutformerpp/scripts/audit_runtime.freeze`; obtain torch-family packages from the cu126 PyTorch index and all other packages from PyPI, using the pinned versions in those files.

## Data

| Dataset   | Source                                                                                                   | Required runtime input                                                                                |
| --------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| RICO25    | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico)           | Approved processed RICO25 data supplied through `LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root>`    |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | Approved processed PubLayNet data supplied through `LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root>` |

The fixed-batch checks use the repository's source-shaped fixtures. The repeated training and loader-stream checks compare the original loader with the package-local DataModule and DataLoader on the approved processed splits. Missing parity data fails the run.

## Configs

Training configs live under `models/layoutformerpp/configs/training`. The RICO25 families use the `rico25_` prefix and the PubLayNet families use the `publaynet_` prefix; each family has `label`, `label_size`, `relation`, `refinement`, `completion`, and `unconditional` variants.

## Scheduler and Recipe Notes

The reference mode is the plain-PyTorch basic training path. The parity harness compares the package scheduler, optimizer defaults, loss inputs, data encoding, task identifiers, and seed-order behavior with the original implementation. The original scheduler reference is constructed from pinned `deepspeed==0.5.10` inside the audited runtime; S0 crosses the warmup boundary and S3 uses that pinned sequence as its reference. Scheduler cadence and state-dependent loader branches are exercised by the staged evidence before a full-run claim is considered.

Sampler rule — not applicable: these recipes do not use a timestep or importance sampler. EMA rule — not applicable: the recipes disable EMA. AMP rule — not applicable: the training configs use `precision: 32-true`. Multi-worker loader rule — not applicable: the package DataModule uses `num_workers=0` for the compared streams. Scheduler cadence is applicable and is covered by the S2 and S3 evidence.

The campaign runner regenerates and normalizes the selected interpreter's freeze at launch, compares it with `models/layoutformerpp/scripts/audit_runtime.freeze`, runs the CUDA smoke first, and then runs the six recorded staged processes; `models/layoutformerpp/scripts/training_parity_record_plugin.py` writes create-only redacted records.

## Seed Policy

The staged records preserve each run's process boundary, runtime freeze reference, selected-device setting, and test report. A numeric seed match alone is not a seed-paired claim. The full-run stage has no accepted training-seed or evaluation-seed scope because it has not been launched.

## Evidence Validity

Attempt-4 is the current staged evidence for source commit `df0b6c24de6d705ca5c1e30c52d199eeb5df74db` and normalized runtime freeze `7d05fc1e0d322bf2c6332b0dab36a0c039edae90ad4507468b107c829047c9fb`; attempt-3 is superseded. The rerun rule required this campaign because the tests changed in campaign-exercised paths: S0 now parameterizes `learning_rate` from each recipe, S2 covers all 12 recipe `reference_optimizer_scheduler_state` learning rates, and S3 covers all 12 recipe `scheduler_last_batch_iteration` sources. The HEAD runner is the code that produced attempt-4. The rerun-rule path set is `models/layoutformerpp/src models/layoutformerpp/configs models/layoutformerpp/tests models/layoutformerpp/scripts vendor lib`. A historical bounded PubLayNet label-conditioned `gen_t` comparison was produced at the older head `2512c25d0e7eae188afed43df94bbb06bb2c52ea`; it remains excluded from current evidence under the regression rule.

## Validation Stages

| Stage | Scope                                      | Purpose                                                                                            |
| ----- | ------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity | Confirms source manifests, configuration values, topology, vocabulary, and forward outputs.        |
| S1    | Fixed-batch pre-optimizer trace parity     | Compares the first fixed batch before optimizer mutation.                                          |
| S2    | One optimizer-step parity                  | Compares backward, optimizer, and scheduler state after one update.                                |
| S3    | Short deterministic multi-batch run        | Compares repeated training, validation, and logging; separately smoke-tests production-fit wiring. |
| S4    | Deterministic loader stream                | Compares the authoritative train and validation streams.                                           |
| S5    | Full-run statistical comparison            | Requires a separate user-owned compute decision and launch manifest.                               |

## Stage Evidence

The campaign runner supplies `PYTHONPATH`, `PARITY_REQUIRE`, `LAYOUTFORMERPP_AUDIT_VENV`, `LAYOUTFORMERPP_RECORD_DIR`, `LAYOUTFORMERPP_RUN_STAGE`, `LAYOUTFORMERPP_RUN_REPEAT`, `LAYOUTFORMERPP_RUN_STARTED_AT`, `LAYOUTFORMERPP_SOURCE_COMMIT`, `LAYOUTFORMERPP_RUNTIME_FREEZE_SHA256`, `LAYOUTFORMERPP_RUNTIME_FREEZE_RELATIVE`, `LAYOUTFORMERPP_REPORT_DATA_ROOT`, and `LAYOUTFORMERPP_REPORT_REPO_ROOT` to each Stage Evidence command. The launch shell supplies `CUDA_VISIBLE_DEVICES` and `LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root>`.

| Stage | Command                                                                                                                                                                                                                                                                                                            | Artifact                                                    | Result                                                                                                                                |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s0 -rs -q`                                                  | `.cache/layoutformerpp/post-merge/attempt-4/s0/record.json` | passed: `50` tests; record SHA-256 `5ef17aeb6d59c5fd49712fd7dabc1541223e53d5e345e5d417f530be48a52df1`                                 |
| S1    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s1 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-4/s1/record.json` | passed: `12` tests; record SHA-256 `7d1e34e502470e1bb392f46a9dbe6c55c05751531d75b639e1161c57718f09c2`                                 |
| S2    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s2 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-4/s2/record.json` | passed: `12` tests; record SHA-256 `e6c4ece7bf530e3fefca90d9674a26f5c8005ba4eab572d890e954b78a8f3d2f`                                 |
| S3    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s3 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-4/s3/record.json` | passed: `22` tests; fixed `rtol=1e-4`, `atol=1e-5`; record SHA-256 `8cb72f9040df5c2a5988880b466cda16ee440d5f6286941c77e96dc535219728` |
| S4    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s4 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-4/s4/record.json` | passed: `13` tests; record SHA-256 `9e4ccbc6c1c73e5e0006746d8168beac5fa97f17e2f23a3862347cef969cde5f`                                 |
| S5    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s5 -rs -q`                                                                                   | `<not produced; S5 not launched>`                           | not launched; no trained-checkpoint result is reported                                                                                |

The S3 comparison gate is the fixed tolerance shown above. A separate S3 process repeated the same comparison for reproducibility evidence; it is not an envelope gate. Its record is `.cache/layoutformerpp/post-merge/attempt-4/s3-repeat-2/record.json`, with `22` passed tests and record SHA-256 `047e86969f0ef4291d6bbd99c1276af3f48ce33a5d937aa253fb74c1fd94532a`. Every process wrote its own record and freeze file, and no divergence was observed in the accepted comparison.

## Reproduction Results

The staged S0-S4 checks pass for every supported recipe family. Full-run trained-checkpoint reproduction is not claimed for any family because the full-run stage has not been launched.

| Dataset                 | System  | Status                                                                                               | Seed scope     | Primary metrics | Loss evidence | Artifact summary                              |
| ----------------------- | ------- | ---------------------------------------------------------------------------------------------------- | -------------- | --------------- | ------------- | --------------------------------------------- |
| RICO25 label            | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-4/` |
| RICO25 label-size       | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-4/` |
| RICO25 relation         | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-4/` |
| RICO25 refinement       | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-4/` |
| RICO25 completion       | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-4/` |
| RICO25 unconditional    | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-4/` |
| PubLayNet label         | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-4/` |
| PubLayNet label-size    | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-4/` |
| PubLayNet relation      | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-4/` |
| PubLayNet refinement    | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-4/` |
| PubLayNet completion    | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-4/` |
| PubLayNet unconditional | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-4/` |

### Comparison Scope

The staged evaluator is the LayoutFormer++ training parity harness. It compares both systems on the train and validation loader paths for the approved processed dataset inputs. Full-run checkpoint selection and evaluation metrics do not apply until the user-owned full-run decision is made.

| Dataset                 | System | Evaluator               | Test split           | Checkpoint-selection rule | Sample count                          |
| ----------------------- | ------ | ----------------------- | -------------------- | ------------------------- | ------------------------------------- |
| RICO25 label            | both   | training parity harness | train and validation | not applicable            | `2` train plus `2` validation batches |
| RICO25 label-size       | both   | training parity harness | train and validation | not applicable            | `2` train plus `2` validation batches |
| RICO25 relation         | both   | training parity harness | train and validation | not applicable            | `2` train plus `2` validation batches |
| RICO25 refinement       | both   | training parity harness | train and validation | not applicable            | `2` train plus `2` validation batches |
| RICO25 completion       | both   | training parity harness | train and validation | not applicable            | `2` train plus `2` validation batches |
| RICO25 unconditional    | both   | training parity harness | train and validation | not applicable            | `2` train plus `2` validation batches |
| PubLayNet label         | both   | training parity harness | train and validation | not applicable            | `2` train plus `2` validation batches |
| PubLayNet label-size    | both   | training parity harness | train and validation | not applicable            | `2` train plus `2` validation batches |
| PubLayNet relation      | both   | training parity harness | train and validation | not applicable            | `2` train plus `2` validation batches |
| PubLayNet refinement    | both   | training parity harness | train and validation | not applicable            | `2` train plus `2` validation batches |
| PubLayNet completion    | both   | training parity harness | train and validation | not applicable            | `2` train plus `2` validation batches |
| PubLayNet unconditional | both   | training parity harness | train and validation | not applicable            | `2` train plus `2` validation batches |

## Regeneration Metadata

The launch manifest records the source commit, runtime freeze hash, selected-device setting, redacted data-root placeholder, fixed S3 tolerances, and every record and freeze path under `.cache/layoutformerpp/post-merge/attempt-4/launch_manifest.json`. The CUDA preflight output is under `.cache/layoutformerpp/post-merge/attempt-4/cuda-smoke.json`. The preflight uses a tiny matmul on the selected logical CUDA device after `CUDA_VISIBLE_DEVICES=<device>` is applied and fails closed before launching a parity process when the smoke fails.

The runtime and campaign artifacts are cache-local and ignored; no generated tensors, checkpoints, images, downloaded datasets, or full-run artifacts belong in git.

## Training Commands

Set `LAYOUTFORMERPP_AUDIT_VENV=<your audit venv>` to the selected audited interpreter directory before the campaign command.

```bash
CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> \
  "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" \
  models/layoutformerpp/scripts/run_training_parity_campaign.py --attempt "$LAYOUTFORMERPP_ATTEMPT"
```

Set `LAYOUTFORMERPP_ATTEMPT` to a new name before running the campaign. The runner refuses to overwrite an existing attempt directory.

Train one recipe after the staged evidence and a separate user-owned full-run decision.

```bash
CUDA_VISIBLE_DEVICES=<device> LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> \
  uv run --package layoutformerpp --extra training traingen fit \
  --config models/layoutformerpp/configs/training/<recipe>.yaml \
  --trainer.devices=1
```

Do not launch the full-run stage from this document. It remains a separate compute decision after the complete staged ladder and any required evaluation-path parity prerequisite are available.

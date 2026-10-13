---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutFormerPP
---

# LayoutFormer++ Training

The package has passing staged S0-S4 evidence for all twelve LayoutFormer++ recipe families covered by this package. Full-run trained-checkpoint reproduction is not claimed because S5 has not been launched.

The covered recipe families are RICO25 label, RICO25 label-size, RICO25 relation, RICO25 refinement, RICO25 completion, RICO25 unconditional, PubLayNet label, PubLayNet label-size, PubLayNet relation, PubLayNet refinement, PubLayNet completion, and PubLayNet unconditional. The evidence campaign used source commit `080c492bd893f21c6e8add85b322ec26a78e7fb0` and the normalized runtime freeze `408012a8b7451384836b3d7bf6f2c0cc1efeb3c30521a4c0817ba6891a635ebb`, which every attempt-3 record cites.

Run commands from the repository root. Keep generated evidence, logs, data, and checkpoints under `.cache/layoutformerpp/`. Keep the authoritative processed data outside repository artifacts and provide it through `LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root>`.

## Install

```bash
uv sync --package layoutformerpp --extra training
uv sync --package layoutformerpp --extra training --extra vendor
```

The vendor extra is needed for original-implementation reference checks. The package training path uses the package-local DataModule and Lightning module.

Build the audited runtime with the committed requirements. The builder fails if the target directory already exists, so each rebuild starts from a fresh directory. The committed normalized freeze records the exact installed package versions for the evidence runtime; a fresh build is expected to match those versions, not the byte hash. The recorded hash binds each evidence record to the runtime used for that campaign.

```bash
models/layoutformerpp/scripts/build_audit_runtime.sh --runtime-dir .cache/layoutformerpp/runtime/cu126-py311
```

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

The committed runtime builder reads `models/layoutformerpp/scripts/audit_runtime_requirements.txt`, which pins the complete audited stack including `torch==2.12.0+cu126`, `torchvision==0.27.0+cu126`, `triton==3.7.0`, and `deepspeed==0.5.10` from the cu126 index. It normalizes wheel URLs and editable paths before writing the fresh freeze. The committed campaign runner runs the CUDA smoke first and then the six recorded staged processes, while `models/layoutformerpp/scripts/training_parity_record_plugin.py` writes create-only redacted records.

## Seed Policy

The staged records preserve each run's process boundary, runtime freeze reference, selected-device setting, and test report. A numeric seed match alone is not a seed-paired claim. The full-run stage has no accepted training-seed or evaluation-seed scope because it has not been launched.

## Evidence Validity

The attempt-3 records are current staged evidence for source commit `080c492bd893f21c6e8add85b322ec26a78e7fb0` and normalized runtime freeze `408012a8b7451384836b3d7bf6f2c0cc1efeb3c30521a4c0817ba6891a635ebb`. The rerun-rule path set is `models/layoutformerpp/src models/layoutformerpp/configs models/layoutformerpp/tests models/layoutformerpp/scripts vendor lib`. Its training-path diff command, `git diff --stat 080c492bd893f21c6e8add85b322ec26a78e7fb0 HEAD -- models/layoutformerpp/src models/layoutformerpp/configs lib`, is empty. The changes under `models/layoutformerpp/tests` and `models/layoutformerpp/scripts` are harness-only, so they do not invalidate attempt-3; no package source, model configuration, or shared-library training path changed. A historical bounded PubLayNet label-conditioned `gen_t` comparison was produced at the older head `2512c25d0e7eae188afed43df94bbb06bb2c52ea`. The same training-path diff from that head to the current head is 40 files changed, 1383 insertions, and 251 deletions; the regression rule invalidates that historical comparison as current evidence, so it is excluded.

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

| Stage | Command                                                                                                                                                                                                                                                                                                            | Artifact                                                    | Result                                                                                                                               |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| S0    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s0 -rs -q`                                                  | `.cache/layoutformerpp/post-merge/attempt-3/s0/record.json` | passed: `39` tests; record SHA-256 `f0526949d267d46c89b3092398413ff8f4c967c853ead78ffeafd9a6100ec079`                                |
| S1    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s1 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-3/s1/record.json` | passed: `12` tests; record SHA-256 `0edf2e6796e3b5ceea6115623cd44f1e5afa03e067daf5b2f8cdbf955671bc94`                                |
| S2    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s2 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-3/s2/record.json` | passed: `12` tests; record SHA-256 `b2db598a12567537d856eff38507efe89e2d9ee68af6c433278195523473aedc`                                |
| S3    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s3 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-3/s3/record.json` | passed: `22` tests; fixed `rtol=1e-4`, `atol=1e-5`; record SHA-256 `b886a9c8d2155b045dd26b486eb9b2900e156639b604e8c7538693d0d5ad8ee` |
| S4    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s4 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-3/s4/record.json` | passed: `13` tests; record SHA-256 `aabd919935c785533b08e3ca1bd7746809e16fe20d832906189a2954c9baef32`                                |
| S5    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s5 -rs -q`                                                                                   | `<not produced; S5 not launched>`                           | not launched; no trained-checkpoint result is reported                                                                               |

The S3 comparison gate is the fixed tolerance shown above. A separate S3 process repeated the same comparison for reproducibility evidence; it is not an envelope gate. Its record is `.cache/layoutformerpp/post-merge/attempt-3/s3-repeat-2/record.json`, with `22` passed tests and record SHA-256 `d4b95dbe1b622ad8add66e8a1d3789e3bd9be032404bfff7fd9ec3ad6001d936`. Every process wrote its own record and freeze file, and no divergence was observed in the accepted comparison.

## Reproduction Results

The staged S0-S4 checks pass for every supported recipe family. Full-run trained-checkpoint reproduction is not claimed for any family because the full-run stage has not been launched.

| Dataset                 | System  | Status                                                                                               | Seed scope     | Primary metrics | Loss evidence | Artifact summary                              |
| ----------------------- | ------- | ---------------------------------------------------------------------------------------------------- | -------------- | --------------- | ------------- | --------------------------------------------- |
| RICO25 label            | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-3/` |
| RICO25 label-size       | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-3/` |
| RICO25 relation         | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-3/` |
| RICO25 refinement       | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-3/` |
| RICO25 completion       | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-3/` |
| RICO25 unconditional    | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-3/` |
| PubLayNet label         | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-3/` |
| PubLayNet label-size    | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-3/` |
| PubLayNet relation      | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-3/` |
| PubLayNet refinement    | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-3/` |
| PubLayNet completion    | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-3/` |
| PubLayNet unconditional | package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-3/` |

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

The launch manifest records the source commit, runtime freeze hash, selected-device setting, redacted data-root placeholder, fixed S3 tolerances, and every record and freeze path under `.cache/layoutformerpp/post-merge/attempt-3/launch_manifest.json`. The CUDA preflight output is under `.cache/layoutformerpp/post-merge/attempt-3/cuda-smoke.json`. The preflight uses a tiny matmul on the selected logical CUDA device after `CUDA_VISIBLE_DEVICES=<device>` is applied and fails closed before launching a parity process when the smoke fails.

The committed builder can install the pinned runtime versions into a fresh directory by running `models/layoutformerpp/scripts/build_audit_runtime.sh` from the repository root; this documents version reproducibility, not byte-identical wheel reproduction. The runtime and campaign artifacts are cache-local and ignored; no generated tensors, checkpoints, images, downloaded datasets, or full-run artifacts belong in git.

## Training Commands

Build the audited runtime before running the staged parity campaign.

```bash
models/layoutformerpp/scripts/build_audit_runtime.sh
```

Set `LAYOUTFORMERPP_AUDIT_VENV` to `.cache/layoutformerpp/runtime/cu126-py311` before the campaign command.

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

---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutFormerPP
---

# LayoutFormer++ Training

LayoutFormer++ has no current full-run training reproduction claim for RICO25 or PubLayNet. Historical RICO25 package results for training seeds 0–2 are retained as void for the current source; PubLayNet full-run training has not been run. The staged S0–S4 training evidence is also void starting at S0 and must be rerun before a new S5 launch. RICO25 evaluation-path parity remains an inference-only result.

This report gives researchers and maintainers LayoutFormer++ reproduction results and commands to rerun the evidence checks. RICO25 is the 25-category split of the RICO interface-layout benchmark. Its six task families generate layouts from these conditions: `unconditional` uses no layout condition; `label` takes labels; `label_size` takes labels with matching boxes; `relation` takes labels and relation tuples; `completion` and `refinement` take partial layouts with labels, boxes, and a mask. The [LayoutFormer++ usage guide](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutformerpp/README.md#uses) lists the public input shapes. Each dataset-family pair is one recipe, giving 12 configurations per implementation across RICO25 and PubLayNet.

The six reproduction stages are S0 (configuration and initialization), S1 (fixed-batch pre-optimizer trace), S2 (one optimizer update), S3 (short multi-batch trajectory), S4 (deterministic data-loader stream), and S5 (full-run comparison). The S3 numerical check uses fixed tolerances. The protocol also defines a pre-S5 production loss-envelope gate based on per-seed median losses from each system's own training entry point, at least three matched seeds, and 1,500 updates per optimizer; the staged records here do not claim that check.

The evaluator reports eight metrics: alignment (mean score for how related boxes line up), `bbox_acc` (share of boxes scored correct by the evaluator), FID (Fréchet Inception Distance over generated and reference layout features), `label_acc` (share of generated elements scored with the correct label), `mIoU` (mean intersection-over-union of predicted and target boxes), `new_overlap` (mean overlap score for newly generated boxes), overlap (mean overlap score across predicted boxes), and `violation_rate` (violating relation pairs divided by evaluated relation pairs).

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

The reference mode is the plain-PyTorch basic training path. The parity harness compares the package scheduler, optimizer defaults, loss inputs, data encoding, task identifiers, and seed-order behavior with the original implementation. The original scheduler reference is constructed from pinned `deepspeed==0.5.10` inside the audited runtime; S0 crosses the warmup boundary and S3 uses that pinned sequence as its reference. The recorded scheduler and loader checks are void for the current source.

The original and package leave model initialization unseeded and apply the training seed immediately before constructing the first shuffled training loader.

Sampler rule: not applicable because these recipes do not use a timestep or importance sampler. EMA rule: not applicable because the recipes disable EMA. AMP rule: not applicable because the training configs use `precision: 32-true`. Multi-worker loader rule: not applicable because the package DataModule uses `num_workers=0` for the compared streams. Scheduler cadence is applicable and is covered by the S2 and S3 evidence.

The campaign runner records a normalized runtime freeze: sorted `uv pip freeze` output from the selected interpreter with repository and wheel paths replaced by stable package paths and versions. At launch it compares that listing with `models/layoutformerpp/scripts/audit_runtime.freeze`, runs the CUDA smoke first, and then runs the six recorded staged processes; `models/layoutformerpp/scripts/training_parity_record_plugin.py` writes create-only redacted records.

## Seed Policy

The historical RICO25 S5 comparison used training seeds 0–2 for both systems and evaluation seed 500 on the shared TEST split. Its package training results are void for the current source. Evaluation-path parity used each family's training-seed-0 vendor checkpoint on both inference paths with the same resolved evaluator settings, inputs, and evaluation seed.

## Evidence Validity

The lowest affected stage is S0. Current source changes void the staged S0–S4 records, so rerun the ladder from S0 before any new S5 launch. The staged artifacts record the source revision, normalized runtime freeze hash, and artifact hashes.

## Stage Evidence

These records are historical and void for the current source from S0.

The campaign runner supplies `PYTHONPATH`, `PARITY_REQUIRE`, `LAYOUTFORMERPP_AUDIT_VENV`, `LAYOUTFORMERPP_RECORD_DIR`, `LAYOUTFORMERPP_RUN_STAGE`, `LAYOUTFORMERPP_RUN_REPEAT`, `LAYOUTFORMERPP_RUN_STARTED_AT`, `LAYOUTFORMERPP_SOURCE_COMMIT`, `LAYOUTFORMERPP_RUNTIME_FREEZE_SHA256`, `LAYOUTFORMERPP_RUNTIME_FREEZE_RELATIVE`, `LAYOUTFORMERPP_REPORT_DATA_ROOT`, and `LAYOUTFORMERPP_REPORT_REPO_ROOT` to each Stage Evidence command. The launch shell supplies `CUDA_VISIBLE_DEVICES` and `LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root>`.

| Stage | Command                                                                                                                                                                                                                                                                                                            | Checkpoint Smoke Command                                                                                                                                                                                  | Full-run Config                                            | Artifact                                                                                                                                                                                                   | Result                                                                                                                                                                                               | Checkpoint Path                                                                                                             | Checkpoint SHA-256                                                 |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| S0    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s0 -rs -q`                                                  | not required                                                                                                                                                                                              | not required                                               | `.cache/layoutformerpp/post-merge/attempt-4/s0/record.json`                                                                                                                                                | passed: `50` tests; record SHA-256 `5EF17AEB6D59C5FD49712FD7DABC1541223E53D5E345E5D417F530BE48A52DF1`                                                                                                | not required                                                                                                                | not required                                                       |
| S1    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s1 -rs -q` | not required                                                                                                                                                                                              | not required                                               | `.cache/layoutformerpp/post-merge/attempt-4/s1/record.json`                                                                                                                                                | passed: `12` tests; record SHA-256 `7D1E34E502470E1BB392F46A9DBE6C55C05751531D75B639E1161C57718F09C2`                                                                                                | not required                                                                                                                | not required                                                       |
| S2    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s2 -rs -q` | not required                                                                                                                                                                                              | not required                                               | `.cache/layoutformerpp/post-merge/attempt-4/s2/record.json`                                                                                                                                                | passed: `12` tests; record SHA-256 `E6C4ECE7BF530E3FEFCA90D9674A26F5C8005BA4EAB572D890E954B78A8F3D2F`                                                                                                | not required                                                                                                                | not required                                                       |
| S3    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s3 -rs -q` | `traingen fit --config models/layoutformerpp/configs/training/rico25_label.yaml --trainer.max_epochs=1 --trainer.limit_train_batches=2 --trainer.limit_val_batches=2 --trainer.check_val_every_n_epoch=1` | not required                                               | `.cache/layoutformerpp/post-merge/attempt-4/s3/record.json`                                                                                                                                                | passed: `22` tests; fixed `rtol=1e-4`, `atol=1e-5`; record SHA-256 `8CB72F9040DF5C2A5988880B466CDA16EE440D5F6286941C77E96DC535219728`; all six checked-in family configs passed the checkpoint smoke | `.cache/layoutformerpp/s5/checkpoint-smoke-20261105T022447Z/label/lightning_logs/version_0/checkpoints/epoch=0-step=2.ckpt` | `17232dd4bebf8ac688577fe3fed851f76a10fcf1c1e2976b158d633b8009c636` |
| S4    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s4 -rs -q` | not required                                                                                                                                                                                              | not required                                               | `.cache/layoutformerpp/post-merge/attempt-4/s4/record.json`                                                                                                                                                | passed: `13` tests; record SHA-256 `9E4CCBC6C1C73E5E0006746D8168BEAC5FA97F17E2F23A3862347CEF969CDE5F`                                                                                                | not required                                                                                                                | not required                                                       |
| S5    | `python main.py --dataset rico --tasks <family evaluator task> --test --eval_seed 500 --eval_ckpt_tag s5`                                                                                                                                                                                                          | not required                                                                                                                                                                                              | `models/layoutformerpp/configs/training/rico25_label.yaml` | `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-stage-evidence-20261105/manifest.json; evaluation-path-parity: .cache/layoutformerpp/s5/posthoc-parity/recheck-2ulp-20261105T071700Z/manifest.json` | Historical RICO25 training seeds 0–2: void for current seed timing. Evaluation-path predictions match bitwise for all six; metrics match bitwise for four; two meet the 2-ULP rule.                  | not required                                                                                                                | not required                                                       |

The checkpoint smoke ran on 2026-11-05, after the full runs launched from 2026-10-13 through 2026-10-17. It used the six checked-in RICO25 family configs unchanged from those runs. Each config's SHA-256 matches its `config_sha256` value in the campaign launch records. The six per-family smoke entries record no separate source commit. All six one-epoch smokes used two training and two validation batches, exited successfully, and produced a top-k checkpoint named `epoch=0-step=2.ckpt`. The smoke manifest records each config, checkpoint path, SHA-256, and runtime. The S3 row records `rico25_label.yaml`, matching the `config` field in the smoke manifest. The smoke used Python `3.11.15`, Torch `2.12.0+cu126`, and CUDA build `12.6`; its normalized runtime freeze matches the recorded S5 freeze SHA-256 `7D05FC1E0D322BF2C6332B0DAB36A0C039EDAE90AD4507468B107C829047C9FB`. The smoke manifest SHA-256 is `1F874462706A1D007E36C1EFAD2D24B95EB6B6E47B8BD75ACD95917FAC50B6C4`.

| Family        | Checked-in config                                                  | Config SHA-256, equal to campaign launch records                   |
| ------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| unconditional | `models/layoutformerpp/configs/training/rico25_unconditional.yaml` | `1bddb8e6dbdb4615c88b02aaabad54138362f47d5641d07826e9a4bfbd6a1749` |
| label         | `models/layoutformerpp/configs/training/rico25_label.yaml`         | `1b16d07f3095dd67bdf381abc0b2215ac8cf145a39ad9551f21d9782fd350620` |
| label_size    | `models/layoutformerpp/configs/training/rico25_label_size.yaml`    | `85d404d8aa9e6a99e06cfb7b56495bfaf10f248f46c0c807dd72f46ca8344c0e` |
| relation      | `models/layoutformerpp/configs/training/rico25_relation.yaml`      | `5de5d45d5d9983f98145ea328da1b1b57750100e5e3a7e5ef454e68283f74710` |
| refinement    | `models/layoutformerpp/configs/training/rico25_refinement.yaml`    | `eaa3c301ef1f7b4951c7064a25e10dbfb638c25b05bd9defbbebc53b714caa54` |
| completion    | `models/layoutformerpp/configs/training/rico25_completion.yaml`    | `071c5abb14b267c72a5574fccfa85d916311ad5eafaa0c15deae8c5de23ff9c8` |

The S5 manifest records the six config hashes; its path and SHA-256 appear below. The parity manifest lists the six family artifacts and their SHA-256 values.

## Reproduction Results

The six RICO25 S5 metric sets below are historical and void for the current source. Their reruns are tracked by [`creative-graphic-design/design-generators#265`](https://github.com/creative-graphic-design/design-generators/issues/265). PubLayNet S5 has not been run. No current full-run training reproduction is claimed for either dataset.

| Dataset                 | System           | Status                                                                                               | Seed scope                 | Primary metrics          | Loss evidence     | Artifact summary                                                               |
| ----------------------- | ---------------- | ---------------------------------------------------------------------------------------------------- | -------------------------- | ------------------------ | ----------------- | ------------------------------------------------------------------------------ |
| RICO25 label            | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | historical seeds 0–2; void | historical metrics; void | not current       | `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |
| RICO25 label-size       | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | historical seeds 0–2; void | historical metrics; void | not current       | `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |
| RICO25 relation         | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | historical seeds 0–2; void | historical metrics; void | not current       | `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |
| RICO25 refinement       | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | historical seeds 0–2; void | historical metrics; void | not current       | `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |
| RICO25 completion       | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | historical seeds 0–2; void | historical metrics; void | not current       | `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |
| RICO25 unconditional    | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | historical seeds 0–2; void | historical metrics; void | not current       | `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |
| PubLayNet label         | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not run                    | not evaluated            | staged S0-S4 only | `.cache/layoutformerpp/post-merge/attempt-4/`                                  |
| PubLayNet label-size    | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not run                    | not evaluated            | staged S0-S4 only | `.cache/layoutformerpp/post-merge/attempt-4/`                                  |
| PubLayNet relation      | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not run                    | not evaluated            | staged S0-S4 only | `.cache/layoutformerpp/post-merge/attempt-4/`                                  |
| PubLayNet refinement    | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not run                    | not evaluated            | staged S0-S4 only | `.cache/layoutformerpp/post-merge/attempt-4/`                                  |
| PubLayNet completion    | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not run                    | not evaluated            | staged S0-S4 only | `.cache/layoutformerpp/post-merge/attempt-4/`                                  |
| PubLayNet unconditional | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not run                    | not evaluated            | staged S0-S4 only | `.cache/layoutformerpp/post-merge/attempt-4/`                                  |

### Comparison Scope

The staged evaluator is the LayoutFormer++ training parity harness. It compares both systems on the train and validation loader paths for the approved processed dataset inputs. Full-run checkpoint selection and evaluation metrics are recorded separately below.

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

### S5 Full-Run Evaluation Scope

The S5 evaluator is the vendor `main.py` entry point. Run `python main.py --dataset rico --tasks <family evaluator task> --test --eval_seed 500 --eval_ckpt_tag s5`; each family's resolved arguments are recorded in its evaluation record. The evaluation records and their hashes are listed in the campaign manifest.

| Dataset | System           | Evaluator                                                | Test split  | Checkpoint-selection rule                                                      | Sample count         |
| ------- | ---------------- | -------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------ | -------------------- |
| RICO25  | vendor + package | `vendor/ms-layout-generation/LayoutFormer++/src/main.py` | `rico/test` | package: minimum aggregate `val_loss` checkpoint; vendor: `best_epoch` pointer | 3,729 per evaluation |

Evaluation-path parity compares both paths using the same training-seed-0 vendor checkpoint, resolved family settings, evaluation seed 500, and RICO25 TEST inputs. Its manifest records the source revisions, input hashes, and six family artifacts.

The comparison manifest `.cache/layoutformerpp/s5/posthoc-parity/recheck-2ulp-20261105T071700Z/manifest.json` (SHA-256 `f59a023a30b4d4776929ef2d137f8abc8ae5247cb476e328e461aa734ee17042`) maps the six parity records to their SHA-256 values. Each record stores the runner SHA-256, clean-tree flag, and hashes for the train, validation, and TEST input files.

### S5 Evaluation-Path Parity

The evaluation-path parity check used the same training-seed-0 vendor checkpoint, resolved family settings, evaluation seed 500, and complete RICO25 TEST split of 3,729 layouts for both paths. The package side called the public `LayoutFormerPPPipeline.from_pretrained` path with its published decoding defaults; the runner does not rebuild or pass vendor constraint logic. For every family, the original predictions came from vendor evaluator commit `1498ff300710b4fc204aece537582d37ca447fc7`, package predictions came from pipeline commit `704ef10a2c16255538a7a2bf84fe36d3f54256a1`, and the saved-output comparison used runner commit `489726b9d1ac748c453d9a5894bdac3e89cec9e6`. The comparison reused saved predictions and metrics without rerunning inference or the original evaluator.

Predictions match bitwise for all six families. Metrics match bitwise for four families. The `label_size` mIoU values `0.49364049868866455` and `0.49364049868866466` differ by `1.1102230246251565e-16`, exactly 2 ULP at 0.49. The `refinement` mIoU values `0.7643045336663307` and `0.7643045336663308` differ by `1.1102230246251565e-16`, 1 ULP at that magnitude. Both pass the 2-ULP rule; every other metric is bitwise equal. An out-of-bounds (OOB) count is the number of predicted boxes outside the original input frame.

| Family        | Prediction commits (vendor / package)                                                   | Comparison runner commit                   | Predictions bitwise equal | Metrics bitwise equal | Metrics accepted | Metric difference (package − original)                                                                                                                                | OOB original/package | Parity artifact and SHA-256                                                                                                                                            |
| ------------- | --------------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------- | --------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| unconditional | `1498ff300710b4fc204aece537582d37ca447fc7` / `704ef10a2c16255538a7a2bf84fe36d3f54256a1` | `489726b9d1ac748c453d9a5894bdac3e89cec9e6` | true                      | true                  | yes              | all eight metrics bitwise equal                                                                                                                                       | 9/9                  | `.cache/layoutformerpp/s5/posthoc-parity/recheck-2ulp-20261105T071700Z/unconditional/parity.json` (`3ac5cb7dc24eb31b075f20737cc710948a7fb1cdcb3c607750766fcd1d5741ab`) |
| label         | `1498ff300710b4fc204aece537582d37ca447fc7` / `704ef10a2c16255538a7a2bf84fe36d3f54256a1` | `489726b9d1ac748c453d9a5894bdac3e89cec9e6` | true                      | true                  | yes              | all eight metrics bitwise equal                                                                                                                                       | 14/14                | `.cache/layoutformerpp/s5/posthoc-parity/recheck-2ulp-20261105T071700Z/label/parity.json` (`20f2d43c8a1189ea663061cf6bcd40b0851f0ea90bc65ba940ec38c624c6cc36`)         |
| label_size    | `1498ff300710b4fc204aece537582d37ca447fc7` / `704ef10a2c16255538a7a2bf84fe36d3f54256a1` | `489726b9d1ac748c453d9a5894bdac3e89cec9e6` | true                      | false                 | yes              | mIoU original `0.49364049868866455`; package `0.49364049868866466`; delta and absolute difference `1.1102230246251565e-16` (2 ULP); other seven metrics bitwise equal | 155/155              | `.cache/layoutformerpp/s5/posthoc-parity/recheck-2ulp-20261105T071700Z/label_size/parity.json` (`f474a7f87470dbe5e20c6fb2b49db1c5f1a1e4c06f406c12814d184814a6333b`)    |
| relation      | `1498ff300710b4fc204aece537582d37ca447fc7` / `704ef10a2c16255538a7a2bf84fe36d3f54256a1` | `489726b9d1ac748c453d9a5894bdac3e89cec9e6` | true                      | true                  | yes              | all eight metrics bitwise equal                                                                                                                                       | 533/533              | `.cache/layoutformerpp/s5/posthoc-parity/recheck-2ulp-20261105T071700Z/relation/parity.json` (`d48fe7e29fb8867df55b6512e0a54c6c365e09f377866fee245793a1e1e395e9`)      |
| refinement    | `1498ff300710b4fc204aece537582d37ca447fc7` / `704ef10a2c16255538a7a2bf84fe36d3f54256a1` | `489726b9d1ac748c453d9a5894bdac3e89cec9e6` | true                      | false                 | yes              | mIoU original `0.7643045336663307`; package `0.7643045336663308`; delta and absolute difference `1.1102230246251565e-16` (1 ULP); other seven metrics bitwise equal   | 60/60                | `.cache/layoutformerpp/s5/posthoc-parity/recheck-2ulp-20261105T071700Z/refinement/parity.json` (`6f7220e76d92940e56824e49fea4afdb2be8ad521e6f350208f3d726c829927c`)    |
| completion    | `1498ff300710b4fc204aece537582d37ca447fc7` / `704ef10a2c16255538a7a2bf84fe36d3f54256a1` | `489726b9d1ac748c453d9a5894bdac3e89cec9e6` | true                      | true                  | yes              | all eight metrics bitwise equal                                                                                                                                       | 38/38                | `.cache/layoutformerpp/s5/posthoc-parity/recheck-2ulp-20261105T071700Z/completion/parity.json` (`6d0c35497539be935a4ca861666298939e45b3d3d96c3cbc1f0694a3ffff20df`)    |

The runner records original and package values, package-minus-original delta, absolute difference, and comparison case for every metric. Predictions must be bitwise equal. Each metric passes when the binary64 values are equal or `abs(package - original) <= 2 * math.ulp(max(abs(package), abs(original)))`; a larger difference fails.

### Historical three-seed S5 metrics (training verdict void for current source)

Each value is the mean ± sample standard deviation across training seeds 0–2 for evaluation seed 500. The verdict uses unrounded min/max range intersection for each metric; Δ/σ uses pooled sample standard deviation and is report-only. Means and sample standard deviations are rounded to six decimals, Δ/σ to three decimals; `not defined` means pooled sample standard deviation is zero.

| Family        | Metric         | Vendor mean ± SD      | Package mean ± SD     | Ranges intersect | Δ/σ (report-only) |
| ------------- | -------------- | --------------------- | --------------------- | ---------------- | ----------------- |
| unconditional | alignment      | 0.000604 ± 0.000671   | 0.000298 ± 0.000037   | yes              | -0.643            |
| unconditional | bbox_acc       | 0.000000 ± 0.000000   | 0.000000 ± 0.000000   | yes              | not defined       |
| unconditional | fid            | 71.377748 ± 20.709848 | 57.189807 ± 10.754078 | yes              | -0.860            |
| unconditional | label_acc      | 0.000000 ± 0.000000   | 0.000000 ± 0.000000   | yes              | not defined       |
| unconditional | mIoU           | 0.773253 ± 0.034851   | 0.747503 ± 0.022345   | yes              | -0.880            |
| unconditional | new_overlap    | 0.530536 ± 0.031699   | 0.568032 ± 0.072011   | yes              | 0.674             |
| unconditional | overlap        | 0.686729 ± 0.032002   | 0.715415 ± 0.089252   | yes              | 0.428             |
| unconditional | violation_rate | 0.000000 ± 0.000000   | 0.000000 ± 0.000000   | yes              | not defined       |
| label         | alignment      | 0.001116 ± 0.000865   | 0.000536 ± 0.000317   | yes              | -0.891            |
| label         | bbox_acc       | 0.292289 ± 0.004505   | 0.294989 ± 0.001814   | yes              | 0.786             |
| label         | fid            | 21.797342 ± 2.331120  | 19.561880 ± 1.480448  | yes              | -1.145            |
| label         | label_acc      | 1.000000 ± 0.000000   | 1.000000 ± 0.000000   | yes              | not defined       |
| label         | mIoU           | 0.365724 ± 0.001597   | 0.366942 ± 0.001320   | yes              | 0.832             |
| label         | new_overlap    | 0.833023 ± 0.058000   | 0.811227 ± 0.046869   | yes              | -0.413            |
| label         | overlap        | 0.985945 ± 0.063651   | 0.967414 ± 0.044961   | yes              | -0.336            |
| label         | violation_rate | 0.000000 ± 0.000000   | 0.000000 ± 0.000000   | yes              | not defined       |
| label_size    | alignment      | 0.001456 ± 0.000230   | 0.001260 ± 0.000080   | yes              | -1.138            |
| label_size    | bbox_acc       | 0.737945 ± 0.001245   | 0.735515 ± 0.004703   | yes              | -0.706            |
| label_size    | fid            | 3.200331 ± 0.493546   | 3.006926 ± 0.973540   | yes              | -0.251            |
| label_size    | label_acc      | 1.000000 ± 0.000000   | 1.000000 ± 0.000000   | yes              | not defined       |
| label_size    | mIoU           | 0.495399 ± 0.001675   | 0.493888 ± 0.006890   | yes              | -0.301            |
| label_size    | new_overlap    | 0.936665 ± 0.031662   | 0.893638 ± 0.052140   | yes              | -0.998            |
| label_size    | overlap        | 1.113906 ± 0.026794   | 1.071584 ± 0.054098   | yes              | -0.991            |
| label_size    | violation_rate | 0.000000 ± 0.000000   | 0.000000 ± 0.000000   | yes              | not defined       |
| relation      | alignment      | 0.001526 ± 0.000980   | 0.000429 ± 0.000033   | yes              | -1.582            |
| relation      | bbox_acc       | 0.289439 ± 0.003394   | 0.289777 ± 0.003326   | yes              | 0.101             |
| relation      | fid            | 16.359654 ± 0.716409  | 16.353267 ± 1.255906  | yes              | -0.006            |
| relation      | label_acc      | 1.000000 ± 0.000000   | 1.000000 ± 0.000000   | yes              | not defined       |
| relation      | mIoU           | 0.363895 ± 0.001220   | 0.361642 ± 0.001363   | yes              | -1.743            |
| relation      | new_overlap    | 0.728202 ± 0.020464   | 0.715350 ± 0.014839   | yes              | -0.719            |
| relation      | overlap        | 0.880878 ± 0.022663   | 0.870539 ± 0.016347   | yes              | -0.523            |
| relation      | violation_rate | 0.151051 ± 0.004617   | 0.151499 ± 0.001525   | yes              | 0.130             |
| refinement    | alignment      | 0.002061 ± 0.000026   | 0.002041 ± 0.000035   | yes              | -0.655            |
| refinement    | bbox_acc       | 0.598306 ± 0.008504   | 0.605102 ± 0.005645   | yes              | 0.942             |
| refinement    | fid            | 0.135199 ± 0.006722   | 0.111623 ± 0.017722   | yes              | -1.759            |
| refinement    | label_acc      | 1.000000 ± 0.000000   | 1.000000 ± 0.000000   | yes              | not defined       |
| refinement    | mIoU           | 0.772383 ± 0.008185   | 0.777226 ± 0.003134   | yes              | 0.781             |
| refinement    | new_overlap    | 0.508284 ± 0.004928   | 0.504018 ± 0.004595   | yes              | -0.895            |
| refinement    | overlap        | 0.679272 ± 0.003985   | 0.675939 ± 0.004127   | yes              | -0.821            |
| refinement    | violation_rate | 0.000000 ± 0.000000   | 0.000000 ± 0.000000   | yes              | not defined       |
| completion    | alignment      | 0.000777 ± 0.000707   | 0.000837 ± 0.000324   | yes              | 0.110             |
| completion    | bbox_acc       | 0.000000 ± 0.000000   | 0.000000 ± 0.000000   | yes              | not defined       |
| completion    | fid            | 36.215005 ± 18.767991 | 26.564235 ± 3.434770  | yes              | -0.715            |
| completion    | label_acc      | 0.000000 ± 0.000000   | 0.000000 ± 0.000000   | yes              | not defined       |
| completion    | mIoU           | 0.728138 ± 0.018269   | 0.707292 ± 0.024083   | yes              | -0.975            |
| completion    | new_overlap    | 0.576700 ± 0.034335   | 0.549157 ± 0.045742   | yes              | -0.681            |
| completion    | overlap        | 0.730054 ± 0.044647   | 0.720508 ± 0.066074   | yes              | -0.169            |
| completion    | violation_rate | 0.000000 ± 0.000000   | 0.000000 ± 0.000000   | yes              | not defined       |

## Regeneration Metadata

The source commit identifies the code used, and the dependency-inventory SHA-256 identifies the package versions used for each evidence set:

| Evidence set | Source commit SHA-1                        | Dependency inventory SHA-256                                       |
| ------------ | ------------------------------------------ | ------------------------------------------------------------------ |
| Staged S0–S4 | `DF0B6C24DE6D705CA5C1E30C52D199EEB5DF74DB` | `7D05FC1E0D322BF2C6332B0DAB36A0C039EDAE90AD4507468B107C829047C9FB` |
| RICO25 S5    | `7EA6A89A0F24283626B31BC8DA2D9150F527B763` | `7D05FC1E0D322BF2C6332B0DAB36A0C039EDAE90AD4507468B107C829047C9FB` |

The campaign manifest records four launch manifests, 36 evaluation records, six parity artifacts, and per-metric comparison cases:

| Artifact                                                                       | Purpose                  | SHA-256 and verification                                                                                                                                                 |
| ------------------------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` | S5 and parity provenance | `F62956B66DE9E55CE747C7B2F5F95C9B76B983FF0E4811D7C61F18CF736FA9D3`; verify with `sha256sum .cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |

The staged campaign launch manifest and CUDA preflight output are `.cache/layoutformerpp/post-merge/attempt-4/launch_manifest.json` and `.cache/layoutformerpp/post-merge/attempt-4/cuda-smoke.json`. The preflight runs a small matrix multiplication on the selected CUDA device and stops before parity if the CUDA check fails.

The runtime and campaign artifacts are cache-local and ignored; no generated tensors, checkpoints, images, downloaded datasets, or full-run artifacts belong in git.

## Training Commands

Set `LAYOUTFORMERPP_AUDIT_VENV=<your audit venv>` to the selected audited interpreter directory before the campaign command.

```bash
CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" models/layoutformerpp/scripts/run_training_parity_campaign.py --attempt "$LAYOUTFORMERPP_ATTEMPT" # Start the campaign.
```

Set `LAYOUTFORMERPP_ATTEMPT` to a new name before running the campaign. The runner refuses to overwrite an existing attempt directory.

Before a new S5 training run, complete S0–S4 and evaluation-path parity. The package owner is the maintainer responsible for approving training-compute allocation; the owner decides based on the requested datasets, task families, training seeds, and available GPU budget, then records the scope and allocation decision in [Issue 265](https://github.com/creative-graphic-design/design-generators/issues/265). Run one family and one seed per command, using a new attempt directory for every run.

```bash
FAMILY=label; TRAINING_SEED=0; ATTEMPT=example-s5-attempt; CUDA_VISIBLE_DEVICES=<device> LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> uv run --package layoutformerpp --extra training traingen fit --config "models/layoutformerpp/configs/training/rico25_${FAMILY}.yaml" --trainer.devices=1 --seed_everything=false --data.seed="$TRAINING_SEED" --trainer.default_root_dir=".cache/layoutformerpp/s5/current/runs/${ATTEMPT}/package-rico25_${FAMILY}-seed${TRAINING_SEED}"
```

Evaluate a trained checkpoint with the vendor evaluator and its resolved family settings.

```bash
"$LAYOUTFORMERPP_AUDIT_VENV/bin/python" vendor/ms-layout-generation/LayoutFormer++/src/main.py --dataset rico --tasks <family evaluator task> --test --eval_seed 500 --eval_ckpt_tag s5
```

Run evaluation-path parity for one family with training-seed 0. The runner loads each system's resolved settings and evaluation environment from its S5 record, checks the three processed-split hashes before vendor dataset code, and writes a new cache-local record.

```bash
FAMILY=label; PARITY_OUTPUT=.cache/layoutformerpp/s5/posthoc-parity/new-attempt/${FAMILY}; CUDA_VISIBLE_DEVICES=<device> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" models/layoutformerpp/tests/vendor_parity/run_evaluation_path_parity.py --mode orchestrate --vendor-src vendor/ms-layout-generation/LayoutFormer++/src --data-root "$LAYOUTFORMERPP_PARITY_DATA_ROOT" --family "$FAMILY" --seed 0 --output "$PARITY_OUTPUT" # Run the selected family's comparison.
```

These recorded results need no new campaign. A future RICO25 full-run campaign requires complete S0–S4 evidence and evaluation-path parity. Do not launch that stage from this document. The package owner records the separate compute allocation decision in [Issue 265](https://github.com/creative-graphic-design/design-generators/issues/265) before launch.

Relative callback `dirpath` values resolve under `trainer.default_root_dir`; give each seed a distinct root. Durable resume is opt-in. The launcher sets `CAMPAIGN_ROOT`, `LAYOUTFORMERPP_CHECKPOINT_IDENTITY`, and `LAYOUTFORMERPP_LAUNCH_MANIFEST`. It also sets `LAYOUTFORMERPP_DURABLE_ROOT` and `LAYOUTFORMERPP_DURABLE_TRACK_RELATIVE`. Set the identity's absolute checkpoint path to the callback path resolved from that seed's root. The fit command above needs none of the durable settings.

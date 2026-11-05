---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutFormerPP
---

# LayoutFormer++ Training

This report gives researchers and maintainers LayoutFormer++ reproduction results and commands to rerun the evidence checks. RICO25 is the 25-category split of the RICO interface-layout benchmark. Its six task families generate layouts from these conditions: `unconditional` uses no layout condition; `label` takes labels; `label_size` takes labels with matching boxes; `relation` takes labels and relation tuples; `completion` and `refinement` take partial layouts with labels, boxes, and a mask. The [LayoutFormer++ usage guide](https://github.com/creative-graphic-design/design-generators/blob/main/models/layoutformerpp/README.md#uses) lists the public input shapes. Each dataset-family pair is one recipe, giving 12 configurations per implementation across RICO25 and PubLayNet.

The vendor implementation (original) and this repository's LayoutFormer++ package (package) have intersecting metric ranges for all eight evaluator metrics across training seeds 0–2 in each task family. This meets the `s5-practical-reproduction` verdict, which uses per-metric range intersection. Stage S5 is the full-run comparison stage; evaluation-path parity checks original and package predictions from the same checkpoint, inputs, settings, and evaluation seed. Δ/σ is the difference between system means divided by pooled sample standard deviation and is report-only. PubLayNet has no full-run result.

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

The reference mode is the plain-PyTorch basic training path. The parity harness compares the package scheduler, optimizer defaults, loss inputs, data encoding, task identifiers, and seed-order behavior with the original implementation. The original scheduler reference is constructed from pinned `deepspeed==0.5.10` inside the audited runtime; S0 crosses the warmup boundary and S3 uses that pinned sequence as its reference. Scheduler cadence and state-dependent loader branches are exercised by the staged evidence before a full-run claim is considered.

Sampler rule: not applicable because these recipes do not use a timestep or importance sampler. EMA rule: not applicable because the recipes disable EMA. AMP rule: not applicable because the training configs use `precision: 32-true`. Multi-worker loader rule: not applicable because the package DataModule uses `num_workers=0` for the compared streams. Scheduler cadence is applicable and is covered by the S2 and S3 evidence.

The campaign runner records a normalized runtime freeze: sorted `uv pip freeze` output from the selected interpreter with repository and wheel paths replaced by stable package paths and versions. At launch it compares that listing with `models/layoutformerpp/scripts/audit_runtime.freeze`, runs the CUDA smoke first, and then runs the six recorded staged processes; `models/layoutformerpp/scripts/training_parity_record_plugin.py` writes create-only redacted records.

### S5 amendment audit trail

- Evaluation order: S5 ran before evaluation-path parity. The shared checkpoint, original-side outputs, resolved settings, input hashes, and evaluation seed let both paths use the same inputs without rerunning the original evaluator. This replaces the protocol's parity-before-S5 timing. The owner approved completing parity after S5; see the [Issue 265 amendment comment](https://github.com/creative-graphic-design/design-generators/issues/265#issuecomment-6406556602).
- Manifest timing: four saved launch manifests and 36 completed evaluation records retain the run provenance. The manifest is labelled `RECONSTRUCTED AFTER RUN`; this replaces the protocol's requirement to create the manifest before S5. See the [Issue 265 amendment comment](https://github.com/creative-graphic-design/design-generators/issues/265#issuecomment-6406556602).
- Verdict basis: the full-run decision uses per-metric range intersection across three training seeds for every family and metric. Pooled-sample Δ/σ is report-only. This replaces the planned verdict basis; see the [Issue 265 amendment comment](https://github.com/creative-graphic-design/design-generators/issues/265#issuecomment-6406556602).
- Original-evaluator metric rule: for every bitwise-mode evaluation-path comparison, predictions must match bitwise, and each metric's recorded absolute difference must be at most `2^-53` (about `1.11e-16`). The original evaluator's last floating-point digit varies between runs. One unit in the last place (one ULP) is the spacing to the adjacent finite binary64 value; below `0.5`, that spacing is half its size just above `0.5`, while the absolute bound stays fixed at every magnitude. At evaluation seed 500, vendor-record / parity original-side mIoU was unconditional `0.7937300449935831` / `0.793730044993583`, refinement `0.7643045336663307` / `0.7643045336663308`, and completion `0.7334952412958227` / `0.7334952412958228`. The parity-original-minus-vendor deltas were `-1.1102230246251565e-16`, `1.1102230246251565e-16`, and `1.1102230246251565e-16`, respectively. The owner approved this rule for the nondeterministic original evaluator and accepted the unconditional result under it; see the [Issue 265 amendment](https://github.com/creative-graphic-design/design-generators/issues/265).

## Seed Policy

The staged records preserve each run's process boundary, runtime freeze reference, selected-device setting, and test report. A numeric seed match alone is not a seed-paired claim. The RICO25 S5 comparison covers training seeds 0–2 for both systems and one evaluation seed, 500, on the shared TEST split. Evaluation-path parity used each family's training-seed-0 vendor checkpoint on both inference paths with the same resolved evaluator settings, inputs, and evaluation seed. The three-seed ranges assess training variation; the seed-0 parity comparison isolates the evaluation path.

## Evidence Validity

Attempt-4 is the fourth complete execution of the staged S0–S4 checks and supplies the current staged evidence. It superseded attempt-3 because S0, S2, and S3 coverage changed to exercise recipe-specific learning rates and scheduler-state sources across all 12 recipes. The reconstructed manifest records the source revision, the normalized runtime freeze hash, and artifact hashes.

## Stage Evidence

The campaign runner supplies `PYTHONPATH`, `PARITY_REQUIRE`, `LAYOUTFORMERPP_AUDIT_VENV`, `LAYOUTFORMERPP_RECORD_DIR`, `LAYOUTFORMERPP_RUN_STAGE`, `LAYOUTFORMERPP_RUN_REPEAT`, `LAYOUTFORMERPP_RUN_STARTED_AT`, `LAYOUTFORMERPP_SOURCE_COMMIT`, `LAYOUTFORMERPP_RUNTIME_FREEZE_SHA256`, `LAYOUTFORMERPP_RUNTIME_FREEZE_RELATIVE`, `LAYOUTFORMERPP_REPORT_DATA_ROOT`, and `LAYOUTFORMERPP_REPORT_REPO_ROOT` to each Stage Evidence command. The launch shell supplies `CUDA_VISIBLE_DEVICES` and `LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root>`.

| Stage | Command                                                                                                                                                                                                                                                                                                            | Artifact                                                                                                                                                                                                  | Result                                                                                                                                                                                                                         |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s0 -rs -q`                                                  | `.cache/layoutformerpp/post-merge/attempt-4/s0/record.json`                                                                                                                                               | passed: `50` tests; record SHA-256 `5EF17AEB6D59C5FD49712FD7DABC1541223E53D5E345E5D417F530BE48A52DF1`                                                                                                                          |
| S1    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s1 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-4/s1/record.json`                                                                                                                                               | passed: `12` tests; record SHA-256 `7D1E34E502470E1BB392F46A9DBE6C55C05751531D75B639E1161C57718F09C2`                                                                                                                          |
| S2    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s2 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-4/s2/record.json`                                                                                                                                               | passed: `12` tests; record SHA-256 `E6C4ECE7BF530E3FEFCA90D9674A26F5C8005BA4EAB572D890E954B78A8F3D2F`                                                                                                                          |
| S3    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s3 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-4/s3/record.json`                                                                                                                                               | passed: `22` tests; fixed `rtol=1e-4`, `atol=1e-5`; record SHA-256 `8CB72F9040DF5C2A5988880B466CDA16EE440D5F6286941C77E96DC535219728`                                                                                          |
| S4    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" -m pytest -p training_parity_record_plugin models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s4 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-4/s4/record.json`                                                                                                                                               | passed: `13` tests; record SHA-256 `9E4CCBC6C1C73E5E0006746D8168BEAC5FA97F17E2F23A3862347CEF969CDE5F`                                                                                                                          |
| S5    | `python main.py --dataset rico --tasks <family evaluator task> --test --eval_seed 500 --eval_ckpt_tag s5`                                                                                                                                                                                                          | `.cache/layoutformerpp/s5/posthoc-parity/fixed-bound-20261105T012036Z/manifest.json; evaluation-path-parity: .cache/layoutformerpp/s5/posthoc-parity/fixed-bound-20261105T012036Z/label_size/parity.json` | RICO25: training seeds 0–2; all eight metric ranges intersect for all six variants; Δ/σ report-only. Evaluation-path predictions match bitwise for all six; metrics match bitwise for four and meet the `2^-53` bound for two. |

The fixed-bound manifest's `evaluation_path_parity.artifacts` map lists all six parity JSON paths and their SHA-256 values.

The S3 numerical-parity gate checks the short training trajectory at fixed `rtol=1e-4` and `atol=1e-5`. A separate S3 process repeated those tests; it did not run the production-path loss-envelope gate, which compares per-seed loss medians over at least 1,500 updates and three matched seeds. Its record is `.cache/layoutformerpp/post-merge/attempt-4/s3-repeat-2/record.json`, with `22` passed tests and record SHA-256 `047E86969F0EF4291D6BBD99C1276AF3F48CE33A5D937AA253FB74C1FD94532A`. Every process wrote its own record and freeze file, and no divergence was observed in the accepted comparison.

## Reproduction Results

RICO25 has `s5-practical-reproduction` for the six task/configuration variants. The verdict uses intersection of the vendor and package three-seed min/max ranges for each of eight metrics: `alignment`, `bbox_acc`, `fid`, `label_acc`, `mIoU`, `new_overlap`, `overlap`, and `violation_rate`. Per-metric Δ/σ is report-only. PubLayNet S5 has not been run and is not claimed.

| Dataset                 | System           | Status                                                                                               | Seed scope                   | Primary metrics                 | Loss evidence     | Artifact summary                                                               |
| ----------------------- | ---------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------- | ------------------------------- | ----------------- | ------------------------------------------------------------------------------ |
| RICO25 label            | vendor + package | `s5-practical-reproduction`                                                                          | training 0–2; evaluation 500 | eight metrics; ranges intersect | not used for S5   | `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |
| RICO25 label-size       | vendor + package | `s5-practical-reproduction`                                                                          | training 0–2; evaluation 500 | eight metrics; ranges intersect | not used for S5   | `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |
| RICO25 relation         | vendor + package | `s5-practical-reproduction`                                                                          | training 0–2; evaluation 500 | eight metrics; ranges intersect | not used for S5   | `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |
| RICO25 refinement       | vendor + package | `s5-practical-reproduction`                                                                          | training 0–2; evaluation 500 | eight metrics; ranges intersect | not used for S5   | `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |
| RICO25 completion       | vendor + package | `s5-practical-reproduction`                                                                          | training 0–2; evaluation 500 | eight metrics; ranges intersect | not used for S5   | `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |
| RICO25 unconditional    | vendor + package | `s5-practical-reproduction`                                                                          | training 0–2; evaluation 500 | eight metrics; ranges intersect | not used for S5   | `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |
| PubLayNet label         | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not run                      | not evaluated                   | staged S0-S4 only | `.cache/layoutformerpp/post-merge/attempt-4/`                                  |
| PubLayNet label-size    | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not run                      | not evaluated                   | staged S0-S4 only | `.cache/layoutformerpp/post-merge/attempt-4/`                                  |
| PubLayNet relation      | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not run                      | not evaluated                   | staged S0-S4 only | `.cache/layoutformerpp/post-merge/attempt-4/`                                  |
| PubLayNet refinement    | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not run                      | not evaluated                   | staged S0-S4 only | `.cache/layoutformerpp/post-merge/attempt-4/`                                  |
| PubLayNet completion    | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not run                      | not evaluated                   | staged S0-S4 only | `.cache/layoutformerpp/post-merge/attempt-4/`                                  |
| PubLayNet unconditional | vendor + package | `not-yet-run` ([issue 265](https://github.com/creative-graphic-design/design-generators/issues/265)) | not run                      | not evaluated                   | staged S0-S4 only | `.cache/layoutformerpp/post-merge/attempt-4/`                                  |

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

The S5 evaluator is the vendor `main.py` entry point. The campaign ran `python main.py --dataset rico --tasks <family evaluator task> --test --eval_seed 500 --eval_ckpt_tag s5`; each family's resolved arguments are recorded in its evaluation record. The four launch manifests and 36 completed evaluation records are listed with hashes in the reconstructed S5 manifest.

| Dataset | System           | Evaluator                                                | Test split  | Checkpoint-selection rule                                                      | Sample count         |
| ------- | ---------------- | -------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------ | -------------------- |
| RICO25  | vendor + package | `vendor/ms-layout-generation/LayoutFormer++/src/main.py` | `rico/test` | package: minimum aggregate `val_loss` checkpoint; vendor: `best_epoch` pointer | 3,729 per evaluation |

For evaluation-path parity, both paths loaded the same training-seed-0 vendor checkpoint. The campaign manifest `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` is labelled **RECONSTRUCTED AFTER RUN** and derives from four saved campaign launch manifests and 36 completed evaluation records. The follow-up manifest `.cache/layoutformerpp/s5/posthoc-parity/fixed-bound-20261105T012036Z/manifest.json` (SHA-256 `3bef4a62e9f40108095f15be5d24f0fd9fdbabef8ee122c92c2f4951f23d1d64`) is also labelled **RECONSTRUCTED AFTER RUN** and maps the six comparison records to their SHA-256 values. The committed runner maps saved `eval_main.py` wrapper entries to vendor `main.py` and installs the scheduler compatibility shim in-process; it does not depend on an ignored cache-side helper. Each parity record stores the runner SHA-256, dirty-tree flag, and hashes for the train, validation, and TEST input files. Parity followed the completed S5 campaign using retained original-side outputs; the package owner's selected timing is recorded in the [S5 evidence amendment](https://github.com/creative-graphic-design/design-generators/issues/265#issuecomment-6406556602).

### S5 Evaluation-Path Parity

The evaluation-path parity check used the same training-seed-0 vendor checkpoint, resolved family settings, evaluation seed 500, and complete RICO25 TEST split of 3,729 layouts for both paths. The package side called the public `LayoutFormerPPPipeline.from_pretrained` path with its published decoding defaults; the runner does not rebuild or pass vendor constraint logic. For every family, the original predictions came from vendor evaluator commit `1498ff300710b4fc204aece537582d37ca447fc7`, package predictions came from pipeline commit `704ef10a2c16255538a7a2bf84fe36d3f54256a1`, and the comparison runner came from commit `b19067e88120d950341b2e812a704c6fc120906d`. The S5 campaign finished first, and parity followed using the same evaluator settings; the comparison reuses saved predictions and metrics without rerunning inference or the original evaluator.

Predictions match bitwise for all six families. Metrics match bitwise for four families; label_size and refinement each have an mIoU absolute difference of `1.1102230246251565e-16`, equal to `2^-53`, and pass. Every other metric is bitwise equal. An out-of-bounds (OOB) count is the number of predicted boxes outside the original input frame. The fixed-bound records recompare saved predictions and metrics with the runner at `b19067e88120d950341b2e812a704c6fc120906d`; no inference was rerun. In this recheck, unconditional metrics match bitwise. The earlier unconditional record at `.cache/layoutformerpp/s5/posthoc-parity/postmerge-482-setsid-20261104T062649Z/unconditional/parity.json` (SHA-256 `c5627dcaefd4e08c94336d6d2554c71d21769ae525de18a9dacb915e2d8c8f41`) remains unchanged. It was produced before the comparator encoded the fixed bound and also passes: predictions match bitwise, original mIoU is `0.793730044993583`, package mIoU is `0.7937300449935831`, and the absolute difference is `1.1102230246251565e-16`.

| Family        | Prediction commits (vendor / package)                                                   | Comparison runner commit                   | Predictions bitwise equal | Metrics bitwise equal | Metrics accepted | Metric difference (package − original)                                                                                                                                  | OOB original/package | Parity artifact and SHA-256                                                                                                                                           |
| ------------- | --------------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------- | --------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| unconditional | `1498ff300710b4fc204aece537582d37ca447fc7` / `704ef10a2c16255538a7a2bf84fe36d3f54256a1` | `b19067e88120d950341b2e812a704c6fc120906d` | true                      | true                  | yes              | all eight metrics bitwise equal                                                                                                                                         | 9/9                  | `.cache/layoutformerpp/s5/posthoc-parity/fixed-bound-20261105T012036Z/unconditional/parity.json` (`bf442e92471e37dea0b1af7de784018b30d11dbde43c60cae3476b5d8fd6caac`) |
| label         | `1498ff300710b4fc204aece537582d37ca447fc7` / `704ef10a2c16255538a7a2bf84fe36d3f54256a1` | `b19067e88120d950341b2e812a704c6fc120906d` | true                      | true                  | yes              | all eight metrics bitwise equal                                                                                                                                         | 14/14                | `.cache/layoutformerpp/s5/posthoc-parity/fixed-bound-20261105T012036Z/label/parity.json` (`85c60841ff37afdcb7877b202e9cb559c1169b6a106eb15f5c240d79d8938c34`)         |
| label_size    | `1498ff300710b4fc204aece537582d37ca447fc7` / `704ef10a2c16255538a7a2bf84fe36d3f54256a1` | `b19067e88120d950341b2e812a704c6fc120906d` | true                      | false                 | yes              | mIoU original `0.49364049868866455`; package `0.49364049868866466`; delta and absolute difference `1.1102230246251565e-16` (`2^-53`); other seven metrics bitwise equal | 155/155              | `.cache/layoutformerpp/s5/posthoc-parity/fixed-bound-20261105T012036Z/label_size/parity.json` (`351b4006cec8afe685875326d29513b8216c79c3916747ba148ade5690082d91`)    |
| relation      | `1498ff300710b4fc204aece537582d37ca447fc7` / `704ef10a2c16255538a7a2bf84fe36d3f54256a1` | `b19067e88120d950341b2e812a704c6fc120906d` | true                      | true                  | yes              | all eight metrics bitwise equal                                                                                                                                         | 533/533              | `.cache/layoutformerpp/s5/posthoc-parity/fixed-bound-20261105T012036Z/relation/parity.json` (`9d03e03af1b19ab00e7687b3649ce136e9db439c714664ea8ba4df222d0e52ab`)      |
| refinement    | `1498ff300710b4fc204aece537582d37ca447fc7` / `704ef10a2c16255538a7a2bf84fe36d3f54256a1` | `b19067e88120d950341b2e812a704c6fc120906d` | true                      | false                 | yes              | mIoU original `0.7643045336663307`; package `0.7643045336663308`; delta and absolute difference `1.1102230246251565e-16` (`2^-53`); other seven metrics bitwise equal   | 60/60                | `.cache/layoutformerpp/s5/posthoc-parity/fixed-bound-20261105T012036Z/refinement/parity.json` (`b8d85ac65c2688aed57891c3149ce2e3c1d5fcfa40f5a6a0c8661f64003516ba`)    |
| completion    | `1498ff300710b4fc204aece537582d37ca447fc7` / `704ef10a2c16255538a7a2bf84fe36d3f54256a1` | `b19067e88120d950341b2e812a704c6fc120906d` | true                      | true                  | yes              | all eight metrics bitwise equal                                                                                                                                         | 38/38                | `.cache/layoutformerpp/s5/posthoc-parity/fixed-bound-20261105T012036Z/completion/parity.json` (`1d2ca4dd16f76cb116e1b5c086fb79ed1873418708f8137437d1b758f97f5abf`)    |

The runner records the original and package values, package-minus-original delta, absolute difference, and comparison case for every metric. Predictions must remain bitwise equal, and each metric passes when its binary64 values are equal or its absolute difference is at most `2^-53`; larger differences fail.

### Three-seed S5 metric results

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

The manifest is labelled `RECONSTRUCTED AFTER RUN`; it records the four launch manifests, 36 completed evaluation records, six parity artifacts, and per-metric comparison cases:

| Artifact                                                                       | Purpose                                | SHA-256 and verification                                                                                                                                                 |
| ------------------------------------------------------------------------------ | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `.cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` | Reconstructed S5 and parity provenance | `F62956B66DE9E55CE747C7B2F5F95C9B76B983FF0E4811D7C61F18CF736FA9D3`; verify with `sha256sum .cache/layoutformerpp/s5/posthoc-parity/reconstructed-20261104/manifest.json` |

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
FAMILY=label; TRAINING_SEED=0; ATTEMPT=example-s5-attempt; CUDA_VISIBLE_DEVICES=<device> LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> uv run --package layoutformerpp --extra training traingen fit --config "models/layoutformerpp/configs/training/rico25_${FAMILY}.yaml" --trainer.devices=1 --seed_everything="$TRAINING_SEED" --trainer.default_root_dir=".cache/layoutformerpp/s5/current/runs/${ATTEMPT}/package-rico25_${FAMILY}-seed${TRAINING_SEED}"
```

Evaluate a trained checkpoint with the vendor evaluator and its resolved family settings.

```bash
"$LAYOUTFORMERPP_AUDIT_VENV/bin/python" vendor/ms-layout-generation/LayoutFormer++/src/main.py --dataset rico --tasks <family evaluator task> --test --eval_seed 500 --eval_ckpt_tag s5
```

Run evaluation-path parity for one family with training-seed 0. The runner loads each system's resolved settings and evaluation environment from its S5 record, checks the three processed-split hashes before vendor dataset code, and writes a new cache-local record.

```bash
FAMILY=label; PARITY_OUTPUT=.cache/layoutformerpp/s5/posthoc-parity/new-attempt/${FAMILY}; CUDA_VISIBLE_DEVICES=<device> "$LAYOUTFORMERPP_AUDIT_VENV/bin/python" models/layoutformerpp/tests/vendor_parity/run_evaluation_path_parity.py --mode orchestrate --vendor-src vendor/ms-layout-generation/LayoutFormer++/src --data-root "$LAYOUTFORMERPP_PARITY_DATA_ROOT" --family "$FAMILY" --seed 0 --output "$PARITY_OUTPUT" # Run the selected family's comparison.
```

These recorded results need no new campaign. Start a future RICO25 full-run campaign only after the package has complete S0–S4 evidence and evaluation-path parity and the package owner has recorded a compute allocation decision in [Issue 265](https://github.com/creative-graphic-design/design-generators/issues/265).

---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LACE
---

# LACE Training

This document is for a first-time contributor who needs to reproduce the package-local LACE checks. The ordered stages and their gate rules are defined in [the training reproduction protocol](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md). The completed evidence covers PubLayNet and RICO25 through the full TEST evaluation path; full-run statistical training remains S5 and has not been run.

Run commands from the repository root. Generated data, logs, checkpoints, runtime metadata, and evaluation artifacts stay under `.cache/lace/` and are not committed.

## Install

The lockfile environment is used for CPU checks and repository tests. Every `uv` command below is frozen.

```bash
UV_FROZEN=1 uv sync --package lace --extra training --extra vendor
```

The verified CUDA runtime is supplied through `LACE_AUDIT_VENV=<your audit venv>`; select one device with `CUDA_VISIBLE_DEVICES=<gpu>`. No host path is part of the package configuration.

```bash
LACE_AUDIT_VENV="<your audit venv>"
python3.11 -m venv "$LACE_AUDIT_VENV"
"$LACE_AUDIT_VENV/bin/python" -m pip install --upgrade pip
"$LACE_AUDIT_VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple "torch==2.8.0" "torchvision==0.23.0"
"$LACE_AUDIT_VENV/bin/python" -m pip install lightning torch-geometric datasets fsspec matplotlib pycocotools seaborn tqdm
"$LACE_AUDIT_VENV/bin/python" -m pip install --no-deps -e models/lace
```

At run time, the harness records `torch.__version__`, both distributions' `direct_url.json`, their wheel SHA-256 values, and a `pip freeze` artifact and SHA-256. The S0-S3 records were produced with Python 3.11 and torch `2.8.0+cu128`; S4 records the complete runtime metadata in its own artifact.

## Data

The approved sources are [creative-graphic-design/PubLayNet](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) and the `ui-screenshots-and-hierarchies-with-semantic-annotations` configuration of [creative-graphic-design/Rico](https://huggingface.co/datasets/creative-graphic-design/Rico), as specified by [the repository data-source policy](https://github.com/creative-graphic-design/design-generators/blob/main/docs/data-sources.md). The processed files are made with LACE's own `InMemoryDataset` `process()` path from those approved sources and are consumed by every evidence stage:

```text
.cache/lace/data/<dataset>-max25/processed/{train,val,test}.pt
```

The acquisition and processing command is the vendor-faithful dataset constructor, with the raw approved-source files supplied at `LACE_RAW_DATA_ROOT`:

```bash
PYTHONPATH=vendor/lace "$LACE_AUDIT_VENV/bin/python" -c "from util.datasets.publaynet import PubLayNetDataset; from util.datasets.rico import Rico25Dataset; PubLayNetDataset(dir='$LACE_RAW_DATA_ROOT', max_seq_length=25); Rico25Dataset(dir='$LACE_RAW_DATA_ROOT', max_seq_length=25)"
```

The evidence stream is LACE's own `InMemoryDataset` processed stream because the vendor constructors produce the `processed/*.pt` files used by training and evaluation. The approved-source and stream decision is recorded in the [issue 423 amendment](https://github.com/creative-graphic-design/design-generators/issues/423#issuecomment-6035776247).

| Dataset   |                      Processed layouts |                         Processed elements | Evidence stream                                                  |
| --------- | -------------------------------------: | -----------------------------------------: | ---------------------------------------------------------------- |
| PubLayNet | train 315,757; val 16,619; test 11,142 | train 3,033,717; val 159,541; test 119,402 | `.cache/lace/data/publaynet-max25/processed/{train,val,test}.pt` |
| RICO25    |   train 358,510; val 2,109; test 4,218 |   train 3,954,250; val 23,650; test 47,129 | `.cache/lace/data/rico25-max25/processed/{train,val,test}.pt`    |

The S0 source records show identical record IDs and element tuples for PubLayNet and for RICO25 validation and TEST against the LayoutDM compatibility files; RICO25 training is intentionally the LACE tenfold repeated source stream. The direct-processed-path comparison produced identical file, dataset, and sample-ID hashes for all six splits, so the S0-S3 evidence remains valid under the rerun rule. The measured path-proof values are recorded in the regeneration metadata below.

## Configs

Training configs live under `models/lace/configs/training` and use the processed directory directly.

| Config                              | Dataset   | Seed mode     | Purpose                               |
| ----------------------------------- | --------- | ------------- | ------------------------------------- |
| `lace_publaynet.yaml`               | PubLayNet | default       | Package training recipe.              |
| `lace_rico25.yaml`                  | RICO25    | default       | Package training recipe.              |
| `lace_publaynet_deterministic.yaml` | PubLayNet | deterministic | Controlled deterministic diagnostics. |
| `lace_rico25_deterministic.yaml`    | RICO25    | deterministic | Controlled deterministic diagnostics. |

`models/lace/tests/test_training_configs.py` instantiates all four YAML files and asserts their resolved processed roots.

## Scheduler and Recipe Notes

The source recipe uses dimension 1024, four transformer layers, feed-forward dimension 2048, batch size 256, learning rate `1e-5`, 1,000 diffusion timesteps, gradient clipping at norm 1.0, and Adam with betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, and no AMSGrad. There is no scheduler. EMA is registered before the first optimizer step with `mu=0.9999` and updated after `optimizer.step()`.

The adapter executes the vendor iteration in this order: `sample_t`, `forward_t`, loss computation, `optimizer.zero_grad`, backward, `clip_grad_norm_`, `optimizer.step`, and `ema.update`. The package side uses the production Lightning training path.

| Fix                                     | Regression test                                                                                 |
| --------------------------------------- | ----------------------------------------------------------------------------------------------- |
| x0 reconstruction arithmetic            | `training_stage_evidence.py s1` and `s2` exact reconstruction and loss traces for both datasets |
| GPU positional-embedding device         | `test_s0_copied_checkpoint_forward_matches_package` and the CUDA S1 trace                       |
| Scheduler device-local cumulative alpha | S0 schedule assertion and exact S1/S2 trace records                                             |
| EMA operation order                     | `test_training_config_seed_and_ema` and exact S2 post-step EMA records                          |
| Alignment autograd helper order         | `test_training_loss_helpers_cover_reference_operations` and exact S1/S2 loss records            |

The following protocol rules are inapplicable to this evidence: `Scheduler cadence — not applicable: the loop has no scheduler.` `AMP scale state — not applicable: AMP is disabled in the recorded recipe.` `Sampler activation — not applicable: the loop has no timestep or importance sampler beyond the seeded loader.` `Multi-worker loader — not applicable: both systems use DataLoader `num_workers=0`.` `S4/S5 data path — amended by the [issue 423 amendment](https://github.com/creative-graphic-design/design-generators/issues/423#issuecomment-6035776247): use LACE's processed stream from the approved sources.`

## Seed Policy

S0-S3 use the real loader seed 42975 and the natural training random seed 10000 for each system. S4 uses `20260000 + test batch index` for both evaluation paths. These are parity and evaluation seeds, not S5 training-seed evidence.

## Validation Stages

The stage table distinguishes asserted gates from report-only diagnostics. A gate is an asserted pass over the stated population; a report is measured context that is not promoted to exactness.

| Stage | Scope                                                                                                                     | Gate or report                                                                                                                     |
| ----- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static topology, independent initialization, optimizer, EMA, and source-stream compatibility for both datasets            | Gate: package/vendor construction and the measured data policy agree.                                                              |
| S1    | One real train batch of 256 layouts per dataset before optimizer mutation                                                 | Gate: trace equality over the batch.                                                                                               |
| S2    | One real train batch and one optimizer step under natural, strict deterministic, warn-only, math-SDPA, and CPU conditions | Gate: gradients, clipping, optimizer state, parameters, EMA, and learning rate are equal; warning text is a captured measurement.  |
| S3    | 300 real shuffled train batches at batch size 256 per dataset, plus two repeat runs per system                            | Gate: natural traces are bitwise exact for both datasets; synchronized layer not needed. Repeat envelopes are report-only context. |
| S4    | Every train, validation, and TEST layout in both processed streams; full TEST evaluation through both entry points        | Gate: full loader stream and full TEST path equality.                                                                              |
| S5    | Full-run statistical training comparison                                                                                  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)`.                                           |

The S0 topology guard is also collected by the regular member-test command as `test_s0_copied_checkpoint_forward_matches_package` in `models/lace/tests/test_training_lace.py`; it copies each authors' checkpoint into the vendor and package models, checks parameter and state-key parity, and asserts bitwise forward equality on both claimed datasets.

## Stage Evidence

All rows below cite the exact command family that produced the cited artifact. The source commits are copied from the machine-written JSON records.

| Stage | Command                                                                                                                                                                                                                                                                                                                                     | Artifact                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Source commit                              | Result                                                                                                                              |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `LACE_AUDIT_VENV="<your audit venv>" CUDA_VISIBLE_DEVICES="<gpu>" "$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s0 --dataset <publaynet\|rico25> --lace-data-root .cache/lace/data --layoutdm-data-root <layoutdm-data-root> --output-root .cache/lace/stage-evidence/2c4f37691c9a-s0-<dataset>` | PubLayNet `.cache/lace/stage-evidence/2c4f37691c9a-s0-publaynet/s0-static/summary.json` — SHA-256 `ece2db54b85dcfd3c61f0d0bc77093cef24b388eb999c3aff341bad45ad90713`; RICO25 `.cache/lace/stage-evidence/2c4f37691c9a-s0-rico25/s0-static/summary.json` — SHA-256 `de09d8fde75b4a57052f81a6f66a453e247dde2f8e6418f95f5d43b410bdc880`                                                                                                                                                                                                                                           | `2c4f37691c9ae663fcf6e000eb977cd688ac8e6a` | Gate passed for both datasets.                                                                                                      |
| S1    | `LACE_AUDIT_VENV="<your audit venv>" CUDA_VISIBLE_DEVICES="<gpu>" "$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s1 --dataset <publaynet\|rico25> --device cuda --batch-size 256 --lace-data-root .cache/lace/data --output-root .cache/lace/stage-evidence/2c4f37691c9a-s1-<dataset>`            | PubLayNet `.cache/lace/stage-evidence/2c4f37691c9a-s1-publaynet/s1-fixed-batch/summary.json` — SHA-256 `a28402b0d08e24c7c40b510ac76b7cb6cebb8cdc96a18da29b0d7b57285f638d`; RICO25 `.cache/lace/stage-evidence/2c4f37691c9a-s1-rico25/s1-fixed-batch/summary.json` — SHA-256 `78f8e0384dde11cd7f17dd72b4e0e14ff9eb1015d44783de0f08d8c7ec54bb7b`                                                                                                                                                                                                                                 | `2c4f37691c9ae663fcf6e000eb977cd688ac8e6a` | Gate passed: exact on both real 256-layout batches.                                                                                 |
| S2    | See the five exact condition commands below.                                                                                                                                                                                                                                                                                                | Natural summaries: PubLayNet `.cache/lace/stage-evidence/2c4f37691c9a-s2-natural-publaynet/s2-optimizer-step/summary.json` — SHA-256 `bbe6e9b8d2e66596b377e80c34b4fe17be0252600b28580bed4963416a69630d`; RICO25 `.cache/lace/stage-evidence/2c4f37691c9a-s2-natural-rico25/s2-optimizer-step/summary.json` — SHA-256 `8fc74674c3b5ae3c3d94947244b56f8a3871fd81ef38149ed3db3f0d8a188a5c`; all condition artifacts are listed below.                                                                                                                                             | `2c4f37691c9ae663fcf6e000eb977cd688ac8e6a` | Gate passed for both datasets under all five conditions.                                                                            |
| S3    | See the exact command below, once per dataset.                                                                                                                                                                                                                                                                                              | PubLayNet `.cache/lace/stage-evidence/2c4f37691c9a-s3-publaynet/s3-lockstep/summary.json` — SHA-256 `637ad04ec625ffe6ccf78acb6e4647ccfb01bf777d3cabeaf2aa20b73dbff6bc`; RICO25 `.cache/lace/stage-evidence/2c4f37691c9a-s3-rico25/s3-lockstep/summary.json` — SHA-256 `8d38c6ce5056917579c33010861c6fc9f7deebfda957c74b24845ce651825a75`                                                                                                                                                                                                                                       | `2c4f37691c9ae663fcf6e000eb977cd688ac8e6a` | Gate passed: natural traces are bitwise exact for both datasets over 300 steps; synchronized layer not needed for the parity claim. |
| S4    | See the full command below.                                                                                                                                                                                                                                                                                                                 | `.cache/lace/stage-evidence/532b2312f13dfd4d-s4-full/s4-loader-evaluation/summary.json` — SHA-256 `945e74cd7fd59cf65de559345a14482b1f4ae847de9131903ff155aaa3bbb67f`; PubLayNet `.cache/lace/stage-evidence/532b2312f13dfd4d-s4-full/s4-loader-evaluation/publaynet-evaluation/evaluation.json` — SHA-256 `7fe0402c87c14de1edf6629e6ca6ade80321e46cd500b0647bef8442a5949596`; RICO25 `.cache/lace/stage-evidence/532b2312f13dfd4d-s4-full/s4-loader-evaluation/rico25-evaluation/evaluation.json` — SHA-256 `5f16d5b8ddfadd2e12e93733a2ccde639609b7c47f2d752cfda98a1afc8dab2d` | `532b2312f13dfd4d9821d623d311aef440b6676f` | Gate passed: all streams use vendor/package `num_workers=0` and match; full TEST evaluation is exact through both entry points.     |
| S5    | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)`                                                                                                                                                                                                                                                     | `.cache/lace/full-run/publaynet/manifest.json; evaluation-path-parity: https://github.com/creative-graphic-design/design-generators/issues/423`                                                                                                                                                                                                                                                                                                                                                                                                                                | —                                          | Not run.                                                                                                                            |

S2 condition artifacts:

| Condition            | PubLayNet artifact and SHA-256                                                                                                                                           | RICO25 artifact and SHA-256                                                                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| strict deterministic | `.cache/lace/stage-evidence/2c4f37691c9a-s2-deterministic-publaynet/s2-optimizer-step/summary.json` — `156ec41da2249f4b974c446bfcae4007799a4a1b30a1020d32df2674add6332e` | `.cache/lace/stage-evidence/2c4f37691c9a-s2-deterministic-rico25/s2-optimizer-step/summary.json` — `81eb9f9a297e7445d8c7d5e46af744bef0319350d947aad8c373370d87a16c78` |
| warn-only            | `.cache/lace/stage-evidence/2c4f37691c9a-s2-warn-only-publaynet/s2-optimizer-step/summary.json` — `59f7b08515a231006db1ecbfe46af027842a870a8828f691ac2abb8f721e9f24`     | `.cache/lace/stage-evidence/2c4f37691c9a-s2-warn-only-rico25/s2-optimizer-step/summary.json` — `e1094d7a7ab4bbaa85f28914599c3439c4aa6791a83a0fc0e6d575f656a8e3b0`     |
| math-SDPA            | `.cache/lace/stage-evidence/2c4f37691c9a-s2-math-sdpa-publaynet/s2-optimizer-step/summary.json` — `7b4e04ed3983302695e29810d4b5e042eab5e9a9d4928ac7e6aaf445f467d440`     | `.cache/lace/stage-evidence/2c4f37691c9a-s2-math-sdpa-rico25/s2-optimizer-step/summary.json` — `f21435df5bc9943151af92fd8a07155c2b703a2290ce6f1f4bc392dfca4d2605`     |
| CPU                  | `.cache/lace/stage-evidence/2c4f37691c9a-s2-cpu-publaynet/s2-optimizer-step/summary.json` — `02f35c4d14bd287df205f3979501fea0160e3cc8b69867ea62ec8bba748d9179`           | `.cache/lace/stage-evidence/2c4f37691c9a-s2-cpu-rico25/s2-optimizer-step/summary.json` — `5a64e853fff99a3bc2710dd856d0f45c7213a4ab31c6592b115cc11bc24458cd`           |

Run S2 with the conditions recorded by the machine-written artifacts:

```bash
LACE_AUDIT_VENV="<your audit venv>"
CUDA_VISIBLE_DEVICES="<gpu>" "$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s2 --dataset <publaynet|rico25> --device cuda --batch-size 256 --lace-data-root .cache/lace/data --output-root .cache/lace/stage-evidence/2c4f37691c9a-s2-natural-<dataset>
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="<gpu>" "$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s2 --dataset <publaynet|rico25> --device cuda --batch-size 256 --lace-data-root .cache/lace/data --deterministic-algorithms --output-root .cache/lace/stage-evidence/2c4f37691c9a-s2-deterministic-<dataset>
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="<gpu>" "$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s2 --dataset <publaynet|rico25> --device cuda --batch-size 256 --lace-data-root .cache/lace/data --deterministic-algorithms --warn-only --output-root .cache/lace/stage-evidence/2c4f37691c9a-s2-warn-only-<dataset>
CUDA_VISIBLE_DEVICES="<gpu>" "$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s2 --dataset <publaynet|rico25> --device cuda --batch-size 256 --lace-data-root .cache/lace/data --sdpa-math --output-root .cache/lace/stage-evidence/2c4f37691c9a-s2-math-sdpa-<dataset>
"$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s2 --dataset <publaynet|rico25> --device cpu --batch-size 256 --lace-data-root .cache/lace/data --output-root .cache/lace/stage-evidence/2c4f37691c9a-s2-cpu-<dataset>
```

Run the 300-step stage and the full loader/evaluation stage on the selected GPU:

```bash
CUDA_VISIBLE_DEVICES="<gpu>" "$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s3 --dataset <publaynet|rico25> --device cuda --steps 300 --batch-size 256 --lace-data-root .cache/lace/data --output-root .cache/lace/stage-evidence/2c4f37691c9a-s3-<dataset>
CUDA_VISIBLE_DEVICES="<gpu>" "$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s4 --device cuda --evaluation-batch-size 256 --lace-data-root .cache/lace/data --checkpoint-root .cache/lace/original/model --fid-root .cache/lace/fidroot --output-root .cache/lace/stage-evidence/532b2312f13dfd4d-s4-full
```

S2 compares raw gradient, clipped-gradient, optimizer-state, parameter, EMA, and learning-rate values like-for-like. The warn-only record captures the memory-efficient attention backward warning; the five condition records remain separate gate results.

## Reproduction Results

The package and original implementation agree exactly for the asserted S0-S4 populations on PubLayNet and RICO25. S3 is a 300-step parity check, not statistical training evidence. S4 evaluates all 11,142 PubLayNet TEST layouts and all 4,218 RICO25 TEST layouts with the authors' checkpoints through `vendor/lace/test.py::test_layout_cond` and `models/lace/src/lace/pipeline_lace.py::__call__`.

The full loader gate checked PubLayNet train/val/test populations of 315,757/16,619/11,142 layouts in 1,234/65/44 batches and RICO25 train/val/test populations of 358,510/2,109/4,218 layouts in 1,401/9/17 batches. Train used seeded shuffle with seed 42975; validation and TEST used the vendor-faithful unshuffled loader.

| Dataset   | System | Status                                                                                  | Seed scope                       | Primary metrics                                                                           | Loss evidence                                      | Artifact summary                                                                                                |
| --------- | ------ | --------------------------------------------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| PubLayNet | both   | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)` | parity and evaluation seeds only | alignment 0.1216632277; overlap 4.4502921104; FID 4.8615175926; maximum IoU 0.3281917111  | S1, S2, and S3 exact over their stated populations | `.cache/lace/stage-evidence/532b2312f13dfd4d-s4-full/s4-loader-evaluation/publaynet-evaluation/evaluation.json` |
| RICO25    | both   | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)` | parity and evaluation seeds only | alignment 0.1111927703; overlap 83.5333480835; FID 3.3846049958; maximum IoU 0.3358800030 | S1, S2, and S3 exact over their stated populations | `.cache/lace/stage-evidence/532b2312f13dfd4d-s4-full/s4-loader-evaluation/rico25-evaluation/evaluation.json`    |
| RICO13    | both   | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)` | no claimed seed scope            | no RICO13 authors' checkpoint in the approved archive                                     | no S0-S4 claim                                     | `https://github.com/creative-graphic-design/design-generators/issues/423`                                       |

S5 remains `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)`; the status above describes the completed practical S0-S4 evaluation path, not a full-run statistical claim.

### Comparison Scope

| Dataset   | System | Evaluator                                                                                         | Test split       | Checkpoint-selection rule    |                       Sample count |
| --------- | ------ | ------------------------------------------------------------------------------------------------- | ---------------- | ---------------------------- | ---------------------------------: |
| PubLayNet | both   | `vendor/lace/test.py::test_layout_cond` and `LacePipeline.__call__`, with vendor metric functions | LACE TEST stream | authors' `publaynet_best.pt` | 11,142 layouts per evaluation seed |
| RICO25    | both   | `vendor/lace/test.py::test_layout_cond` and `LacePipeline.__call__`, with vendor metric functions | LACE TEST stream | authors' `rico25_best.pt`    |  4,218 layouts per evaluation seed |
| RICO13    | both   | not applicable; no approved authors' RICO13 checkpoint                                            | not applicable   | not applicable               |   0 layouts; outside this evidence |

The captured vendor `real_layout` is the evaluation input. The package receives the same captured layout array; the harness does not re-encode a second source array. The records hash each package and vendor input artifact, each loaded package state dict, each prediction artifact, the FID evaluator, FIDNetV3 weights, and feature cache. Vendor `init_dataset`, `test_fid_feat`, and `load_fidnet_v3` are recorded as explicit substitutions in the evaluation records.

## Regeneration Metadata

The authors' checkpoints came from [the LACE model archive](https://huggingface.co/datasets/puar-playground/LACE/resolve/main/model.tar.gz). The current files are `.cache/lace/original/model/publaynet_best.pt` (SHA-256 `d13ea9a64d913d910a35db6204a4be7b115101ffea54874683ef8c9f98607ff4`) and `.cache/lace/original/model/rico25_best.pt` (SHA-256 `9c2f2198dbab7d530363a413d98efa779bc5a1ebcdf55d1c1fc64175361abe93`).

FID provenance is measured from `.cache/lace/fidroot/provenance.json`: the evaluator is `fid/model.py`, source commit `873b5eebe4c61862e5c08a10859accf65a168dfd`, source URL [at the measured source path](https://github.com/CyberAgentAILab/layout-dm/blob/873b5eebe4c61862e5c08a10859accf65a168dfd/src/trainer/trainer/fid/model.py), and measured file SHA-256 `5df4cf82a869167d8cb6ab19470e58ecf42a576c0b8ae134c424cbd8505fdced`. FIDNetV3 came from the recorded LayoutDM release command and archive SHA-256 `357a0b8cd305793164ae4e9da1033ac1b687bfd46a53716673b32c83280c443b`. The measured FIDNetV3 and feature-cache hashes are PubLayNet weights `ff7208304e5c5f673ddd7cd5d73f85a0982df97a70713016f1b034a9c972d6fd`, features `ae9f84bb3c87eb5a4d94c4d0b3a0a32a49a3360fcc1e99d30ac35815fc3b027d`; RICO25 weights `3e99f113bdea8f6e4623103bea88aff6217f322eef3deba5a787f3acd19e3296`, features `61fbdfc6e2255eb7a36441509e165858932931866f2193834cfa6a6e8741b975`. The harness records these values and the explicit evaluator substitutions in each evaluation JSON.

The direct-path proof measured both loader roots against the same approved `processed/*.pt` files:

| Dataset   | Split | Layouts |  Elements | Dataset hash                                                       | Sample-ID hash                                                     |
| --------- | ----- | ------: | --------: | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| PubLayNet | train | 315,757 | 3,033,717 | `5b9912664b5b276c712a4561efdffd552b7131fe09f6ac289956f4380d9d0c73` | `fafc2cc2cccbcf5ed22a3ba06264b0277ef162f85d365782e9810b6351a3ad21` |
| PubLayNet | val   |  16,619 |   159,541 | `d88cec15caa48602b7fcecc1a677df89034d5251c9d2a704d9cb3661be2bacb5` | `c7be3a18f32b04accd046b72999faa2b69e47119d338fab2082750e8b3b99700` |
| PubLayNet | test  |  11,142 |   119,402 | `a3f5f6032d0b2ab87d045307e6c76c8392285a57d4aaa0e7e4ee65ea3424d828` | `6d3fd886468786e86c7ca981bbdeb88666b354b200c30f3c635114e627fc8c8e` |
| RICO25    | train | 358,510 | 3,954,250 | `aae8e685afa6f0551c091749b02f5d8defcc5df8803f67dc8b7d13751842d381` | `d45131f61c51b836d958f3065d06fd9c8a1490bbb20e000c84fcf3cc6ad72b53` |
| RICO25    | val   |   2,109 |    23,650 | `9807eb04353bcde676a9d6d80107a81a4b458663dbf2de46d878929058687a80` | `a17d601b3f000c1cb94f297704241c0d98c3a9162954da1135db5efc53308b8a` |
| RICO25    | test  |   4,218 |    47,129 | `73eedb4859cdaa121596e952f06830eb7744d6286af889708457bc56ed556d16` | `b14f37a4dfcc1dd482edb9500aceaefd6472a02dd283f79942752e1f8314c45d` |

Each pair had `byte_identical=true`; the file SHA-256 is recorded by the measurement. Runtime freeze artifacts and wheel provenance are recorded in the stage JSON files.

```text
.cache/lace/data/
.cache/lace/original/model/
.cache/lace/fidroot/
.cache/lace/stage-evidence/
```

## Training Commands

The repository training command is `traingen fit`; `python -m lightning.pytorch.cli` is not a training entry point.

```bash
UV_FROZEN=1 uv run --package lace --extra training traingen fit --config models/lace/configs/training/lace_publaynet.yaml --trainer.devices=1
UV_FROZEN=1 uv run --package lace --extra training traingen fit --config models/lace/configs/training/lace_rico25.yaml --trainer.devices=1
UV_FROZEN=1 uv run --package lace scripts/run_member_tests.sh models/lace
```

S5 training is intentionally not launched by this document.

---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LACE
---

# LACE Training

Package-local LACE reproduction passes the asserted S0-S4 gates for PubLayNet and RICO25. The evidence covers exact initialization, fixed-batch and optimizer-step traces, 300 natural training steps, the bounded production-wiring run, the complete processed loader streams, and the full TEST evaluation path. S5 full-run statistical reproduction is not claimed.

Run commands from the repository root. Generated data, logs, checkpoints, runtime metadata, and evaluation artifacts stay under `.cache/lace/` and are not committed.

## Install

```bash
UV_FROZEN=1 uv sync --package lace --extra training --extra vendor
```

The verified CUDA runtime is selected through `LACE_AUDIT_VENV=<your audit venv>` and one selected device through `CUDA_VISIBLE_DEVICES=<gpu>`. The recorded audited freeze is `.cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-s4-fid-refresh/s4-loader-evaluation/runtime/pip-freeze.txt` with SHA-256 `90bace448bc88abf9a1683fd625e83f73d8f19239d2f2f8eec2348a331418332`. It contains editable `lace`, `laygen`, `traingen`, and `traingen-parity` installs, torch `2.8.0+cu128`, and torchvision `0.23.0+cu128`. The same freeze also contains editable `ds_gan` and `posgen` from another worktree; they are present in the freeze, not imported by LACE.

Recreate the audited runtime with the commands used for the fresh proof. The frozen distribution input includes `imageio` and the CUDA `nvidia-*` distributions; the hash-checked local-wheel input enforces the recorded torch and torchvision wheels. The recorded wheel SHA-256 values are torch `039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed` and torchvision `93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb`.

```bash
export LACE_AUDIT_VENV="<your audit venv>"
UV_FROZEN=1 uv venv "$LACE_AUDIT_VENV" --python 3.11
UV_FROZEN=1 uv pip sync --python "$LACE_AUDIT_VENV/bin/python" .cache/lace/runtime/recorded-runtime-requirements.txt
UV_FROZEN=1 uv pip install --python "$LACE_AUDIT_VENV/bin/python" --require-hashes --no-deps -r .cache/lace/runtime/wheel-hash-requirements.txt
UV_FROZEN=1 uv pip install --python "$LACE_AUDIT_VENV/bin/python" --no-deps -e models/lace -e lib/laygen -e lib/traingen -e lib/traingen-parity
UV_FROZEN=1 uv pip freeze --python "$LACE_AUDIT_VENV/bin/python" > .cache/lace/runtime/pip-freeze.txt
```

The fresh proof used this recipe, then ran both S0 and S1 on CPU. Its freeze SHA-256 is `90bace448bc88abf9a1683fd625e83f73d8f19239d2f2f8eec2348a331418332`; the distribution diff against the cited freeze is empty. The freeze also contains editable `ds_gan` and `posgen` from another worktree; they are present in the freeze, not imported by LACE.

```bash
export LACE_AUDIT_VENV="<your audit venv>"
export CUDA_VISIBLE_DEVICES="<gpu>"
```

## Data

The claimed datasets are [creative-graphic-design/PubLayNet](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) and the `ui-screenshots-and-hierarchies-with-semantic-annotations` configuration of [creative-graphic-design/Rico](https://huggingface.co/datasets/creative-graphic-design/Rico). The package and evidence harness consume LACE's processed stream:

```text
.cache/lace/data/<dataset>-max25/processed/{train,val,test}.pt
```

| Dataset   | Source                              | Config or path                                                   |
| --------- | ----------------------------------- | ---------------------------------------------------------------- |
| PubLayNet | `creative-graphic-design/PubLayNet` | `.cache/lace/data/publaynet-max25/processed/{train,val,test}.pt` |
| RICO25    | `creative-graphic-design/Rico`      | `.cache/lace/data/rico25-max25/processed/{train,val,test}.pt`    |

The approved source and processed-stream amendment is recorded in [issue 423](https://github.com/creative-graphic-design/design-generators/issues/423#issuecomment-6035776247). The machine-written dataset identity proof `.cache/lace/stage-evidence/532b2312f13dfd4d-s4-full/dataset-path-identity.json` has SHA-256 `d6cf7b0f68ac34a47e142a427c2f227ed07ad32d4a0dc429292a8ae198fca2ba` and records equal file, loaded-dataset, and sample-ID hashes for all six split pairs.

The proof compares the legacy data-root form used at source commit `2c4f37691c9ae663fcf6e000eb977cd688ac8e6a` with the direct processed-root form introduced at source commit `532b2312f13dfd4d9821d623d311aef440b6676f`. It records `file_sha256_equal=true`, `loaded_dataset_equal=true`, and `sample_ids_equal=true` for every split; the loaded-dataset and sample-ID hashes are:

| Dataset   | Split | Layouts |  Elements | Loaded-dataset hash                                                | Sample-ID hash                                                     |
| --------- | ----- | ------: | --------: | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| PubLayNet | train | 315,757 | 3,033,717 | `5bfe22aed397f64222d0f549f7387ac04683ae897355a02ca5ee0e0a7ca40f96` | `fafc2cc2cccbcf5ed22a3ba06264b0277ef162f85d365782e9810b6351a3ad21` |
| PubLayNet | val   |  16,619 |   159,541 | `f01c6859c92e392d30d47a19f8d8a84724c43dd70431d9040b84f765b2d90fb0` | `c7be3a18f32b04accd046b72999faa2b69e47119d338fab2082750e8b3b99700` |
| PubLayNet | test  |  11,142 |   119,402 | `f82ccf6279561b4d2d4d02257a4ab1c243dde47a402017aa69a0eb6218bd0a90` | `6d3fd886468786e86c7ca981bbdeb88666b354b200c30f3c635114e627fc8c8e` |
| RICO25    | train | 358,510 | 3,954,250 | `57d3b859a9d3136114468e171bfee438ef858476aa17b7bd034804362a4356a1` | `d45131f61c51b836d958f3065d06fd9c8a1490bbb20e000c84fcf3cc6ad72b53` |
| RICO25    | val   |   2,109 |    23,650 | `406078b1390fb983859e7afae76fe7fcab9f5f4db8113804f6a76b6cbdb56a18` | `a17d601b3f000c1cb94f297704241c0d98c3a9162954da1135db5efc53308b8a` |
| RICO25    | test  |   4,218 |    47,129 | `b27f2555ac6864629f004f3ccb14bed0feed3dd29f65bb04c7bf65dc7937fdd9` | `b14f37a4dfcc1dd482edb9500aceaefd6472a02dd283f79942752e1f8314c45d` |

This proof is distinct from the S0 LACE/LayoutDM compatibility comparison: for the five matching splits, serialized files differ while loaded element tuples and record IDs match; RICO25 train is intentionally a tenfold LACE stream and is excluded from that cross-source comparison.

## Configs

Training configs live under `models/lace/configs/training`.

| Config                              | Dataset   | Seed mode     | Purpose                  |
| ----------------------------------- | --------- | ------------- | ------------------------ |
| `lace_publaynet.yaml`               | PubLayNet | default       | Package training recipe. |
| `lace_rico25.yaml`                  | RICO25    | default       | Package training recipe. |
| `lace_publaynet_deterministic.yaml` | PubLayNet | deterministic | Controlled evidence run. |
| `lace_rico25_deterministic.yaml`    | RICO25    | deterministic | Controlled evidence run. |

## Scheduler and Recipe Notes

The recipe uses batch size 256, learning rate `1e-5`, Adam with betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, gradient clipping at norm 1.0, and EMA `mu=0.9999` updated after `optimizer.step()`. The package configs and shipped vendor loader use `num_workers=4` with pinned memory; package S3 runs use the shipped 4 workers, while the vendor side and S1, S2, and S4 evidence use `num_workers=0` and `pin_memory=False`.

- Scheduler cadence — not applicable: the LACE optimizer has no learning-rate scheduler.
- AMP scale state — not applicable: AMP is disabled in the recorded recipe.
- S4/S5 data path — amended by [issue 423](https://github.com/creative-graphic-design/design-generators/issues/423#issuecomment-6035776247): use LACE's processed stream from the approved sources.
- Sampler activation — applicable: both loops execute the vendor timestep-sampling branch over the configured training range.
- EMA activation — applicable: EMA is registered before the first optimizer step and updated after the optimizer step.
- Multi-worker loader — applicable: package S3 uses the shipped 4 workers; the vendor side and S1, S2, and S4 use 0 workers and record that deviation.

The pre-evidence fixes are recorded as fixes with regression tests:

| Fix                                     | Regression test or evidence gate                                                   |
| --------------------------------------- | ---------------------------------------------------------------------------------- |
| x0 reconstruction arithmetic            | Fixed-batch and optimizer-step exact traces in the stage-evidence harness.         |
| GPU positional-embedding device         | `test_s0_copied_checkpoint_forward_matches_package` and the CUDA S3 trace.         |
| Scheduler device-local cumulative alpha | `test_training_cumulative_alphas_use_requested_device` and the CUDA S3 trace.      |
| EMA operation order                     | `test_training_config_seed_and_ema` and the optimizer-step trace.                  |
| Alignment autograd helper order         | `test_training_loss_helpers_cover_reference_operations` and the fixed-batch trace. |

## Seed Policy

The step-level and 300-step evidence uses loader seed 42975 and natural training seed 10000 for both systems. Evaluation uses the recorded TEST sampling seed formula `20260000 + test batch index`. These are parity and evaluation scopes, not S5 training-seed evidence.

## Validation Stages

| Stage | Scope                                                                                                        | Gate status                                                                                                                                  |
| ----- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static topology, initialization, optimizer/EMA state, and source-stream compatibility for both datasets.     | Passed for both datasets at the recorded head.                                                                                               |
| S1    | One real 256-layout train batch before optimizer mutation for both datasets.                                 | Exact for both datasets.                                                                                                                     |
| S2    | One real 256-layout batch and one optimizer step under the refreshed CPU condition for both datasets.        | Exact for both datasets; the record reports zero gradient, clipped-gradient, EMA, optimizer-state, parameter, and learning-rate differences. |
| S3    | 300 real shuffled train batches at batch size 256, two runs per system, plus the production-wiring boundary. | Natural traces bitwise exact for both datasets; synchronized layer not needed; production wiring passed separately.                          |
| S4    | Every train, validation, and TEST layout in both processed streams and the full TEST evaluation path.        | Loader and evaluation gates passed for both datasets.                                                                                        |
| S5    | Full-run statistical comparison.                                                                             | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)`.                                                     |

## Stage Evidence

All artifact hashes below are SHA-256 values of the cited machine-written files. The final-head S0, S1, production-wiring S3, and S4 records use source commit `314a625da0dffe43ea6d60d91461cf4edb80c88c`; the S2 and natural S3 records use source commit `0b3576960fa9c227ce5499754407c4d09616ab51` and stand under the scoped rerun rule because the later harness changes only callback attachment and evidence summarization. The vendor source commit in every record is `3df36879a1e80cce58affa4aadeeb768f676c7f1`.

| Stage                | Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Artifact and SHA-256                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Result                                                                                                                                                                                                                                                                                                                                                           |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0                   | `"$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s0 --dataset <publaynet\|rico25> --lace-data-root .cache/lace/data --layoutdm-data-root <layoutdm-data-root> --output-root .cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-fresh-s0-<dataset>`                                                                                                                                                                                         | PubLayNet `.cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-fresh-s0-publaynet/s0-static/summary.json` — `002d68e1c8eec8dff5a69501b51f31240af09ce80c3b8f887ab0f0a8c84647e0`; RICO25 `.cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-fresh-s0-rico25/s0-static/summary.json` — `f2d59fac8286db2107905368531d19b90fd9ad8ca8398a466c2e2325145f7e25a`                                                                                                                                                                                                                                                                                                                                                                              | Machine gate passed for both datasets, including `source_stream_compatibility_equal=true`.                                                                                                                                                                                                                                                                       |
| S1                   | `"$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s1 --dataset <publaynet\|rico25> --device cpu --batch-size 256 --lace-data-root .cache/lace/data --output-root .cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-fresh-s1-<dataset>`                                                                                                                                                                                                     | PubLayNet `.cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-fresh-s1-publaynet/s1-fixed-batch/summary.json` — `0c7f0d2626125d94544affaf7e56f8fb592a4a29c3f1b95a649416b7d6478d68`; RICO25 `.cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-fresh-s1-rico25/s1-fixed-batch/summary.json` — `7ffb7b137b0c844da0c5400a97df6ea637103435bf794c95cf8f101a7cb87986`                                                                                                                                                                                                                                                                                                                                                                     | Exact for both real batches.                                                                                                                                                                                                                                                                                                                                     |
| S2                   | `"$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s2 --dataset <publaynet\|rico25> --device cpu --batch-size 256 --lace-data-root .cache/lace/data --output-root .cache/lace/stage-evidence/0b3576960fa9c227ce5499754407c4d09616ab51-s2-natural-<dataset>`                                                                                                                                                                                                   | PubLayNet `.cache/lace/stage-evidence/0b3576960fa9c227ce5499754407c4d09616ab51-s2-natural-publaynet/s2-optimizer-step/summary.json` — `f094519ee852e9640611ac4541f653bd000a1a7eed12499f5cf15340a327cf39`; RICO25 `.cache/lace/stage-evidence/0b3576960fa9c227ce5499754407c4d09616ab51-s2-natural-rico25/s2-optimizer-step/summary.json` — `fc515f2b4d1a482af4646fe4efe29c8d4700f4b3b88af2a3b17d28ee5ee16900`                                                                                                                                                                                                                                                                                                                                                           | Exact for both datasets; all recorded norm and state differences are `0.0`.                                                                                                                                                                                                                                                                                      |
| S3 natural           | `CUDA_VISIBLE_DEVICES="<gpu>" uv run --package lace --extra training traingen fit --config models/lace/configs/training/lace_<dataset>.yaml --seed_everything 123 --model.init_args.seed_mode deterministic --model.init_args.seed 10000 --data.init_args.loader_seed 42975 --data.init_args.processed_data_dir .cache/lace/data/<dataset>-max25/processed --trainer.max_steps 300 --trainer.limit_train_batches 300 --trainer.default_root_dir <natural-root>`                                      | PubLayNet summary `.cache/lace/stage-evidence/0b3576960fa9c227ce5499754407c4d09616ab51-s3-publaynet-lace-audit/s3-lockstep/summary.json` — `b38dfdad5545644890ac976308023081872c0a1cbb8f415f697ef173441bb54f`; natural `natural.jsonl` — `1fcadb3a5408d7e03f03f1c82630cefc627e2cc67c96098765c3dc49f1da4e38`; RICO25 summary `.cache/lace/stage-evidence/0b3576960fa9c227ce5499754407c4d09616ab51-s3-rico25-lace-audit/s3-lockstep/summary.json` — `188f505cc4922fa545554ef2576f61cd988b585f5b786578116d30a15bcbe7b1`; natural `natural.jsonl` — `244d9d1f7de560e991bcf7285cc82d47dffcc20822820349308a302378bd681a`                                                                                                                                                     | Natural traces are bitwise exact for both datasets over 300 steps; loss and every shared norm summary difference are `0.0`; synchronized layer not needed.                                                                                                                                                                                                       |
| S3 synchronized      | Same S3 command; retained as a diagnostic layer.                                                                                                                                                                                                                                                                                                                                                                                                                                                     | PubLayNet `synchronized.jsonl` — `2ada6ec0358f8debe3c47861b4c093fc77ffad4d06cb1dfff73f14f19ac8270a`; RICO25 `synchronized.jsonl` — `244d330e54a17bd03078cfad64a5ad7797f3258b17f6f1e5d037cc3128c9c7d4`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Diagnostic pre-sync gate passed with zero state differences; no post-sync exactness claim and no synchronized layer needed for the natural gate.                                                                                                                                                                                                                 |
| S3 production wiring | `uv run --package lace --extra training traingen fit --config models/lace/configs/training/lace_<dataset>.yaml --seed_everything 123 --model.init_args.seed_mode deterministic --model.init_args.seed 10000 --data.init_args.loader_seed 42975 --data.init_args.processed_data_dir .cache/lace/data/<dataset>-max25/processed --trainer.max_steps 2 --trainer.limit_train_batches 2 --trainer.default_root_dir <production-root> --trainer.callbacks+=lace.training.trace.LaceTrainingTraceCallback` | PubLayNet `.cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-s3-wiring-publaynet/s3-lockstep/summary.json` — `08e68c0f7fbaed8a846a634c2ba3e22fd384cb4222b520aa264fb7a2a2b7a2c0`; RICO25 `.cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-s3-wiring-rico25/s3-lockstep/summary.json` — `80ecc443a450306bddc361ef7f7c5ad37bf9cca684bfbc35506c4d2d82f877cc`                                                                                                                                                                                                                                                                                                                                                                         | Both commands returned `0`, wrote a logger and checkpoint, recorded the trainer state, and recorded scheduler `not applicable`; the shipped worker count is 4.                                                                                                                                                                                                   |
| S4                   | `"$LACE_AUDIT_VENV/bin/python" models/lace/tests/vendor_parity/training_stage_evidence.py s4 --device cuda --evaluation-batch-size 256 --lace-data-root .cache/lace/data --checkpoint-root .cache/lace/original/model --fid-root .cache/lace/fidroot-refresh-20261008T125333 --output-root .cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-s4-fid-refresh`                                                                                                                       | Summary `.cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-s4-fid-refresh/s4-loader-evaluation/summary.json` — `a9a094e9fe815f0cb4545a6fe3061e6909d4ea55996c5ffa3e24c5d2db60a919`; PubLayNet evaluation `.cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-s4-fid-refresh/s4-loader-evaluation/publaynet-evaluation/evaluation.json` — `ea595edecfe0dc06d0cdf2673a32bbffb3c16f6f7d864cced9d1b0577ae31364`; RICO25 evaluation `.cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-s4-fid-refresh/s4-loader-evaluation/rico25-evaluation/evaluation.json` — `4698218b2b15e015a55c04aab65de25746e651af78772e2772ad612cc9b45727`; identity proof — `d6cf7b0f68ac34a47e142a427c2f227ed07ad32d4a0dc429292a8ae198fca2ba` | Every loader row and every full TEST prediction is exact for both datasets; the fresh-cache feature files are PubLayNet `7eb0debcc4f22e77e19ec199069202b3a0f8995a7b9788e92e57bb2522336fa4` and RICO25 `fc8cdd44a1f05ee49f6f77d3788607ebdb381aceb9a496354de8456701bdaf51`; the executed pipeline/evaluation diff against the original S4 execution path is empty. |
| S5                   | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)`                                                                                                                                                                                                                                                                                                                                                                                                              | `.cache/lace/full-run/<dataset>/manifest.json; evaluation-path-parity: https://github.com/creative-graphic-design/design-generators/issues/423`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Not run.                                                                                                                                                                                                                                                                                                                                                         |

## Reproduction Results

The package and original implementation agree exactly for the asserted S0-S4 populations on PubLayNet and RICO25. The reported metrics are the full TEST evaluation path; S5 training-seed reproduction remains `not-yet-run` for both datasets.

| Dataset   | System | Status                                                                                  | Seed scope                       | Primary metrics                                                                                                 | Loss evidence                                                                 | Artifact summary                                                                                                                               |
| --------- | ------ | --------------------------------------------------------------------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| PubLayNet | both   | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)` | parity and evaluation seeds only | alignment 0.12166322767734528; overlap 4.450292110443115; FID 4.861519020556045; maximum IoU 0.3281917111234028 | Fixed-batch, optimizer-step, and 300-step traces exact; loss difference `0.0` | `.cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-s4-fid-refresh/s4-loader-evaluation/publaynet-evaluation/evaluation.json` |
| RICO25    | both   | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)` | parity and evaluation seeds only | alignment 0.11119277030229568; overlap 83.5333480834961; FID 3.384604672204091; maximum IoU 0.33588000299444437 | Fixed-batch, optimizer-step, and 300-step traces exact; loss difference `0.0` | `.cache/lace/stage-evidence/314a625da0dffe43ea6d60d91461cf4edb80c88c-s4-fid-refresh/s4-loader-evaluation/rico25-evaluation/evaluation.json`    |
| RICO13    | both   | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/423)` | no claimed seed scope            | no approved authors' RICO13 checkpoint                                                                          | no S0-S4 claim                                                                | `https://github.com/creative-graphic-design/design-generators/issues/423`                                                                      |

The S4 loader gate covers PubLayNet train/val/test populations of 315757/16619/11142 layouts in 1234/65/44 batches and RICO25 train/val/test populations of 358510/2109/4218 layouts in 1401/9/17 batches. Exact evidence loaders use zero workers; the shipped and production configuration uses four workers.

### Comparison Scope

| Dataset   | System | Evaluator                                                                                         | Test split       | Checkpoint-selection rule    | Sample count                      |
| --------- | ------ | ------------------------------------------------------------------------------------------------- | ---------------- | ---------------------------- | --------------------------------- |
| PubLayNet | both   | `vendor/lace/test.py::test_layout_cond` and `LacePipeline.__call__`, with vendor metric functions | LACE TEST stream | authors' `publaynet_best.pt` | 11142 layouts per evaluation seed |
| RICO25    | both   | `vendor/lace/test.py::test_layout_cond` and `LacePipeline.__call__`, with vendor metric functions | LACE TEST stream | authors' `rico25_best.pt`    | 4218 layouts per evaluation seed  |
| RICO13    | both   | not applicable; no approved authors' RICO13 checkpoint                                            | not applicable   | not applicable               | 0 layouts; outside this evidence  |

The compared evaluation inputs are the captured vendor `real_layout` and the package processor's own encoding of the same `bbox`, `labels`, and `mask`; predictions are then compared on the same decoded layout. The harness records both input artifacts, loaded package state, prediction files, evaluator, FIDNetV3 weights, and feature cache.

The first and second fresh feature-cache extractions were bitwise equal on the same CUDA runtime; the maximum absolute difference between the fresh and superseded caches is PubLayNet `0.0006326884031295776` and RICO25 `0.0002557337284088135`, attributed to the superseded cache's unrecorded device/kernel/runtime environment rather than same-runtime nondeterminism. Vendor and package parity uses the same fresh cache on both sides, so cache choice does not affect the S4 comparison.

The measured source diff from the original S4 execution commit `532b2312f13dfd4d9821d623d311aef440b6676f` to the current head over `models/lace/src` and `lib` is confined to `models/lace/src/lace/training/dataset.py`, `models/lace/src/lace/training/lightning_module.py`, and the trace callback `models/lace/src/lace/training/trace.py`; the evidence/harness commits changed only harness and documentation files. The Lightning hunk moves EMA shadow tensors to the model device at fit start; the dataset hunk removes the legacy-root fallback; the trace hunk adds the opt-in evidence callback and shared tensor-hash and norm summaries. The executed `models/lace/src/lace/pipeline*` and evaluation paths have an empty diff against the original S4 execution commit, so the S4 result stands.

The natural S3 records actually ran `traingen fit` with the shipped config and the deterministic, seed, loader, data-root, step, batch-limit, and output-root overrides shown in their `command.json` files; those records predate the harness-only callback move and therefore have no callback override in the command. The final production-wiring rows ran the same package entry point with `--trainer.callbacks+=lace.training.trace.LaceTrainingTraceCallback`; this changes only callback attachment, not package construction, initialization, or training numerics.

## Regeneration Metadata

The authors' checkpoints are under `.cache/lace/original/model/`; the cited fresh FID evaluator, FIDNetV3 weights, and feature caches are under `.cache/lace/fidroot-refresh-20261008T125333/`; the superseded cache is retained under `.cache/lace/fidroot-superseded/`. The FID evaluator source is LayoutDM commit `873b5eebe4c61862e5c08a10859accf65a168dfd`; its SHA-256 is `5df4cf82a869167d8cb6ab19470e58ecf42a576c0b8ae134c424cbd8505fdced`; the fresh PubLayNet and RICO25 FIDNetV3 weight SHA-256 values are `ff7208304e5c5f673ddd7cd5d73f85a0982df97a70713016f1b034a9c972d6fd` and `3e99f113bdea8f6e4623103bea88aff6217f322eef3deba5a787f3acd19e3296`; the LayoutDM starter archive SHA-256 is `357a0b8cd305793164ae4e9da1033ac1b687bfd46a53716673b32c83280c443b`. The machine-written fresh-cache provenance is `.cache/lace/fidroot-refresh-20261008T125333/provenance.json` with SHA-256 `d3e2ccc630680f0b8455e46ac6fc832a615c0a42b9a58265960e1425784f3842`.

```text
.cache/lace/data/
.cache/lace/original/model/
.cache/lace/fidroot-refresh-20261008T125333/
.cache/lace/fidroot-superseded/
.cache/lace/stage-evidence/
```

Acquire the FID evaluator and weights into the cache, then create the test feature cache without writing under `vendor/`:

```bash
export LACE_AUDIT_VENV="<your audit venv>"
export LACE_DATA_ROOT=".cache/lace/data"
export LACE_FID_ROOT=".cache/lace/fidroot-refresh-20261008T125333"
mkdir -p "$LACE_FID_ROOT/fid" "$LACE_FID_ROOT/FIDNetV3" "$LACE_FID_ROOT/feature" .cache/lace/original
curl --fail --location --output "$LACE_FID_ROOT/fid/model.py" "https://raw.githubusercontent.com/CyberAgentAILab/layout-dm/873b5eebe4c61862e5c08a10859accf65a168dfd/src/trainer/trainer/fid/model.py"
curl --fail --location --output .cache/lace/original/layoutdm_starter.zip https://github.com/CyberAgentAILab/layout-dm/releases/download/v1.0.0/layoutdm_starter.zip
unzip -o .cache/lace/original/layoutdm_starter.zip 'download/fid_weights/FIDNetV3/*/model_best.pth.tar' -d .cache/lace/original/layoutdm-unpacked
cp -a .cache/lace/original/layoutdm-unpacked/download/fid_weights/FIDNetV3/. "$LACE_FID_ROOT/FIDNetV3/"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="vendor/lace:$LACE_FID_ROOT" "$LACE_AUDIT_VENV/bin/python" -B - <<'PY'
import os
import pickle
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from fid.model import load_fidnet_v3
from util.datasets.publaynet import PubLayNetDataset
from util.datasets.rico import Rico25Dataset
from util.seq_util import sparse_to_dense

data_root = Path(os.environ["LACE_DATA_ROOT"])
fid_root = Path(os.environ["LACE_FID_ROOT"])
device = torch.device("cuda")
for dataset_name, dataset_class in (("publaynet", PubLayNetDataset), ("rico25", Rico25Dataset)):
    dataset = dataset_class(dir=str(data_root), split="test", max_seq_length=25)
    loader = DataLoader(dataset, shuffle=False, batch_size=20, num_workers=4, pin_memory=True)
    model = load_fidnet_v3(dataset, str(fid_root / "FIDNetV3"), device=device)
    features = []
    with torch.no_grad():
        for batch in loader:
            bbox, labels, padding_mask, _ = sparse_to_dense(batch)
            features.append(model.extract_features(bbox.to(device), labels.to(device), padding_mask.to(device)).cpu())
    with (fid_root / "feature" / f"fid_feat_test_{dataset_name}.pk").open("wb") as stream:
        pickle.dump(features, stream)
PY
```

## Training Commands

The package training entry point is `traingen fit` with the shipped config. The production-wiring harness appends the evidence-only callback with `--trainer.callbacks+=lace.training.trace.LaceTrainingTraceCallback`; this changes only how the callback is attached, not package construction, initialization, or training numerics. Its exact overrides are `--model.init_args.seed_mode deterministic` (a determinism-policy override of the shipped `default`), `--seed_everything 123` (versus shipped `false`), `--model.init_args.seed 10000`, `--data.init_args.loader_seed 42975`, `--data.init_args.processed_data_dir .cache/lace/data/<dataset>-max25/processed`, `--trainer.max_steps 2`, `--trainer.limit_train_batches 2`, and `--trainer.default_root_dir <production-root>`. The logger, checkpoint, scheduler status, and trainer state are recorded in the S3 summary artifacts.

```bash
export LACE_AUDIT_VENV="<your audit venv>"
CUDA_VISIBLE_DEVICES="<gpu>" UV_FROZEN=1 UV_PROJECT_ENVIRONMENT="$LACE_AUDIT_VENV" \
  uv run --package lace --extra training traingen fit \
  --config models/lace/configs/training/lace_publaynet.yaml \
  --trainer.devices=1
```

```bash
export LACE_AUDIT_VENV="<your audit venv>"
CUDA_VISIBLE_DEVICES="<gpu>" UV_FROZEN=1 UV_PROJECT_ENVIRONMENT="$LACE_AUDIT_VENV" \
  uv run --package lace --extra training traingen fit \
  --config models/lace/configs/training/lace_rico25.yaml \
  --trainer.devices=1
```

```bash
scripts/run_member_tests.sh models/lace
```

The checkpoint-gated topology test `test_s0_copied_checkpoint_forward_matches_package` is local-only because the authors' checkpoints remain under `.cache/lace/original/model/`; the regular member suite records that missing local asset as a skip.

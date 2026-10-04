---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - Layout-Corrector
---

# Layout-Corrector Training

The staged evidence passes for RICO25 and PubLayNet at training seed 42975, evaluation seed 0, and loader-stream control seed 314159. Natural production-path S3 is bitwise for 300 steps on both datasets, so the synchronized layer is not needed. S4 covers the configured 16-worker loader stream and the original evaluation entry point with retained original-code outputs. S5 is `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`.

Run commands from the repository root. Generated data, logs, checkpoints, converted local pipelines, and evaluation artifacts stay under `.cache/layout-corrector/` or the explicitly named local audit and evaluation roots.

## Install

The repository environment uses the `training` extra:

```bash
UV_FROZEN=1 uv sync --package layout-corrector --extra training
```

Lockfile generation for the `training` extra uses this one deliberate command:

```bash
uv lock
```

The audited GPU runtime was installed with this complete command. `AUDIT_VENV` is a local path supplied by the operator.

```bash
AUDIT_VENV=<audit-venv>
UV_FROZEN=1 uv venv --python 3.11 "$AUDIT_VENV"
UV_FROZEN=1 uv pip install --python "$AUDIT_VENV/bin/python" --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
UV_FROZEN=1 uv pip install --python "$AUDIT_VENV/bin/python" lightning hydra-core scikit-learn torch-geometric
UV_FROZEN=1 uv pip install --python "$AUDIT_VENV/bin/python" -e lib/laygen -e models/layout-dm -e models/layout-corrector
UV_FROZEN=1 uv -q pip freeze --python "$AUDIT_VENV/bin/python" | sha256sum
```

The verified runtime is Python 3.11.15, torch 2.8.0+cu128, and torchvision 0.23.0+cu128. The freeze SHA-256 is `f718a14d14e97427cf29be4dbe46cffd0c57cc1487b1b759fb0cd9650a75b919`. The torch wheel SHA-256 is `039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed`; the torchvision wheel SHA-256 is `93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb`.

## Data

| Dataset   | Source                                                                                                                                                                 | Config or path                                                                                                                                   |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| RICO25    | [creative-graphic-design/Rico](https://huggingface.co/datasets/creative-graphic-design/Rico), configuration `ui-screenshots-and-hierarchies-with-semantic-annotations` | `datasets/rico25-max25/{train,val,test}.pt`; frozen LayoutDM checkpoint `pretrained_weights/layoutdm_rico/0/best_model.pt`                       |
| PubLayNet | [creative-graphic-design/PubLayNet](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)                                                                 | `datasets/publaynet-max25/{train,val,test}.pt`; frozen LayoutDM checkpoint `pretrained_weights/layoutdm_publaynet/0/best_model.pt`               |
| Crello    | [cyberagent/crello](https://huggingface.co/datasets/cyberagent/crello)                                                                                                 | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`; no approved starter-kit evaluation assets are available |

The processed split SHA-256 values are RICO25 train `7ae17f4f5ef5061932609e93fb6568c2b0d4bd553d27e72affa550e1686f37b9`, val `6bd5a3d0333cfb4cd97ce3e526eb580ace1959fa7b766bdbe33d9f92b0a926f5`, test `2377d2893db7d0356fd174f943f894ea40e44166f4142827137b2c7d1f217ebc`; PubLayNet train `efb09fd0571be3a4be07afeccdc07bc04f283f0f8b946ef742be3ef17e662d71`, val `5725c2cae89549497c07d4ca8a96fc24a82bb397f146cc66bf3cb0388a648874`, test `392ca5119d1d8c21878f43fb9214cca705a9600e10dfac744a5624a9a48c8e73`.

## Configs

Training configs live under `models/layout-corrector/configs/training`.

| Config                                         | Dataset   | Seed mode     | Purpose                                                                             |
| ---------------------------------------------- | --------- | ------------- | ----------------------------------------------------------------------------------- |
| `layoutcorrector_rico25.yaml`                  | RICO25    | default       | Package Lightning training recipe with the frozen LayoutDM reference and 16 workers |
| `layoutcorrector_publaynet.yaml`               | PubLayNet | default       | Package Lightning training recipe with the frozen LayoutDM reference and 16 workers |
| `layoutcorrector_rico25_deterministic.yaml`    | RICO25    | deterministic | Deterministic smoke configuration                                                   |
| `layoutcorrector_publaynet_deterministic.yaml` | PubLayNet | deterministic | Deterministic smoke configuration                                                   |

The shipped default configs are accepted by the production entry point. The exact config validation commands were:

```bash
UV_FROZEN=1 <audit-venv>/bin/traingen fit --config models/layout-corrector/configs/training/layoutcorrector_rico25.yaml --print_config
UV_FROZEN=1 <audit-venv>/bin/traingen fit --config models/layout-corrector/configs/training/layoutcorrector_publaynet.yaml --print_config
```

Both exited 0. The first output line for each was `# lightning.pytorch==2.6.5`. The natural S3 harness invokes these same shipped configs and overrides only the seed, step limit, runtime root, and cache paths.

## Scheduler and recipe notes

The package training path is `traingen fit` with `LayoutCorrectorTrainingModule`, `LayoutCorrectorDataModule`, Lightning `Trainer`, the configured `ReduceLROnPlateau` scheduler, validation, and 16 loader workers. The optimizer is AdamW with learning rate `5.0e-4`, betas `(0.9, 0.98)`, and the shipped weight decay `0.1`. The trainer clips gradients by norm at `1.0`, and the module registration order matches the original model's `model.parameters()` order.

The S0-S3 runs use the importance-sampled corruption branch, CUDA device 0, 32-bit arithmetic, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms=False`, and `cudnn.deterministic=False`. Initialization follows the entry-point order of setting the seed, building diffusion, then building the corrector. CPU tests cover the initialization device and the complete expected parameter-registration name list.

The shared LayoutDM sampling behavior used by the evidence includes the original `log(1e-30)` posterior floor and vocabulary-axis softmax and reduction before flattening. The S4 evaluator is `vendor/layout-corrector/bin/corrector_test_eval.py`, which routes to `corrector_test`, with conditions `unconditional`, `c`, and `cwh`, `corrector_t_list=[10,20,30]`, 100 diffusion steps, batch size 512, no Gumbel noise, and seed 0. The original-code output files are retained outputs from sweep source commit `5ee3af2`; the package comparison and `bin/calc_metrics.py` were rerun. Evaluator subprocesses used `PYTHONHASHSEED=0` because the evaluator's unordered condition-key set and multiprocessing reduction otherwise make `maximum_iou` process-order dependent.

PubLayNet has 11,142 TEST layouts. Its unconditional count is 1,000 because the vendor `corrector_test` configuration explicitly uses `num_uncond_samples=1000`; `c` and `cwh` use all 11,142 TEST layouts.

## Seed policy

S0 initialization uses seed 123. Natural S1-S3 evidence uses training seed 42975, with each system's production RNG path, no per-step RNG restore, and no injected shared random tensors. S4 evaluation uses seed 0. The loader-stream comparison uses control seed 314159. S5 has no training-seed comparison.

## Validation stages

| Stage | Scope                                                  | Purpose                                                                                                        |
| ----- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity             | CPU/device, topology, frozen LayoutDM state, optimizer, scheduler, and dataset static-state agreement          |
| S1    | Fixed-batch pre-optimizer trace parity                 | Production training-step inputs, corruption, reconstruction, logits, losses, and total loss                    |
| S2    | One optimizer-step parity                              | Production clipping, gradients, optimizer state, scheduler boundary, and post-step parameters                  |
| S3    | Natural 300-step production-path trajectory            | Per-step loss, gradients, learning rate, optimizer state, parameters, importance probability, RNG, and repeats |
| S4    | 16-worker loader stream and evaluation-path comparison | Loader batch identity plus package and original-code prediction, weight, count, hash, OOB, and metric equality |
| S5    | Full-run statistical comparison                        | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`                        |

## Stage evidence

| Stage | Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Artifact                                                                                                                                                                            | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `CUDA_VISIBLE_DEVICES=0 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_DM_CACHE=<layout-dm-cache> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k test_s0_training_static_state_matches_vendor -q`                                                                                                                                                                                                                                                                                                                                                                                      | `.cache/layout-corrector/stage-evidence/s0-static/{rico25,publaynet}/summary.json`                                                                                                  | PASS for both datasets at source commit `aeb2e7e5b03a2bfc8c2a301a0d5156643409f39a`; initialization device, topology, frozen LayoutDM state, optimizer, scheduler, and dataset static values agree.                                                                                                                                                                                                                                                                                                                                                                |
| S1    | `CUDA_VISIBLE_DEVICES=0 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_CORRECTOR_EVIDENCE_WORKERS=16 LAYOUT_DM_CACHE=<layout-dm-cache> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k test_s1_fixed_batch_pre_optimizer_trace_matches_vendor -q`                                                                                                                                                                                                                                                                                                                                       | `.cache/layout-corrector/stage-evidence/s1-fixed-batch/{rico25,publaynet}/summary.json`                                                                                             | PASS for both datasets at source commit `aeb2e7e5b03a2bfc8c2a301a0d5156643409f39a`; every recorded prepared input, corruption, reconstruction, logit, per-attribute loss, weighted loss, and total loss has maximum absolute difference 0.0.                                                                                                                                                                                                                                                                                                                      |
| S2    | `CUDA_VISIBLE_DEVICES=0 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_DM_CACHE=<layout-dm-cache> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k test_s2_one_optimizer_step_matches_vendor -q`                                                                                                                                                                                                                                                                                                                                                                                         | `.cache/layout-corrector/stage-evidence/s2-optimizer-step/{rico25,publaynet}/summary.json`                                                                                          | PASS for both datasets at source commit `aeb2e7e5b03a2bfc8c2a301a0d5156643409f39a`; gradient norms and all post-step parameter differences are 0.0, and the scheduler is checked at the validation boundary.                                                                                                                                                                                                                                                                                                                                                      |
| S3    | `CUDA_VISIBLE_DEVICES=0 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_DM_CACHE=<layout-dm-cache> LAYOUT_CORRECTOR_S3_STEPS=300 LAYOUT_CORRECTOR_DIAGNOSTIC_TAG=post-softmax CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k test_s3_natural_lockstep_matches_vendor -q`                                                                                                                                                                                                                                                                                | `.cache/layout-corrector/stage-evidence/s3-lockstep/aeb2e7e/{rico25,publaynet}/{trace.jsonl,summary.json}`                                                                          | PASS for both datasets at source commit `aeb2e7e5b03a2bfc8c2a301a0d5156643409f39a`; 300 production steps have first divergence `null`, maximum loss, gradient, learning-rate, parameter, optimizer-state, and importance-probability digest differences 0.0, and both repeat envelopes are bitwise stable. The synchronized layer is not needed. The systematic causes covered by regression tests are x0 sampling softmax-before-flatten, conditional initialization of weak or invalid tokens, and `predict_start` reduction order through the corrector steps. |
| S4    | `CUDA_VISIBLE_DEVICES=0 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_EVAL_ASSET_ROOT=<starter-kit-download> LAYOUT_CORRECTOR_EVAL_PIPELINE_ROOT=.cache/layout-corrector/converted/evaluation/pipeline LAYOUT_CORRECTOR_S4_DEVICE=cuda:0 LAYOUT_CORRECTOR_S4_REUSE_VENDOR=1 LAYOUT_CORRECTOR_S4_REUSE_PACKAGE=1 LAYOUT_CORRECTOR_S4_VENDOR_SCRATCH_ROOT=<retained-root>/{dataset} LAYOUT_CORRECTOR_S4_VENDOR_SWEEP_COMMIT=5ee3af2 LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k 'test_retained_vendor_output_unpickles or test_s4_test_evaluation_path_matches_vendor' -q` | `.cache/layout-corrector/stage-evidence/loader-stream/{rico25,publaynet}/summary.json` and `.cache/layout-corrector/stage-evidence/evaluation-path/{rico25,publaynet}/summary.json` | PASS for both datasets at source commit `315de746ee7124b55198a73addd3bd54be452d68`; the loader stream observed 16 workers and matched batch streams, and the retained original-code outputs were unpickled, compared, and scored through the original evaluator.                                                                                                                                                                                                                                                                                                  |
| S5    | No launch command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`                                                                                             | Not run.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

## Reproduction results

S0-S4 pass for RICO25 and PubLayNet at the stated seeds. S0-S3 use source commit `aeb2e7e5b03a2bfc8c2a301a0d5156643409f39a`. S4 uses source commit `315de746ee7124b55198a73addd3bd54be452d68`, which changes only retained-evaluator metric execution; S0-S3 therefore stand under the rerun rule. S5 full-run statistical reproduction is not claimed.

| Dataset   | System   | Status                                                                                  | Seed scope                             | Primary metrics                                                                                                                                                                                                                              | Loss and training evidence                                          | Artifact summary                                                                                                                           |
| --------- | -------- | --------------------------------------------------------------------------------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| RICO25    | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | training seed 42975; evaluation seed 0 | S4 metrics are the full-precision maps in `evaluation-path/rico25/summary.json`; unconditional FID/coverage `5.069486887145331/0.4879089615931721`, c `2.518342496124035/0.8048838311996207`, cwh `2.0221323503336066/0.8560929350403035`    | 300-step natural production trace is bitwise equal to original code | `.cache/layout-corrector/stage-evidence/{s0-static,s1-fixed-batch,s2-optimizer-step,s3-lockstep,loader-stream,evaluation-path}/rico25/`    |
| RICO25    | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | training seed 42975; evaluation seed 0 | The full-precision metric maps equal the original-code maps for every quality field in all three conditions; values are in `evaluation-path/rico25/summary.json`                                                                             | 300-step natural production trace is bitwise equal to original code | `.cache/layout-corrector/stage-evidence/{s0-static,s1-fixed-batch,s2-optimizer-step,s3-lockstep,loader-stream,evaluation-path}/rico25/`    |
| PubLayNet | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | training seed 42975; evaluation seed 0 | S4 metrics are the full-precision maps in `evaluation-path/publaynet/summary.json`; unconditional FID/coverage `11.460619920552091/0.16657691617303896`, c `5.725142420400573/0.665948662717645`, cwh `2.762920882279616/0.7379285586070723` | 300-step natural production trace is bitwise equal to original code | `.cache/layout-corrector/stage-evidence/{s0-static,s1-fixed-batch,s2-optimizer-step,s3-lockstep,loader-stream,evaluation-path}/publaynet/` |
| PubLayNet | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | training seed 42975; evaluation seed 0 | The full-precision metric maps equal the original-code maps for every quality field in all three conditions; values are in `evaluation-path/publaynet/summary.json`                                                                          | 300-step natural production trace is bitwise equal to original code | `.cache/layout-corrector/stage-evidence/{s0-static,s1-fixed-batch,s2-optimizer-step,s3-lockstep,loader-stream,evaluation-path}/publaynet/` |
| Crello    | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | no evidence                            | Not measured                                                                                                                                                                                                                                 | Not measured                                                        | `.cache/layout-corrector/`                                                                                                                 |
| Crello    | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | no evidence                            | Not measured                                                                                                                                                                                                                                 | Not measured                                                        | `.cache/layout-corrector/`                                                                                                                 |

### Comparison scope

| Dataset   | System | Evaluator                                                                                                 | Test split    | Checkpoint-selection rule                                                                                                                                              | Sample count                          |
| --------- | ------ | --------------------------------------------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| RICO25    | both   | `vendor/layout-corrector/bin/corrector_test_eval.py` through `corrector_test`, then `bin/calc_metrics.py` | TEST          | Released seed-0 LayoutDM and Layout-Corrector starter-kit checkpoints; corrector checkpoint SHA-256 `26eca27dcbfb0fdbabc3f45e61d700ff4437e10c3642a5b2ff4f0c1be3bc3a2f` | 1,000 unconditional; 4,218 c and cwh  |
| PubLayNet | both   | `vendor/layout-corrector/bin/corrector_test_eval.py` through `corrector_test`, then `bin/calc_metrics.py` | TEST          | Released seed-0 LayoutDM and Layout-Corrector starter-kit checkpoints; corrector checkpoint SHA-256 `4aa6760952267f419f829669f99207db7673d7540a782859972bb766dea17553` | 1,000 unconditional; 11,142 c and cwh |
| Crello    | both   | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`                   | not evaluated | No approved starter-kit evaluation assets                                                                                                                              | Not measured                          |

The evaluation record contains the package and original-code prediction artifact paths, SHA-256 values, layout and element counts, original normalized center-xywh out-of-bounds counts, full-precision metric maps, and runtimes for each condition. The package and original-code prediction files are byte-identical and their decoded prediction tensors are identical. The original evaluator commit is `ea60d84461c88b3ca4d491cfcd3adcc0c8d0eba8`, the LayoutDM source commit is `873b5eebe4c61862e5c08a10859accf65a168dfd`, and the retained original-code sweep source is `5ee3af2`.

| Dataset   | Condition     | Layouts | Elements, package/original | OOB elements in original normalized center-xywh, package/original | Package prediction file                                                                                   | Original-code prediction file                                                                            | SHA-256, both files                                                |
| --------- | ------------- | ------- | -------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| RICO25    | unconditional | 1,000   | 10,842 / 10,842            | 968 / 968                                                         | `.cache/layout-corrector/stage-evidence/evaluation-path/rico25/unconditional-package-predictions.json`    | `.cache/layout-corrector/stage-evidence/evaluation-path/rico25/unconditional-vendor-predictions.json`    | `a8d378cba6c68d8c1a6d871532b6b19b2861f56b69b512da15d4e968e11b7869` |
| RICO25    | c             | 4,218   | 47,129 / 47,129            | 4,405 / 4,405                                                     | `.cache/layout-corrector/stage-evidence/evaluation-path/rico25/c-package-predictions.json`                | `.cache/layout-corrector/stage-evidence/evaluation-path/rico25/c-vendor-predictions.json`                | `9f09aadef7a62db350cedb49096f8d63c921e8925976259cb655f78eff930f31` |
| RICO25    | cwh           | 4,218   | 47,129 / 47,129            | 4,616 / 4,616                                                     | `.cache/layout-corrector/stage-evidence/evaluation-path/rico25/cwh-package-predictions.json`              | `.cache/layout-corrector/stage-evidence/evaluation-path/rico25/cwh-vendor-predictions.json`              | `7013eaca68d423374e5cd80fe7244da1505dc65e8a7c6cf8b82d668cd0117bae` |
| PubLayNet | unconditional | 1,000   | 9,017 / 9,017              | 1 / 1                                                             | `.cache/layout-corrector/stage-evidence/evaluation-path/publaynet/unconditional-package-predictions.json` | `.cache/layout-corrector/stage-evidence/evaluation-path/publaynet/unconditional-vendor-predictions.json` | `c9bc23e53f1a68dcb47104bdd9e225629e697784f7ccc4d59a8552b00658ed92` |
| PubLayNet | c             | 11,142  | 119,402 / 119,402          | 5 / 5                                                             | `.cache/layout-corrector/stage-evidence/evaluation-path/publaynet/c-package-predictions.json`             | `.cache/layout-corrector/stage-evidence/evaluation-path/publaynet/c-vendor-predictions.json`             | `ec68dd123976d0a7cad4fc53897d53f5a70827dc7bb9c50882c12907a1c7adb9` |
| PubLayNet | cwh           | 11,142  | 119,402 / 119,402          | 3 / 3                                                             | `.cache/layout-corrector/stage-evidence/evaluation-path/publaynet/cwh-package-predictions.json`           | `.cache/layout-corrector/stage-evidence/evaluation-path/publaynet/cwh-vendor-predictions.json`           | `6813a9773599f8bb81b16bbe387d8ac780f9786e9e3bbf3a13ad75bc74d30ccb` |

Identical prediction files produce identical metrics through the same evaluator. Both systems here have full-precision values from `bin/calc_metrics.py` under `PYTHONHASHSEED=0`, and every quality metric has maximum absolute difference `0.0` in all six comparisons; `t_total` is runtime metadata and is not a quality metric. The package values therefore equal the original-code values at full precision, not only after rounding. Out-of-bounds counts are report-only counts of decoded elements outside the original normalized `[0,1]` center-xywh frame; no element is removed from any metric denominator, and no `valid_elements_per_prediction` metric is used.

## Regeneration metadata

The S0-S3 evidence commit `aeb2e7e5b03a2bfc8c2a301a0d5156643409f39a` and the S4 evaluator-only commit `315de746ee7124b55198a73addd3bd54be452d68` are evidence ancestors of the final PR head. The audited runtime freeze SHA-256 is `f718a14d14e97427cf29be4dbe46cffd0c57cc1487b1b759fb0cd9650a75b919`; evidence ran on GPU 0 with 16 loader workers.

```text
.cache/layout-corrector/stage-evidence/s0-static/<dataset>/summary.json
.cache/layout-corrector/stage-evidence/s1-fixed-batch/<dataset>/summary.json
.cache/layout-corrector/stage-evidence/s2-optimizer-step/<dataset>/summary.json
.cache/layout-corrector/stage-evidence/s3-lockstep/aeb2e7e/<dataset>/{trace.jsonl,summary.json}
.cache/layout-corrector/stage-evidence/loader-stream/<dataset>/summary.json
.cache/layout-corrector/stage-evidence/evaluation-path/<dataset>/{summary.json,*-predictions.json,*-input-ids.bin}
```

## Training commands

Initialize the original-code submodules and run one of the shipped package recipes:

```bash
git submodule update --init vendor/layout-corrector vendor/layout-dm
CUDA_VISIBLE_DEVICES=<gpu-index> UV_FROZEN=1 uv run --package layout-corrector --extra training traingen fit --config models/layout-corrector/configs/training/layoutcorrector_<rico25|publaynet>.yaml --trainer.devices=1
```

Run the staged parity suite with local assets and required evidence:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> UV_FROZEN=1 PARITY_REQUIRE=1 uv run --package layout-corrector --extra training --extra vendor pytest models/layout-corrector/tests/vendor_parity -m 'vendor_parity and training' -rs
```

Convert a trained checkpoint and smoke-test package loading with `from_pretrained`:

```bash
UV_FROZEN=1 uv run --package layout-corrector --extra convert models/layout-corrector/scripts/convert_original_checkpoint.py --checkpoint .cache/layout-corrector/training-runs/<dataset>/checkpoints/<checkpoint>.ckpt --output-dir .cache/layout-corrector/converted-trained/<dataset>
UV_FROZEN=1 uv run --package layout-corrector python - <<'PY'
from layout_corrector import LayoutCorrectorPipeline

pipe = LayoutCorrectorPipeline.from_pretrained('.cache/layout-corrector/converted-trained/<dataset>')
output = pipe(condition_type='unconditional', num_inference_steps=2)
print(output.bbox.shape, output.labels.shape, output.mask.shape)
PY
```

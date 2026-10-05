---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - Layout-Corrector
---

# Layout-Corrector Training

The staged evidence passes for RICO25 and PubLayNet at training seed 42975, evaluation seed 0, and loader-stream control seed 314159. Natural production-path S3 is bitwise for 300 steps on both datasets, including the vendor importance-probability (`pt`) digest, so the synchronized layer is not needed. S4 passes the configured 16-worker loader stream and the original evaluation entry point with retained original-code outputs. S5 is `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`.

Run commands from the repository root. Generated data, logs, checkpoints, converted local pipelines, and evaluation artifacts stay under `.cache/layout-corrector/` or explicitly named local audit and evaluation roots.

## Install

The package declares its `training` extra, which provides Lightning, `traingen`, and `traingen-parity` for the shipped `traingen fit` recipes. The lockfile was updated once with the deliberate `uv lock` command after adding that extra.

```bash
UV_FROZEN=1 uv sync --package layout-corrector --extra training
uv lock
```

The audited GPU runtime was installed with this complete command. `AUDIT_VENV`, `TORCH_WHEEL`, and `TORCHVISION_WHEEL` are local paths supplied by the operator; the wheel hashes below identify the verified files.

```bash
AUDIT_VENV=<audit-venv>
TORCH_WHEEL=<torch-wheel>
TORCHVISION_WHEEL=<torchvision-wheel>
UV_FROZEN=1 uv venv --python 3.11 "$AUDIT_VENV"
UV_FROZEN=1 uv pip install --python "$AUDIT_VENV/bin/python" "$TORCH_WHEEL" "$TORCHVISION_WHEEL"
UV_FROZEN=1 uv pip install --python "$AUDIT_VENV/bin/python" lightning hydra-core scikit-learn torch-geometric
UV_FROZEN=1 uv pip install --python "$AUDIT_VENV/bin/python" -e lib/laygen -e models/layout-dm -e models/layout-corrector
UV_FROZEN=1 uv -q pip freeze --python "$AUDIT_VENV/bin/python" | sha256sum
```

The verified runtime is Python 3.11.15, torch 2.8.0+cu128, and torchvision 0.23.0+cu128 on GPU 0. The freeze SHA-256 is `1d4113b27f178f05e9e01ced2fb22c2503ddf107ffbdef7dbb91b6d81c19d0b5`. The torch wheel SHA-256 is `039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed`; the torchvision wheel SHA-256 is `93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb`.

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

The shipped default configs are accepted by the production entry point. These exact validation commands both exited 0; their first output line was `# lightning.pytorch==2.6.5`, so neither command has an error line.

```bash
UV_FROZEN=1 <audit-venv>/bin/traingen fit --config models/layout-corrector/configs/training/layoutcorrector_rico25.yaml --print_config
UV_FROZEN=1 <audit-venv>/bin/traingen fit --config models/layout-corrector/configs/training/layoutcorrector_publaynet.yaml --print_config
```

Natural S3 drives those same shipped configs and overrides only seed, step bounds, validation cadence, runtime root, checkpoint paths, cluster-center paths, and processed-data paths required for the audited local assets. The training run used the shipped scheduler, validation, Trainer wiring, and 16-worker DataLoader.

## Scheduler and Recipe Notes

The package training path is `traingen fit` with `LayoutCorrectorTrainingModule`, `LayoutCorrectorDataModule`, Lightning `Trainer`, the configured `ReduceLROnPlateau` scheduler, validation, and 16 loader workers. The optimizer is AdamW with learning rate `5.0e-4`, betas `(0.9, 0.98)`, weight decay `0.1`, and gradient clipping by norm at `1.0`. The module registration order matches the original model's `model.parameters()` order.

S0 initialization follows the entry-point order of setting the seed, building diffusion, then building the corrector. CPU tests cover the initialization device and the complete expected parameter-registration name list. S0-S3 use CUDA device 0, 32-bit arithmetic, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms=False`, and `cudnn.deterministic=False`.

The shared LayoutDM sampling behavior used by the evidence includes the original `log(1e-30)` posterior floor and vocabulary-axis softmax and reduction before flattening; these behaviors are the oracle because S4 uses the original evaluation entry point. The regression tests are in `models/layout-dm/tests/test_conditioning.py`, `models/layout-dm/tests/test_scheduler.py`, and the LayoutDM vendor-parity suite.

The evaluator is `vendor/layout-corrector/bin/corrector_test_eval.py`, which routes to `corrector_test`, followed by `vendor/layout-corrector/bin/calc_metrics.py`. It uses conditions `unconditional`, `c`, and `cwh`, `corrector_t_list=[10,20,30]`, 100 diffusion steps, batch size 512, no Gumbel noise, and evaluation seed 0. The retained original-code outputs came from vendor sweep source commit `5ee3af2`; the package comparison and metric calculation were rerun against those outputs. The evaluator source and commit are `vendor/layout-corrector` at `ea60d84461c88b3ca4d491cfcd3adcc0c8d0eba8`; the LayoutDM source commit is `873b5eebe4c61862e5c08a10859accf65a168dfd`. `PYTHONHASHSEED=0` was used for evaluator process-order stability.

PubLayNet has 11,142 TEST layouts. Its unconditional count is 1,000 because the vendor `corrector_test` configuration explicitly uses `num_uncond_samples=1000`; `c` and `cwh` use all 11,142 TEST layouts.

The assertion sweep found and fixed 15 compute-without-assert groups across the S0-S4 harness. The final gates assert counts, input streams, prediction tensors and byte hashes, same-weight identity, full-precision metrics, natural-training losses and state digests, scheduler state, and populated vendor/package importance-probability digests.

## Seed Policy

S0 initialization uses seed 123. Natural S1-S3 evidence uses training seed 42975, with each system's production RNG path, no per-step RNG restore, and no injected shared random tensors. S4 evaluation uses evaluation seed 0. The loader-stream comparison uses control seed 314159. S5 has no training-seed comparison.

## Validation Stages

| Stage | Scope                                                  | Purpose                                                                                                        |
| ----- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity             | CPU/device, topology, frozen LayoutDM state, optimizer, scheduler, and dataset static-state agreement          |
| S1    | Fixed-batch pre-optimizer trace parity                 | Production training-step inputs, corruption, reconstruction, logits, losses, and total loss                    |
| S2    | One optimizer-step parity                              | Production clipping, gradients, optimizer state, scheduler boundary, and post-step parameters                  |
| S3    | Natural 300-step production-path trajectory            | Per-step loss, gradients, learning rate, optimizer state, parameters, importance probability, RNG, and repeats |
| S4    | 16-worker loader stream and evaluation-path comparison | Loader batch identity plus package and original-code prediction, weight, count, hash, OOB, and metric equality |
| S5    | Full-run statistical comparison                        | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`                        |

## Stage Evidence

| Stage | Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Artifact                                                                                                                                                                                                        | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `CUDA_VISIBLE_DEVICES=0 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_DM_CACHE=<layout-dm-cache> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k test_s0_training_static_state_matches_vendor -q`                                                                                                                                                                                                                                                                                                                                                                                                                                               | `.cache/layout-corrector/stage-evidence/s0-static/{rico25,publaynet}/summary.json`                                                                                                                              | PASS for both datasets at source commit `1819a1ecc0e86c3156072b46dae1346fb9a4675b`; initialization device, topology, frozen LayoutDM state, optimizer, scheduler, and dataset static values agree.                                                                                                                                                                                                                                                                                                                                  |
| S1    | `CUDA_VISIBLE_DEVICES=0 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_CORRECTOR_EVIDENCE_WORKERS=16 LAYOUT_DM_CACHE=<layout-dm-cache> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k test_s1_fixed_batch_pre_optimizer_trace_matches_vendor -q`                                                                                                                                                                                                                                                                                                                                                                                                | `.cache/layout-corrector/stage-evidence/s1-fixed-batch/{rico25,publaynet}/summary.json`                                                                                                                         | PASS for both datasets at source commit `1819a1ecc0e86c3156072b46dae1346fb9a4675b`; all recorded prepared inputs, corruption, reconstruction, logits, per-attribute losses, weighted losses, and total losses match.                                                                                                                                                                                                                                                                                                                |
| S2    | `CUDA_VISIBLE_DEVICES=0 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_DM_CACHE=<layout-dm-cache> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k test_s2_one_optimizer_step_matches_vendor -q`                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `.cache/layout-corrector/stage-evidence/s2-optimizer-step/{rico25,publaynet}/summary.json`                                                                                                                      | PASS for both datasets at source commit `1819a1ecc0e86c3156072b46dae1346fb9a4675b`; production clipping, gradients, post-step parameters, optimizer state, and the scheduler boundary match.                                                                                                                                                                                                                                                                                                                                        |
| S3    | `CUDA_VISIBLE_DEVICES=0 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_DM_CACHE=<layout-dm-cache> LAYOUT_CORRECTOR_S3_STEPS=300 LAYOUT_CORRECTOR_DIAGNOSTIC_TAG=final-production-validation-v8 CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k test_s3_natural_lockstep_matches_vendor -q`                                                                                                                                                                                                                                                                                                                       | `.cache/layout-corrector/stage-evidence/s3-lockstep/final-production-validation-v8/{rico25,publaynet}/{trace.jsonl,summary.json}`                                                                               | PASS for 300 production steps on both datasets at source commit `1819a1ecc0e86c3156072b46dae1346fb9a4675b`; first divergence is `null`, loss/gradient/LR/optimizer/parameter/RNG/input/timestep traces match, repeats are bitwise stable, and all 300 vendor/package importance-probability digests are populated and equal. The synchronized layer is `not-needed` on this natural result. RICO25 validation and scheduler `best` differ by `1.4901161193847656e-08` within the asserted `2e-8` bound; PubLayNet differs by `0.0`. |
| S4    | `CUDA_VISIBLE_DEVICES=0 UV_FROZEN=1 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_CORRECTOR_EVIDENCE_WORKERS=16 LAYOUT_DM_CACHE=<layout-dm-cache> LAYOUT_CORRECTOR_EVAL_ASSET_ROOT=<starter-kit-download> LAYOUT_CORRECTOR_EVAL_PIPELINE_ROOT=<converted-pipeline-root> LAYOUT_CORRECTOR_S4_DEVICE=cuda:0 LAYOUT_CORRECTOR_S4_REUSE_VENDOR=1 LAYOUT_CORRECTOR_S4_REUSE_PACKAGE=0 LAYOUT_CORRECTOR_S4_VENDOR_SWEEP_COMMIT=5ee3af2 LAYOUT_CORRECTOR_S4_VENDOR_SCRATCH_ROOT=<retained-root-for-dataset> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k 'test_retained_vendor_output_unpickles or test_s4_test_evaluation_path_matches_vendor' -q` | `.cache/layout-corrector/stage-evidence/loader-stream/{rico25,publaynet}/summary.json` and `.cache/layout-corrector/stage-evidence/evaluation-path/{rico25,publaynet}/summary.json`                             | PASS for both datasets; the 16-worker train/validation/test stream matched, retained vendor outputs unpickled through the evaluator import path, package outputs were regenerated on GPU 0, and the original evaluator produced identical full-precision metric maps.                                                                                                                                                                                                                                                               |
| S5    | No launch command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`; evaluation-path prerequisite: `.cache/layout-corrector/stage-evidence/evaluation-path/{rico25,publaynet}/summary.json` | Not run.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

## Reproduction Results

S0-S4 pass for RICO25 and PubLayNet at the stated seeds and source commit `1819a1ecc0e86c3156072b46dae1346fb9a4675b`. The evidence covers the production `traingen fit` path, configured validation and scheduler, 16 loader workers, and the original evaluation entry point. S5 full-run statistical reproduction is not claimed.

| Dataset   | System   | Status                                                                                  | Seed scope                             | Primary metrics                                                                                                                                                          | Loss evidence                                                       | Artifact summary                                                                                                                           |
| --------- | -------- | --------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| RICO25    | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | training seed 42975; evaluation seed 0 | Full-precision FID/coverage: unconditional `5.069486887145331/0.4879089615931721`, c `2.518342496124035/0.8048838311996207`, cwh `2.0221323503336066/0.8560929350403035` | 300-step natural production trace is bitwise equal to package       | `.cache/layout-corrector/stage-evidence/{s0-static,s1-fixed-batch,s2-optimizer-step,s3-lockstep,loader-stream,evaluation-path}/rico25/`    |
| RICO25    | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | training seed 42975; evaluation seed 0 | Full-precision quality metrics equal the original-code maps in all three conditions; maps are in `evaluation-path/rico25/summary.json`                                   | 300-step natural production trace is bitwise equal to original code | `.cache/layout-corrector/stage-evidence/{s0-static,s1-fixed-batch,s2-optimizer-step,s3-lockstep,loader-stream,evaluation-path}/rico25/`    |
| PubLayNet | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | training seed 42975; evaluation seed 0 | Full-precision FID/coverage: unconditional `11.460619920552091/0.16657691617303896`, c `5.725142420400573/0.665948662717645`, cwh `2.762920882279616/0.7379285586070723` | 300-step natural production trace is bitwise equal to package       | `.cache/layout-corrector/stage-evidence/{s0-static,s1-fixed-batch,s2-optimizer-step,s3-lockstep,loader-stream,evaluation-path}/publaynet/` |
| PubLayNet | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | training seed 42975; evaluation seed 0 | Full-precision quality metrics equal the original-code maps in all three conditions; maps are in `evaluation-path/publaynet/summary.json`                                | 300-step natural production trace is bitwise equal to original code | `.cache/layout-corrector/stage-evidence/{s0-static,s1-fixed-batch,s2-optimizer-step,s3-lockstep,loader-stream,evaluation-path}/publaynet/` |
| Crello    | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | no evidence                            | Not measured                                                                                                                                                             | Not measured                                                        | `.cache/layout-corrector/`                                                                                                                 |
| Crello    | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | no evidence                            | Not measured                                                                                                                                                             | Not measured                                                        | `.cache/layout-corrector/`                                                                                                                 |

### Comparison Scope

| Dataset   | System | Evaluator                                                                                                 | Test split    | Checkpoint-selection rule                                                                                                                                   | Sample count                          |
| --------- | ------ | --------------------------------------------------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| RICO25    | both   | `vendor/layout-corrector/bin/corrector_test_eval.py` through `corrector_test`, then `bin/calc_metrics.py` | TEST          | Released seed-0 LayoutDM and Layout-Corrector starter-kit checkpoints; corrector SHA-256 `26eca27dcbfb0fdbabc3f45e61d700ff4437e10c3642a5b2ff4f0c1be3bc3a2f` | 1,000 unconditional; 4,218 c and cwh  |
| PubLayNet | both   | `vendor/layout-corrector/bin/corrector_test_eval.py` through `corrector_test`, then `bin/calc_metrics.py` | TEST          | Released seed-0 LayoutDM and Layout-Corrector starter-kit checkpoints; corrector SHA-256 `4aa6760952267f419f829669f99207db7673d7540a782859972bb766dea17553` | 1,000 unconditional; 11,142 c and cwh |
| Crello    | both   | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`                   | not evaluated | No approved starter-kit evaluation assets                                                                                                                   | Not measured                          |

The evaluation record contains per-system prediction files and SHA-256 values, layout and element counts, original normalized center-xywh out-of-bounds counts, full-precision metric maps, evaluator commits, and runtime metadata. For each dataset and condition, the package file is `.cache/layout-corrector/stage-evidence/evaluation-path/<dataset>/<condition>-package-predictions.json` and the original-code file is `.cache/layout-corrector/stage-evidence/evaluation-path/<dataset>/<condition>-vendor-predictions.json`. Package and original-code prediction files are byte-identical and their decoded prediction tensors are identical.

| Dataset   | Condition     | Layouts | Elements, package/original | OOB elements in original normalized center-xywh, package/original | Package prediction SHA-256                                         | Original-code prediction SHA-256                                   | Package/original runtime seconds        |
| --------- | ------------- | ------- | -------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ | --------------------------------------- |
| RICO25    | unconditional | 1,000   | 10,842 / 10,842            | 968 / 968                                                         | `a8d378cba6c68d8c1a6d871532b6b19b2861f56b69b512da15d4e968e11b7869` | `a8d378cba6c68d8c1a6d871532b6b19b2861f56b69b512da15d4e968e11b7869` | 30.219294173642993 / 31.625059843063354 |
| RICO25    | c             | 4,218   | 47,129 / 47,129            | 4,405 / 4,405                                                     | `9f09aadef7a62db350cedb49096f8d63c921e8925976259cb655f78eff930f31` | `9f09aadef7a62db350cedb49096f8d63c921e8925976259cb655f78eff930f31` | 129.97484252043068 / 136.27404046058655 |
| RICO25    | cwh           | 4,218   | 47,129 / 47,129            | 4,616 / 4,616                                                     | `7013eaca68d423374e5cd80fe7244da1505dc65e8a7c6cf8b82d668cd0117bae` | `7013eaca68d423374e5cd80fe7244da1505dc65e8a7c6cf8b82d668cd0117bae` | 130.42690181918442 / 139.88332533836365 |
| PubLayNet | unconditional | 1,000   | 9,017 / 9,017              | 1 / 1                                                             | `c9bc23e53f1a68dcb47104bdd9e225629e697784f7ccc4d59a8552b00658ed92` | `c9bc23e53f1a68dcb47104bdd9e225629e697784f7ccc4d59a8552b00658ed92` | 29.3330017644912 / 32.13598918914795    |
| PubLayNet | c             | 11,142  | 119,402 / 119,402          | 5 / 5                                                             | `ec68dd123976d0a7cad4fc53897d53f5a70827dc7bb9c50882c12907a1c7adb9` | `ec68dd123976d0a7cad4fc53897d53f5a70827dc7bb9c50882c12907a1c7adb9` | 338.6027402561158 / 358.97434163093567  |
| PubLayNet | cwh           | 11,142  | 119,402 / 119,402          | 3 / 3                                                             | `6813a9773599f8bb81b16bbe387d8ac780f9786e9e3bbf3a13ad75bc74d30ccb` | `6813a9773599f8bb81b16bbe387d8ac780f9786e9e3bbf3a13ad75bc74d30ccb` | 339.65225753188133 / 366.57621693611145 |

Identical prediction files produce identical metrics through the same evaluator. Both systems have full-precision values from `bin/calc_metrics.py` under `PYTHONHASHSEED=0`, and every quality metric has maximum absolute difference `0.0` for all six comparisons; `t_total` is runtime metadata and is not a quality metric. The package values therefore equal the original-code values at full precision, not only after rounding.

Out-of-bounds counts are report-only counts of decoded elements outside the original normalized `[0,1]` center-xywh frame. No element is removed from any metric denominator, and no `valid_elements_per_prediction` metric is used.

The S3 natural record includes the systematic causes fixed during implementation: x0 sampling softmax-before-flatten, conditional initialization of weak or invalid tokens, and `predict_start` reduction order through the corrector steps. The synchronized layer is not needed because the natural production trajectory has no first divergence and bitwise-stable vendor and package repeats.

## Regeneration Metadata

The final evidence source commit is `1819a1ecc0e86c3156072b46dae1346fb9a4675b`. It is an ancestor of the final PR head. The audited runtime freeze SHA-256 is `1d4113b27f178f05e9e01ced2fb22c2503ddf107ffbdef7dbb91b6d81c19d0b5`; evidence ran on GPU 0 with 16 loader workers.

```text
.cache/layout-corrector/stage-evidence/s0-static/<dataset>/summary.json
.cache/layout-corrector/stage-evidence/s1-fixed-batch/<dataset>/summary.json
.cache/layout-corrector/stage-evidence/s2-optimizer-step/<dataset>/summary.json
.cache/layout-corrector/stage-evidence/s3-lockstep/final-production-validation-v8/<dataset>/{trace.jsonl,summary.json}
.cache/layout-corrector/stage-evidence/loader-stream/<dataset>/summary.json
.cache/layout-corrector/stage-evidence/evaluation-path/<dataset>/{summary.json,*-predictions.json,*-input-ids.bin}
```

## Training Commands

Initialize the original-code submodules and run a shipped package recipe:

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

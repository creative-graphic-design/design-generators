---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - Layout-Corrector
---

# Layout-Corrector Training

This record is for contributors reproducing Layout-Corrector training and evaluation. S0-S4 pass for RICO25 and PubLayNet at training seed 42975, evaluation seed 0, and loader-stream control seed 314159. The S0-S4 evidence is from source commit `35d0666d645d5e141d9805af724913f6bed63965`. Natural production-path S3 is bitwise for 300 steps on both datasets, including the vendor importance-probability digest, so the synchronized layer is not needed. S5 is `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`.

Run commands from the repository root. Generated data, logs, checkpoints, converted local pipelines, and evaluation artifacts stay under `.cache/layout-corrector/` or an explicitly supplied local audit root.

## Install

The package `training` extra provides Lightning, `traingen`, and `traingen-parity`. The lockfile was regenerated once after that extra was added.

```bash
UV_FROZEN=1 uv sync --package layout-corrector --extra training
uv lock  # deliberate lockfile regeneration for the training extra
```

The following is the complete audited-runtime install command. The operator supplies `AUDIT_VENV`, `TORCH_WHEEL`, and `TORCHVISION_WHEEL` as local paths.

```bash
AUDIT_VENV=<audit-venv>
TORCH_WHEEL=<torch-2.8.0-cu128-wheel>
TORCHVISION_WHEEL=<torchvision-0.23.0-cu128-wheel>
UV_FROZEN=1 uv venv --python 3.11 "$AUDIT_VENV"
UV_FROZEN=1 uv pip install --python "$AUDIT_VENV/bin/python" "$TORCH_WHEEL" "$TORCHVISION_WHEEL" lightning hydra-core scikit-learn torch-geometric pycocotools
UV_FROZEN=1 uv pip install --python "$AUDIT_VENV/bin/python" -e lib/traingen -e lib/traingen-parity -e lib/laygen -e models/layout-dm -e models/layout-corrector
UV_FROZEN=1 uv pip freeze --python "$AUDIT_VENV/bin/python" | sha256sum
```

The audited runtime is Python 3.11.15, torch 2.8.0+cu128, and torchvision 0.23.0+cu128 on GPU 0. All S0-S4 records report freeze SHA-256 `5cf8957e7847a651f5d02c13124ae49dc5a05ead20a06f994e35ef132bd289fa`. The torch wheel SHA-256 is `039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed`; the torchvision wheel SHA-256 is `93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb`.

## Data

| Dataset   | Source                                                                                                                                                                 | Config or path                                                                                                                                   |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| RICO25    | [creative-graphic-design/Rico](https://huggingface.co/datasets/creative-graphic-design/Rico), configuration `ui-screenshots-and-hierarchies-with-semantic-annotations` | `datasets/rico25-max25/{train,val,test}.pt`; frozen LayoutDM checkpoint `pretrained_weights/layoutdm_rico/0/best_model.pt`                       |
| PubLayNet | [creative-graphic-design/PubLayNet](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)                                                                 | `datasets/publaynet-max25/{train,val,test}.pt`; frozen LayoutDM checkpoint `pretrained_weights/layoutdm_publaynet/0/best_model.pt`               |
| Crello    | [cyberagent/crello](https://huggingface.co/datasets/cyberagent/crello)                                                                                                 | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`; no approved starter-kit evaluation assets are available |

The processed split SHA-256 values are RICO25 train `7ae17f4f5ef5061932609e93fb6568c2b0d4bd553d27e72affa550e1686f37b9`, val `6bd5a3d0333cfb4cd97ce3e526eb580ace1959fa7b766bdbe33d9f92b0a926f5`, test `2377d2893db7d0356fd174f943f894ea40e44166f4142827137b2c7d1f217ebc`; PubLayNet train `efb09fd0571be3a4be07afeccdc07bc04f283f0f8b946ef742be3ef17e662d71`, val `5725c2cae89549497c07d4ca8a96fc24a82bb397f146cc66bf3cb0388a648874`, test `392ca5119d1d8c21878f43fb9214cca705a9600e10dfac744a5624a9a48c8e73`.

## Configs

Training configs live under `models/layout-corrector/configs/training`.

| Config                                         | Dataset   | Seed mode     | Purpose                                                                     |
| ---------------------------------------------- | --------- | ------------- | --------------------------------------------------------------------------- |
| `layoutcorrector_rico25.yaml`                  | RICO25    | default       | Shipped production recipe with the frozen LayoutDM reference and 16 workers |
| `layoutcorrector_publaynet.yaml`               | PubLayNet | default       | Shipped production recipe with the frozen LayoutDM reference and 16 workers |
| `layoutcorrector_rico25_deterministic.yaml`    | RICO25    | deterministic | Deterministic smoke configuration                                           |
| `layoutcorrector_publaynet_deterministic.yaml` | PubLayNet | deterministic | Deterministic smoke configuration                                           |

The shipped default configs are accepted by the production entry point. These exact commands both exit 0 and print `# lightning.pytorch==2.6.5` as their first output line.

```bash
UV_FROZEN=1 <audit-venv>/bin/traingen fit --config models/layout-corrector/configs/training/layoutcorrector_rico25.yaml --print_config
UV_FROZEN=1 <audit-venv>/bin/traingen fit --config models/layout-corrector/configs/training/layoutcorrector_publaynet.yaml --print_config
```

Natural S3 drives these same shipped configs and overrides only the seed, step bounds, validation cadence, runtime paths, checkpoint paths, cluster-center paths, and processed-data paths needed for the audited local assets. It uses the shipped scheduler, validation, Trainer wiring, and 16-worker DataLoader.

## Scheduler and recipe notes

The package training path is `traingen fit` with `LayoutCorrectorTrainingModule`, `LayoutCorrectorDataModule`, Lightning `Trainer`, the configured epoch `ReduceLROnPlateau` scheduler, validation, and 16 loader workers. The optimizer is AdamW with learning rate `5.0e-4`, betas `(0.9, 0.98)`, weight decay `0.1`, and gradient clipping by norm at `1.0`. The module registration order matches the original model's `model.parameters()` order.

S0 initialization follows the entry-point order of setting the seed, building diffusion, then building the corrector. CPU tests cover the initialization device and the complete expected parameter-registration name list. S0-S3 use CUDA device 0, 32-bit arithmetic, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms=False`, and `cudnn.deterministic=False`.

The shared [LayoutDM](https://github.com/creative-graphic-design/design-generators/tree/main/models/layout-dm) sampling behavior used by the evidence includes the original `log(1e-30)` posterior floor and vocabulary-axis softmax and reduction before flattening. These behaviors are the oracle because S4 uses the original evaluation entry point. Regression coverage is in `models/layout-dm/tests/test_conditioning.py`, `models/layout-dm/tests/test_scheduler.py`, and the LayoutDM vendor-parity suite.

The evaluator is `vendor/layout-corrector/bin/corrector_test_eval.py`, which routes to `corrector_test`, followed by `vendor/layout-corrector/bin/calc_metrics.py`. It uses conditions `unconditional`, `c`, and `cwh`, `corrector_t_list=[10,20,30]`, 100 diffusion steps, batch size 512, no Gumbel noise, and evaluation seed 0. The retained original-code outputs came from vendor sweep source commit `5ee3af2`; the package comparison and metric calculation executed against those outputs. The evaluator source commit is `ea60d84461c88b3ca4d491cfcd3adcc0c8d0eba8`; the LayoutDM source commit is `873b5eebe4c61862e5c08a10859accf65a168dfd`.

PubLayNet has 11,142 TEST layouts. Its unconditional count is 1,000 because the vendor entry point explicitly sets `num_uncond_samples=1000`; `c` and `cwh` use all 11,142 TEST layouts.

The final gates assert counts, input streams, prediction tensors and byte hashes, same-weight identity, full-precision metrics, natural-training losses and state digests, scheduler state, and populated vendor/package importance-probability digests.

## Seed policy

S0 initialization uses seed 123. Natural S1-S3 evidence uses training seed 42975, with each system's production RNG path, no per-step RNG restore, and no injected shared random tensors. S4 evaluation uses evaluation seed 0. The loader-stream comparison uses control seed 314159. S5 has no training-seed comparison.

## Validation stages

| Stage | Scope                                                  | Purpose                                                                                                                    |
| ----- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity             | CPU/device, topology, frozen LayoutDM state, optimizer, scheduler, and dataset static-state agreement                      |
| S1    | Fixed-batch pre-optimizer trace parity                 | Production training-step inputs, corruption, reconstruction, logits, losses, and total loss                                |
| S2    | One optimizer-step parity                              | Production clipping, gradients, optimizer state, scheduler boundary, and post-step parameters                              |
| S3    | Natural 300-step production-path trajectory            | Per-step loss, gradients, learning rate, optimizer state, parameters, importance probability, RNG, and repeats             |
| S4    | 16-worker loader stream and evaluation-path comparison | Loader batch identity plus package and original-code predictions, weights, counts, hashes, OOB counts, and metric equality |
| S5    | Full-run statistical comparison                        | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`                                    |

## Stage Evidence

| Stage | Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Artifact                                                                                                                                                                                                        | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0    | `CUDA_VISIBLE_DEVICES=0 UV_FROZEN=1 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_DM_CACHE=<layout-dm-cache> LAYOUT_CORRECTOR_EVIDENCE_WORKERS=16 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 PYTHONPATH="$PWD/lib/traingen-parity/src:$PWD/lib/traingen/src:$PWD/lib/laygen/src:$PWD/models/layout-dm/src:$PWD/models/layout-corrector/src" <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k test_s0_training_static_state_matches_vendor -q`                                                                                                                                                                                                                                                                            | `.cache/layout-corrector/stage-evidence/s0-static/{rico25,publaynet}/summary.json`                                                                                                                              | PASS for both datasets at source commit `35d0666d645d5e141d9805af724913f6bed63965`; initialization device, topology, frozen LayoutDM state, optimizer, scheduler, and dataset static values agree under audited runtime freeze `5cf8957e7847a651f5d02c13124ae49dc5a05ead20a06f994e35ef132bd289fa`.                                                                                                                                                                                                                                   |
| S1    | `CUDA_VISIBLE_DEVICES=0 UV_FROZEN=1 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_EVIDENCE_WORKERS=16 LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_DM_CACHE=<layout-dm-cache> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 PYTHONPATH="$PWD/lib/traingen-parity/src:$PWD/lib/traingen/src:$PWD/lib/laygen/src:$PWD/models/layout-dm/src:$PWD/models/layout-corrector/src" <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k test_s1_fixed_batch_pre_optimizer_trace_matches_vendor -q`                                                                                                                                                                                                                                                                  | `.cache/layout-corrector/stage-evidence/s1-fixed-batch/{rico25,publaynet}/summary.json`                                                                                                                         | PASS for both datasets at source commit `35d0666d645d5e141d9805af724913f6bed63965`; prepared inputs, corruption, reconstruction, logits, per-attribute losses, weighted losses, and total losses match under audited runtime freeze `5cf8957e7847a651f5d02c13124ae49dc5a05ead20a06f994e35ef132bd289fa`.                                                                                                                                                                                                                              |
| S2    | `CUDA_VISIBLE_DEVICES=0 UV_FROZEN=1 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_EVIDENCE_WORKERS=16 LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_DM_CACHE=<layout-dm-cache> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 PYTHONPATH="$PWD/lib/traingen-parity/src:$PWD/lib/traingen/src:$PWD/lib/laygen/src:$PWD/models/layout-dm/src:$PWD/models/layout-corrector/src" <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k test_s2_one_optimizer_step_matches_vendor -q`                                                                                                                                                                                                                                                                               | `.cache/layout-corrector/stage-evidence/s2-optimizer-step/{rico25,publaynet}/summary.json`                                                                                                                      | PASS for both datasets at source commit `35d0666d645d5e141d9805af724913f6bed63965`; production clipping, gradients, post-step parameters, optimizer state, and scheduler boundary match under audited runtime freeze `5cf8957e7847a651f5d02c13124ae49dc5a05ead20a06f994e35ef132bd289fa`.                                                                                                                                                                                                                                             |
| S3    | `CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 UV_FROZEN=1 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_DIAGNOSTIC_TAG=repair-35d0666 LAYOUT_CORRECTOR_EVIDENCE_WORKERS=16 LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_DM_CACHE=<layout-dm-cache> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 PYTHONPATH="$PWD/lib/traingen-parity/src:$PWD/lib/traingen/src:$PWD/lib/laygen/src:$PWD/models/layout-dm/src:$PWD/models/layout-corrector/src" <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m 'vendor_parity and training' -k test_s3_natural_lockstep_matches_vendor -q`                                                                                                                                                                                                  | `.cache/layout-corrector/stage-evidence/s3-lockstep/repair-35d0666/{rico25,publaynet}/{trace.jsonl,summary.json}`                                                                                               | PASS for 300 production steps on both datasets; natural vendor/package loss, gradients, learning rate, optimizer state, parameters, RNG, timesteps, importance probability, scheduler state, validation, and repeat envelopes match. The synchronized layer is not needed because natural traces and repeats are bitwise. Audited runtime freeze is `5cf8957e7847a651f5d02c13124ae49dc5a05ead20a06f994e35ef132bd289fa`.                                                                                                              |
| S4    | `CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 UV_FROZEN=1 LAYOUT_CORRECTOR_AUDIT_VENV=<audit-venv> LAYOUT_CORRECTOR_TORCH_WHEEL=<torch-wheel> LAYOUT_CORRECTOR_TORCHVISION_WHEEL=<torchvision-wheel> LAYOUT_CORRECTOR_EVAL_ASSET_ROOT=<evaluation-assets> LAYOUT_CORRECTOR_EVAL_PIPELINE_ROOT=<converted-pipelines> LAYOUT_CORRECTOR_S4_REUSE_VENDOR=1 LAYOUT_CORRECTOR_S4_VENDOR_SCRATCH_ROOT=<vendor-sweep-scratch-{dataset}> LAYOUT_CORRECTOR_S4_VENDOR_SWEEP_COMMIT=5ee3af2 LAYOUT_CORRECTOR_S4_DEVICE=cuda:0 LAYOUT_DM_CACHE=<layout-dm-cache> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PARITY_REQUIRE=1 PYTHONPATH="$PWD/lib/traingen-parity/src:$PWD/lib/traingen/src:$PWD/lib/laygen/src:$PWD/models/layout-dm/src:$PWD/models/layout-corrector/src" <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity/test_layout_corrector_training_parity.py -m vendor_parity -k 'test_s4_test_evaluation_path_matches_vendor' -q` | `.cache/layout-corrector/stage-evidence/loader-stream/{rico25,publaynet}/summary.json`; `.cache/layout-corrector/stage-evidence/evaluation-path/{rico25,publaynet}/summary.json`                                | PASS for both datasets at source commit `35d0666d645d5e141d9805af724913f6bed63965`. The loader stream matches at 16 workers. The retained vendor outputs unpickle through the evaluator import path, the package pipeline executes at the current head, package predictions match byte-for-byte, same-weight artifacts match, counts match, OOB counts match, and all full-precision quality metrics match exactly. All S4 records report audited runtime freeze `5cf8957e7847a651f5d02c13124ae49dc5a05ead20a06f994e35ef132bd289fa`. |
| S5    | No launch command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`; evaluation-path prerequisite: `.cache/layout-corrector/stage-evidence/evaluation-path/{rico25,publaynet}/summary.json` | Not run.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

## Reproduction Results

S0-S4 pass for RICO25 and PubLayNet from source commit `35d0666d645d5e141d9805af724913f6bed63965`. The evidence covers the shipped production `traingen fit` path, configured validation and scheduler, 16 loader workers, and the original evaluation entry point. S5 full-run statistical reproduction is not claimed.

| Dataset   | System   | Status                                                                                  | Seed scope                             | Primary metrics                                                                                                                                                          | Loss evidence                                                       | Artifact summary                                                                                                                           |
| --------- | -------- | --------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| RICO25    | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | training seed 42975; evaluation seed 0 | Full-precision FID/coverage: unconditional `5.069486887145331/0.4879089615931721`, c `2.518342496124035/0.8048838311996207`, cwh `2.0221323503336066/0.8560929350403035` | 300-step natural production trace is bitwise equal to package       | `.cache/layout-corrector/stage-evidence/{s0-static,s1-fixed-batch,s2-optimizer-step,s3-lockstep,loader-stream,evaluation-path}/rico25/`    |
| RICO25    | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | training seed 42975; evaluation seed 0 | Full-precision quality metric maps equal the original-code maps in all three conditions                                                                                  | 300-step natural production trace is bitwise equal to original code | `.cache/layout-corrector/stage-evidence/{s0-static,s1-fixed-batch,s2-optimizer-step,s3-lockstep,loader-stream,evaluation-path}/rico25/`    |
| PubLayNet | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | training seed 42975; evaluation seed 0 | Full-precision FID/coverage: unconditional `11.460619920552091/0.16657691617303896`, c `5.725142420400573/0.665948662717645`, cwh `2.762920882279616/0.7379285586070723` | 300-step natural production trace is bitwise equal to package       | `.cache/layout-corrector/stage-evidence/{s0-static,s1-fixed-batch,s2-optimizer-step,s3-lockstep,loader-stream,evaluation-path}/publaynet/` |
| PubLayNet | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | training seed 42975; evaluation seed 0 | Full-precision quality metric maps equal the original-code maps in all three conditions                                                                                  | 300-step natural production trace is bitwise equal to original code | `.cache/layout-corrector/stage-evidence/{s0-static,s1-fixed-batch,s2-optimizer-step,s3-lockstep,loader-stream,evaluation-path}/publaynet/` |
| Crello    | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | no evidence                            | Not measured                                                                                                                                                             | Not measured                                                        | `.cache/layout-corrector/`                                                                                                                 |
| Crello    | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)` | no evidence                            | Not measured                                                                                                                                                             | Not measured                                                        | `.cache/layout-corrector/`                                                                                                                 |

### Comparison Scope

| Dataset   | System | Evaluator                                                                                                 | Test split    | Checkpoint-selection rule                                                                                                                                   | Sample count                          |
| --------- | ------ | --------------------------------------------------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| RICO25    | both   | `vendor/layout-corrector/bin/corrector_test_eval.py` through `corrector_test`, then `bin/calc_metrics.py` | TEST          | Released seed-0 LayoutDM and Layout-Corrector starter-kit checkpoints; corrector SHA-256 `26eca27dcbfb0fdbabc3f45e61d700ff4437e10c3642a5b2ff4f0c1be3bc3a2f` | 1,000 unconditional; 4,218 c and cwh  |
| PubLayNet | both   | `vendor/layout-corrector/bin/corrector_test_eval.py` through `corrector_test`, then `bin/calc_metrics.py` | TEST          | Released seed-0 LayoutDM and Layout-Corrector starter-kit checkpoints; corrector SHA-256 `4aa6760952267f419f829669f99207db7673d7540a782859972bb766dea17553` | 1,000 unconditional; 11,142 c and cwh |
| Crello    | both   | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/426)`                   | not evaluated | No approved starter-kit evaluation assets                                                                                                                   | Not measured                          |

The evaluation record contains per-system prediction files and SHA-256 values, layout and element counts, out-of-bounds counts in the original normalized center-xywh frame, full-precision metric maps, evaluator commits, and runtime metadata. Package and original-code prediction files are byte-identical and their decoded prediction tensors are identical.

| Dataset   | Condition     | Layouts | Elements, package/original | OOB elements in original normalized center-xywh, package/original | Prediction SHA-256, package/original                                      |
| --------- | ------------- | ------- | -------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------- |
| RICO25    | unconditional | 1,000   | 10,842 / 10,842            | 968 / 968                                                         | `a8d378cba6c68d8c1a6d871532b6b19b2861f56b69b512da15d4e968e11b7869` / same |
| RICO25    | c             | 4,218   | 47,129 / 47,129            | 4,405 / 4,405                                                     | `9f09aadef7a62db350cedb49096f8d63c921e8925976259cb655f78eff930f31` / same |
| RICO25    | cwh           | 4,218   | 47,129 / 47,129            | 4,616 / 4,616                                                     | `7013eaca68d423374e5cd80fe7244da1505dc65e8a7c6cf8b82d668cd0117bae` / same |
| PubLayNet | unconditional | 1,000   | 9,017 / 9,017              | 1 / 1                                                             | `c9bc23e53f1a68dcb47104bdd9e225629e697784f7ccc4d59a8552b00658ed92` / same |
| PubLayNet | c             | 11,142  | 119,402 / 119,402          | 5 / 5                                                             | `ec68dd123976d0a7cad4fc53897d53f5a70827dc7bb9c50882c12907a1c7adb9` / same |
| PubLayNet | cwh           | 11,142  | 119,402 / 119,402          | 3 / 3                                                             | `6813a9773599f8bb81b16bbe387d8ac780f9786e9e3bbf3a13ad75bc74d30ccb` / same |

Identical prediction files produce identical metrics through the same evaluator. Both systems have full-precision values from `bin/calc_metrics.py`, and every quality metric has maximum absolute difference `0.0` for all six comparisons; `t_total` is runtime metadata and is not a quality metric. The package values equal the original-code values at full precision.

The OOB counts are report-only counts of decoded elements outside the original normalized `[0,1]` center-xywh frame. No element is removed from any metric denominator, and no `valid_elements_per_prediction` metric is used.

The S3 natural record includes the systematic causes fixed during implementation: x0 sampling softmax-before-flatten, conditional initialization of weak or invalid tokens, and `predict_start` reduction order through the corrector steps. The synchronized layer is not needed because the natural production trajectory has no first divergence and vendor and package repeats are bitwise stable.

## Regeneration Metadata

Evidence source commit `35d0666d645d5e141d9805af724913f6bed63965` is an ancestor of the final PR head. All evidence ran on GPU 0 with 16 loader workers under audited runtime freeze SHA-256 `5cf8957e7847a651f5d02c13124ae49dc5a05ead20a06f994e35ef132bd289fa`. The retained vendor evaluation outputs came from sweep source commit `5ee3af2`; the final comparison reused those outputs while executing the package pipeline at the evidence source commit. `git diff --stat 35d0666 35d0666 -- models/layout-corrector/src models/layout-dm/src` is empty because S4 was a clean-head rerun with no package-source change, so S0-S3 stand. The evidence commit is an ancestor of the final PR head.

```text
.cache/layout-corrector/stage-evidence/s0-static/<dataset>/summary.json
.cache/layout-corrector/stage-evidence/s1-fixed-batch/<dataset>/summary.json
.cache/layout-corrector/stage-evidence/s2-optimizer-step/<dataset>/summary.json
.cache/layout-corrector/stage-evidence/s3-lockstep/repair-35d0666/<dataset>/{trace.jsonl,summary.json}
.cache/layout-corrector/stage-evidence/loader-stream/<dataset>/summary.json
.cache/layout-corrector/stage-evidence/evaluation-path/<dataset>/{summary.json,*-predictions.json,*-input-ids.bin}
```

## Training Commands

Initialize the original-code submodules and print a shipped package config:

```bash
git submodule update --init vendor/layout-corrector vendor/layout-dm
UV_FROZEN=1 <audit-venv>/bin/traingen fit --config models/layout-corrector/configs/training/layoutcorrector_<rico25|publaynet>.yaml --print_config
```

Run a shipped training recipe:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> UV_FROZEN=1 <audit-venv>/bin/traingen fit --config models/layout-corrector/configs/training/layoutcorrector_<rico25|publaynet>.yaml --trainer.devices=1
```

Run the staged parity suite with local assets and required evidence:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> UV_FROZEN=1 PARITY_REQUIRE=1 <audit-venv>/bin/pytest models/layout-corrector/tests/vendor_parity -m 'vendor_parity and training' -rs
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

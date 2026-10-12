---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutFormerPP
---

# LayoutFormer++ Training

The current staged reproduction passes for all twelve LayoutFormer++ recipe families covered by this package. The records cover configuration, fixed-batch, optimizer-step, repeated multi-batch, and loader-stream agreement; trained-checkpoint reproduction is not claimed because the full-run stage has not been launched.

The covered recipe families are RICO25 label, RICO25 label-size, RICO25 relation, RICO25 refinement, RICO25 completion, RICO25 unconditional, PubLayNet label, PubLayNet label-size, PubLayNet relation, PubLayNet refinement, PubLayNet completion, and PubLayNet unconditional. The staged evidence uses the source commit `305e96c66b632f77781df7eade1a2d2fec5cbd19` and the dedicated runtime freeze `274f003252fc87c4361749bb83c4669c9cc746c509d17a33a8763466b2628638`.

Run commands from the repository root. Keep generated evidence, logs, data, and checkpoints under `.cache/layoutformerpp/`. Keep the authoritative processed data outside repository artifacts and provide it through `LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root>`.

## Install

```bash
uv sync --package layoutformerpp --extra training
uv sync --package layoutformerpp --extra training --extra vendor
```

The vendor extra is needed for original-implementation reference checks. The package training path uses the package-local DataModule and Lightning module.

## Data

| Dataset   | Source                                                                                                   | Required runtime input                                                                                |
| --------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| RICO25    | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico)           | Approved processed RICO25 data supplied through `LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root>`    |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | Approved processed PubLayNet data supplied through `LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root>` |

The fixed-batch checks use the repository's source-shaped fixtures. The repeated training and loader-stream checks compare the original loader with the package-local DataModule and DataLoader on the approved processed splits. Missing parity data fails the run.

## Configs

Training configs live under `models/layoutformerpp/configs/training`. The RICO25 families use the `rico25_` prefix and the PubLayNet families use the `publaynet_` prefix; each family has `label`, `label_size`, `relation`, `refinement`, `completion`, and `unconditional` variants.

## Scheduler and Recipe Notes

The reference mode is the plain-PyTorch basic training path. The parity harness compares the package scheduler, optimizer defaults, loss inputs, data encoding, task identifiers, and seed-order behavior with the original implementation. Scheduler cadence and state-dependent loader branches are exercised by the staged evidence before a full-run claim is considered.

The campaign preflight uses the dedicated cu126 Python runtime under `.cache/layoutformerpp/runtime/cu126-py311/`. Its normalized `uv pip freeze` is recorded at `.cache/layoutformerpp/runtime/cu126-py311/pip-freeze.txt` with SHA-256 `274f003252fc87c4361749bb83c4669c9cc746c509d17a33a8763466b2628638`. Sanitization replaces editable local file URLs with `<repo root>` and local Torch wheel URLs with package/version pins before hashing; package names and versions otherwise come from `uv pip freeze`.

## Seed Policy

The staged records preserve each run's process boundary, runtime freeze reference, selected-device setting, and test report. A numeric seed match alone is not a seed-paired claim. The full-run stage has no accepted training-seed or evaluation-seed scope because it has not been launched.

## Validation Stages

| Stage | Scope                                      | Purpose                                                                                     |
| ----- | ------------------------------------------ | ------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity | Confirms source manifests, configuration values, topology, vocabulary, and forward outputs. |
| S1    | Fixed-batch pre-optimizer trace parity     | Compares the first fixed batch before optimizer mutation.                                   |
| S2    | One optimizer-step parity                  | Compares backward, optimizer, and scheduler state after one update.                         |
| S3    | Short deterministic multi-batch run        | Compares repeated training, validation, logging, and production-fit wiring.                 |
| S4    | Deterministic loader stream                | Compares the authoritative train and validation streams.                                    |
| S5    | Full-run statistical comparison            | Requires a separate user-owned compute decision and launch manifest.                        |

## Stage Evidence

| Stage | Command                                                                                                                                                                                                                                                                                                | Artifact                                                                                                                                                         | Result                                                                                                                                |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 uv run --package layoutformerpp --extra training --extra vendor pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s0 -rs -q`                                                  | `.cache/layoutformerpp/post-merge/attempt-2/s0/record.json`                                                                                                      | passed: `39` tests; record SHA-256 `a539687f23e22cbc26537d264ad330f779d1390baebba677b2a64fa41379340f`                                 |
| S1    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> uv run --package layoutformerpp --extra training --extra vendor pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s1 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-2/s1/record.json`                                                                                                      | passed: `12` tests; record SHA-256 `26fe0ec46c2adc99372155891ecc64f8e9c6ac56638376938d919210db7e80ab`                                 |
| S2    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> uv run --package layoutformerpp --extra training --extra vendor pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s2 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-2/s2/record.json`                                                                                                      | passed: `12` tests; record SHA-256 `832b10586fb85ce8dff0e249d9aee9cbc9aded77f21999653ac766f0b5d52bcb`                                 |
| S3    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> uv run --package layoutformerpp --extra training --extra vendor pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s3 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-2/s3/record.json`                                                                                                      | passed: `22` tests; fixed `rtol=1e-4`, `atol=1e-5`; record SHA-256 `b20b2bbd04cd0659d867f7d5b78decde0c9357d1ac3bc7c32965d35603d9135b` |
| S4    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> uv run --package layoutformerpp --extra training --extra vendor pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s4 -rs -q` | `.cache/layoutformerpp/post-merge/attempt-2/s4/record.json`                                                                                                      | passed: `13` tests; record SHA-256 `6b298c9edaf5acdfb32f78efd653c67a56b87246d8b3b55e01b7456f84e7d61d`                                 |
| S5    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 uv run --package layoutformerpp --extra training pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s5 -rs -q`                                                                 | `.cache/layoutformerpp/post-merge/attempt-2/s5/manifest.json; evaluation-path-parity: .cache/layoutformerpp/post-merge/attempt-2/s5/evaluation-path-parity.json` | not launched; full-run compute remains a user-owned decision                                                                          |

The S3 comparison gate is the fixed tolerance shown above. A separate S3 process repeated the same comparison for reproducibility evidence; it is not an envelope gate. Its record is `.cache/layoutformerpp/post-merge/attempt-2/s3-repeat-2/record.json`, with `22` passed tests and record SHA-256 `4b7285d35825d4bc08291ea8f6d0a27f27c71e5a1e31f16ab49cfd4e1e67a165`. Every process wrote its own record and freeze file, and all observed divergences were localized to an operation before any nondeterminism attribution.

## Reproduction Results

The staged S0-S4 checks pass for every supported recipe family. Full-run trained-checkpoint reproduction is not claimed for any family because the full-run stage has not been launched.

| Dataset                 | System  | Status                                      | Seed scope     | Primary metrics | Loss evidence | Artifact summary                              |
| ----------------------- | ------- | ------------------------------------------- | -------------- | --------------- | ------------- | --------------------------------------------- |
| RICO25 label            | package | `not-yet-run (full-run stage not launched)` | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-2/` |
| RICO25 label-size       | package | `not-yet-run (full-run stage not launched)` | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-2/` |
| RICO25 relation         | package | `not-yet-run (full-run stage not launched)` | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-2/` |
| RICO25 refinement       | package | `not-yet-run (full-run stage not launched)` | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-2/` |
| RICO25 completion       | package | `not-yet-run (full-run stage not launched)` | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-2/` |
| RICO25 unconditional    | package | `not-yet-run (full-run stage not launched)` | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-2/` |
| PubLayNet label         | package | `not-yet-run (full-run stage not launched)` | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-2/` |
| PubLayNet label-size    | package | `not-yet-run (full-run stage not launched)` | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-2/` |
| PubLayNet relation      | package | `not-yet-run (full-run stage not launched)` | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-2/` |
| PubLayNet refinement    | package | `not-yet-run (full-run stage not launched)` | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-2/` |
| PubLayNet completion    | package | `not-yet-run (full-run stage not launched)` | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-2/` |
| PubLayNet unconditional | package | `not-yet-run (full-run stage not launched)` | not applicable | not evaluated   | not evaluated | `.cache/layoutformerpp/post-merge/attempt-2/` |

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

The launch manifest records the source commit, runtime freeze hash, selected-device setting, redacted data-root placeholder, fixed S3 tolerances, and every record and freeze path at `.cache/layoutformerpp/post-merge/attempt-2/launch_manifest.json`. The CUDA preflight output is at `.cache/layoutformerpp/post-merge/attempt-2/cuda-smoke.json`. The preflight uses a tiny matmul on logical `cuda:0` after `CUDA_VISIBLE_DEVICES=<device>` is applied and fails closed before launching a parity process when the smoke fails.

The dedicated runtime freeze is reproducible from `.cache/layoutformerpp/runtime/cu126-py311/bin/python` with `uv pip freeze --no-config --python .cache/layoutformerpp/runtime/cu126-py311/bin/python`, followed by the sanitization described in `.cache/layoutformerpp/runtime/README.md`. The runtime and campaign artifacts are cache-local and ignored; no generated tensors, checkpoints, images, downloaded datasets, or full-run artifacts belong in git.

## Training Commands

Run the staged parity checks with one selected CUDA device and the approved processed data root.

```bash
CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> \
  uv run --package layoutformerpp --extra training --extra vendor pytest \
  models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py \
  -m "vendor_parity and training" -rs
```

Train one recipe after the staged evidence and a separate user-owned full-run decision.

```bash
CUDA_VISIBLE_DEVICES=<device> LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> \
  uv run --package layoutformerpp --extra training traingen fit \
  --config models/layoutformerpp/configs/training/<recipe>.yaml \
  --trainer.devices=1
```

Do not launch the full-run stage from this document. It remains a separate compute decision after the complete staged ladder and any required evaluation-path parity prerequisite are available.

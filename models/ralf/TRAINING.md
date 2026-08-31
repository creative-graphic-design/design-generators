---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - RALF
---

# RALF Training

CGL's vendor-effective recipe is 30 epochs, not 70. The corrected S0 rerun and
the corrected S3 redo both pass on one Tesla V100-SXM2-32GB with scheduler
milestone 21. The prior 70-epoch S0 and S3 records remain invalidated; S1/S2
diagnostics and S4 loader evidence remain retained at `training-seed n=1`, and
S5 has not run. PKU loader and full-run reproduction are not claimed.

Here, S0 is static configuration/state, S1 is a fixed-batch pre-optimizer
trace, S2 is one optimizer step, S3 is production multi-batch training, S4 is
the authoritative loader stream, and S5 is full-run statistical comparison.

Run commands from the repository root. Generated data, logs, checkpoints, and
downloaded assets remain under `.cache/ralf/` or the authoritative cache
selected below and are not committed.

## Install

```bash
uv sync --package ralf --extra training
```

Install the `vendor` extra only when rerunning the original-code parity checks.

```bash
uv sync --package ralf --extra training --extra vendor
```

## Data

The package-local CGL `RalfDataModule` and the original RALF loader consume the
same authoritative cache for the accepted CGL S0-S4 evidence. The cache is
provided read-only through the `RALF_CACHE_DIR` environment variable; its
machine-specific mount path is intentionally not recorded in repository docs.

| Dataset | Source                                       | Config or path                                                     |
| ------- | -------------------------------------------- | ------------------------------------------------------------------ |
| CGL     | RALF cache supplied through `RALF_CACHE_DIR` | `$RALF_CACHE_DIR/dataset/cgl`, CGL train/validation splits         |
| PKU     | RALF cache supplied through `RALF_CACHE_DIR` | `$RALF_CACHE_DIR/dataset/pku`, not run in the accepted S4 evidence |

Its relevant contents include the CGL/PKU dataset trees, retrieval-index
tables, the ResNet and FIDNet precomputed weights, DreamSim retrieval data,
relationship tables, saliency/FAISS data, FID evaluation features, and the
original training-log directories. It is an authoritative read-only reference
for these runs. The cache is not localized into the repository worktree; the
currently selected user-provided local copy is supplied through
`RALF_CACHE_DIR`. Run provenance records the cache content-role inventory and a
per-file SHA256 manifest; the external cache is identified only by
`RALF_CACHE_DIR`, never by its machine-specific mount path.

The accepted CGL stream uses the vendor-effective transforms `image`,
`sort_label`, and `sort_lexicographic`, fixed retrieval indexes with
`top_k=16`, `random_retrieval=false`, and disjoint train/validation membership.

## Configs

Training configs live under `models/ralf/configs/training`.

| Config     | Dataset | Seed mode                              | Purpose                                                  |
| ---------- | ------- | -------------------------------------- | -------------------------------------------------------- |
| `cgl.yaml` | CGL     | fixed seed, deterministic warning mode | Package-local Lightning training and CGL parity          |
| `pku.yaml` | PKU     | fixed seed, deterministic warning mode | Package-local training configuration; parity not yet run |

## Scheduler and Recipe Notes

The package training path uses `RalfTrainingModule` and `RalfDataModule`
through the member-scoped `traingen fit` construction. The recipe uses AdamW,
weight decay `1e-4`, gradient clipping at `0.1`, batch size `32`, and
`accumulate_grad_batches=1`, so every batch reaches the optimizer directly.
CGL uses the vendor-effective 30-epoch recipe,
with three train batches and two validation batches per epoch in the planned
production control.

CGL effectively uses 30 epochs, not the vendor base value of 50. This schedule
is established by
`vendor/ralf/scripts/train/ralf_cgl.sh:1-3`, which sources
`vendor/ralf/scripts/run_job/end_to_end.sh:10-17,30-32`; that launcher sources
`vendor/ralf/configs/ralf_cgl/uncond.sh:2-5`, where `training.epochs=30` is
added, and `vendor/ralf/scripts/bin/train.sh:45-52` passes the Hydra override.
The vendor base `training.epochs=50` in
`vendor/ralf/image2layout/train/config/__init__.py:14-30` is therefore not the
effective CGL value. The package recipe now uses 30 epochs, and
`RalfTrainingModule.configure_optimizers()` derives the `MultiStepLR` milestone
as `int(0.7 * epochs)`, giving epoch 21 without a hardcoded milestone.
The superseded 70-epoch/49-milestone S3 evidence is retained only as invalidated
provenance. Both configs retain Lightning's `deterministic: warn` mode; the
corrected S0 and the corrected S3 redo establish the effective PyTorch warning
state on the supported runtime.

## Seed Policy

CGL uses seed `1` on one selected Tesla V100-SXM2-32GB for the staged checks.
The staged records are diagnostic evidence, not S5 training-seed evidence.

## Validation Stages

| Stage | Scope                                      | Purpose                                                                                                                                                                        |
| ----- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0    | Static config and initialized state parity | Confirms topology, parameter/state keys, optimizer groups, scheduler milestone, and encoded dataset state.                                                                     |
| S1    | Fixed-batch pre-optimizer trace parity     | Confirms prepared ids/labels, logits, and loss before optimizer mutation.                                                                                                      |
| S2    | One optimizer-step parity                  | Confirms named raw/clipped gradients, optimizer state, post-step parameters, and learning rates.                                                                               |
| S3    | Production multi-batch diagnostic          | Confirms production Trainer construction, scheduler/logging/accumulation/checkpoint wiring, with state-synchronized and natural trajectory layers kept separate.               |
| S4    | Deterministic loader stream                | Confirms train/validation order, transforms, masks, padding, retrieval tensors, and split membership.                                                                          |
| S5    | Full-run statistical comparison            | Three package/vendor seed pairs trained to 30 epochs and evaluated under the original evaluation protocol; every package metric mean falls inside the vendor three-seed range. |

## Stage Evidence

| Stage | Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Artifact                                                                      | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `CUDA_VISIBLE_DEVICES=7 PARITY_REQUIRE=1 uv run --active --no-sync --package ralf --extra training --extra vendor python models/ralf/tests/vendor_parity/run_training_stages.py --stage S0 --dataset cgl --cache-dir "$RALF_CACHE_DIR" --output .cache/ralf/training-reproduction/cgl/s0-corrected-cu126-gpu7-003/s0.json --seed 1`                                                                                                                                     | `.cache/ralf/training-reproduction/cgl/s0-corrected-cu126-gpu7-003/s0.json`   | PASS; artifact SHA-256 `6b8ad865bf7b59cdf1b06f87befc460b8dc7dafbc1e4bdac1ff20034c00d7ab8`, state SHA-256 `0c1c915bc1d2a8e321869ad40b3f2bac0314c66a2b3a790f5c90bcb26bcbe88e`, and scheduler milestone 21.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| S1    | `CUDA_VISIBLE_DEVICES=1 PARITY_REQUIRE=1 uv run --active --no-sync --package ralf --extra training --extra vendor python models/ralf/tests/vendor_parity/run_training_stages.py --stage S1 --dataset cgl --cache-dir "$RALF_CACHE_DIR" --output .cache/ralf/training-reproduction/cgl/s1-after-loader-order-002/s1.json --steps 1 --batch-size 32 --seed 1`                                                                                                             | `.cache/ralf/training-reproduction/cgl/s1-after-loader-order-002/s1.json`     | Pre-optimizer fixed-batch loss/logit evidence retained as an isolated diagnostic within the existing contract; the artifact SHA-256 is recorded in its evidence manifest.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| S2    | `CUDA_VISIBLE_DEVICES=1 PARITY_REQUIRE=1 uv run --active --no-sync --package ralf --extra training --extra vendor python models/ralf/tests/vendor_parity/run_training_stages.py --stage S2 --dataset cgl --cache-dir "$RALF_CACHE_DIR" --output .cache/ralf/training-reproduction/cgl/s2-after-loader-order-002/s2.json --steps 1 --batch-size 32 --seed 1`                                                                                                             | `.cache/ralf/training-reproduction/cgl/s2-after-loader-order-002/s2.json`     | One-step gradient/optimizer evidence retained as an isolated diagnostic within the existing contract; the artifact SHA-256 is recorded in its evidence manifest.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| S3    | `CUDA_VISIBLE_DEVICES="${RALF_GPU:?set RALF_GPU to one selected V100}" RALF_S3_OUTPUT=.cache/ralf/training-reproduction/cgl/s3/runs/after-blank-lines-005/s3.json PARITY_REQUIRE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 uv run --active --no-sync --package ralf --extra training --extra vendor python models/ralf/tests/vendor_parity/run_training_stages.py --stage S3 --dataset cgl --cache-dir "$RALF_CACHE_DIR" --output "$RALF_S3_OUTPUT" --batch-size 4 --seed 1` | `.cache/ralf/training-reproduction/cgl/s3/runs/after-blank-lines-005/s3.json` | Bounded numerical PASS at 30 epochs; artifact SHA-256 `82a20f880c01614e27c3cbda41ded83d46f981b0f18dbc83a4a151edc059ad42`. The natural layer leaves the S0-S2 contract at the first raw-gradient comparison (`epoch[0].batch[1]`, conv1 weight, max abs diff about `2e-6`), so the synchronized layer establishes the bounded PASS with 90/90 lockstep steps and `first_divergence=null`. Natural fits use batch size 4 because batch-32 natural fits peak at 31.0 GiB against the 31.73 GiB V100 ceiling; the batch-32 synchronized surface is covered by the 300-step lockstep probe below. Earlier per-run trace SHA-256 values are retained: `878c76d2fb1872d0cf6fd981fbd4d46f767062c22d9665da79e57734793bacf1` (run-016 natural), `f059356d9c3a32c5acf2a8f472e0388cb2bc43ec94ae5c35601afddb1f7569f9` (run-017 natural), and `09ffe676ffab5252e31812fccbeac8095711124e3f15f0ec13086a2f656f617c` (run-018 synchronized, batch 32).                                      |
| S4    | `CUDA_VISIBLE_DEVICES=1 PARITY_REQUIRE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 uv run --active --no-sync --package ralf --extra training --extra vendor python models/ralf/tests/vendor_parity/run_training_stages.py --stage S4 --dataset cgl --cache-dir "$RALF_CACHE_DIR" --output .cache/ralf/training-reproduction/cgl/s4/runs/run-010/s4.json --steps 8 --batch-size 32 --seed 1`                                                                                    | `.cache/ralf/training-reproduction/cgl/s4/runs/run-010/s4.json`               | Authoritative train/validation stream evidence PASS; artifact SHA-256 is recorded in its evidence manifest.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| S5    | `CUDA_VISIBLE_DEVICES="${RALF_GPU:?set RALF_GPU to one selected V100}" PARITY_REQUIRE=1 uv run --package ralf --extra training traingen fit --config models/ralf/configs/training/cgl.yaml --trainer.max_epochs=30`                                                                                                                                                                                                                                                     | `.cache/ralf/training-reproduction/cgl/s5/`                                   | Seeds 1-3 trained on both systems to 30 epochs (batch 32, torch 2.8.0+cu128, one V100 per run; package runs under `.cache/ralf/training-reproduction/cgl/s5/seed-<N>/` with launch manifests, vendor runs at pinned revision `c51db6032acbd0bd0ce72433becce08317e7874d` under an operator-selected run root outside the repository). The vendor launch sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` because the original targets torch older than 2.6, whose `torch.load` default reads its retrieval cache tables containing `defaultdict`; package launches set `num_workers=4` and `OMP_NUM_THREADS=4`, the original's effective values. The package seed-3 first attempt died when a co-tenant exhausted `/dev/shm`; its artifacts are retained beside the completed rerun. Results are in [Reproduction Results](#reproduction-results); tracked in the [RALF training reproduction issue #44](https://github.com/creative-graphic-design/design-generators/issues/44). |

`RALF_CACHE_DIR` in the table is the explicit environment variable shown in
[Regeneration Metadata](#regeneration-metadata). Set `RALF_GPU` to the selected
V100 index and choose a new `RALF_S3_OUTPUT` path before any S3 rerun.

After the semantic blank-line change to the training-path modules, the staged
ladder was rerun per the protocol's Regression Rule. Every rerun stage passed:
S0 `.cache/ralf/training-reproduction/cgl/s0-after-blank-lines-001/s0.json`
(SHA-256 `466c427251545dae96b4aa9fea2de8bbcc3e391324cf70c0141ced9e4670730b`,
initial-state digest identical to the accepted S0), S1
`s1-after-blank-lines-001/s1.json`
(`ee29e6fc45068f3cb6a99b67de5d1d7c8c95ef5e5c7a50da2a1dafa4937e3b9b`), S2
`s2-after-blank-lines-001/s2.json`
(`7d5ab67feda20b6d94280b6c4a6902c91f013f294d12dc6381e1065f1dc65480`), S4
`s4-after-blank-lines-002/s4.json`
(`f4538ddd758ea75cb06638405b8f73721ac8e30187c548697ef25841791d9a45`, stream
digest, checked batch count, checked sample count, and divergence fields
identical to the accepted S4 run-010), and the S3 artifact in the table above.

The 300-step real-scale synchronized lockstep probe ran on one V100 at the
recipe batch size 32, using 10 train batches per epoch over 30 epochs. All 300
steps record identical package and vendor state SHA-256 values with parameter
and optimizer maximum absolute differences of `0.0` and `first_divergence=null`;
peak allocated memory was 22.8 GB. The artifact is
`.cache/ralf/training-reproduction/cgl/probe/runs/after-blank-lines-001/trace/s3-trace.json`
(SHA-256 `8a535a96272d0a1463358290868edd5fca4ec2f1a5b2f014a911c3abe4cf3f35`).

## Reproduction Results

This matrix accounts for each checkpoint condition separately, so an unrun
condition cannot be mistaken for the unconditional staged evidence above.
Each row identifies the dataset, system, status, seed scope, metrics, loss
evidence, and retained artifact location; only the first CGL row has staged
records.

| Dataset           | System                     | Status                                                                                                                                                                                                                                                                                                                                                | Seed scope          | Primary metrics                                                                                                                                                | Loss evidence                                                                                      | Artifact summary                                                                        |
| ----------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| CGL unconditional | package/vendor staged path | `blocked (three seed pairs trained and evaluated under the original protocol and every package metric mean falls inside the vendor three-seed range; the coordinator's independent rerun and verdict are outstanding; tracked in the [RALF training reproduction issue #44](https://github.com/creative-graphic-design/design-generators/issues/44))` | `training-seed n=3` | Layout FID package 2.20/2.22/1.96 vs vendor 2.24/2.26/2.02; validity 0.999 both; overlap means equal at 0.4017; all fifteen evaluator metrics summarized below | Final validation cross-entropy package 2.420/2.447/2.418 vs vendor 2.440/2.436/2.414 from run logs | `.cache/ralf/training-reproduction/cgl/s5/` plus operator-root score files hashed below |
| CGL label         | package/vendor staged path | `not-yet-run (S5 pending; tracked in the [RALF training reproduction issue #44](https://github.com/creative-graphic-design/design-generators/issues/44))`                                                                                                                                                                                             | `training-seed n=1` | CGL staged evidence is unconditional only                                                                                                                      | Not run for this checkpoint                                                                        | `.cache/ralf/training-reproduction/cgl/`                                                |
| CGL label-size    | package/vendor staged path | `not-yet-run (S5 pending; tracked in the [RALF training reproduction issue #44](https://github.com/creative-graphic-design/design-generators/issues/44))`                                                                                                                                                                                             | `training-seed n=1` | CGL staged evidence is unconditional only                                                                                                                      | Not run for this checkpoint                                                                        | `.cache/ralf/training-reproduction/cgl/`                                                |
| CGL completion    | package/vendor staged path | `not-yet-run (S5 pending; tracked in the [RALF training reproduction issue #44](https://github.com/creative-graphic-design/design-generators/issues/44))`                                                                                                                                                                                             | `training-seed n=1` | CGL staged evidence is unconditional only                                                                                                                      | Not run for this checkpoint                                                                        | `.cache/ralf/training-reproduction/cgl/`                                                |
| CGL refinement    | package/vendor staged path | `not-yet-run (S5 pending; tracked in the [RALF training reproduction issue #44](https://github.com/creative-graphic-design/design-generators/issues/44))`                                                                                                                                                                                             | `training-seed n=1` | CGL staged evidence is unconditional only                                                                                                                      | Not run for this checkpoint                                                                        | `.cache/ralf/training-reproduction/cgl/`                                                |
| CGL relation      | package/vendor staged path | `not-yet-run (S5 pending; tracked in the [RALF training reproduction issue #44](https://github.com/creative-graphic-design/design-generators/issues/44))`                                                                                                                                                                                             | `training-seed n=1` | CGL staged evidence is unconditional only                                                                                                                      | Not run for this checkpoint                                                                        | `.cache/ralf/training-reproduction/cgl/`                                                |
| PKU unconditional | package                    | `not-yet-run (staged evidence pending; tracked in the [RALF training reproduction issue #44](https://github.com/creative-graphic-design/design-generators/issues/44))`                                                                                                                                                                                | Not run             | No staged evidence                                                                                                                                             | Not run                                                                                            | `.cache/ralf/training-reproduction/cgl/`                                                |
| PKU label         | package                    | `not-yet-run (staged evidence pending; tracked in the [RALF training reproduction issue #44](https://github.com/creative-graphic-design/design-generators/issues/44))`                                                                                                                                                                                | Not run             | No staged evidence                                                                                                                                             | Not run                                                                                            | `.cache/ralf/training-reproduction/cgl/`                                                |
| PKU label-size    | package                    | `not-yet-run (staged evidence pending; tracked in the [RALF training reproduction issue #44](https://github.com/creative-graphic-design/design-generators/issues/44))`                                                                                                                                                                                | Not run             | No staged evidence                                                                                                                                             | Not run                                                                                            | `.cache/ralf/training-reproduction/cgl/`                                                |
| PKU completion    | package                    | `not-yet-run (staged evidence pending; tracked in the [RALF training reproduction issue #44](https://github.com/creative-graphic-design/design-generators/issues/44))`                                                                                                                                                                                | Not run             | No staged evidence                                                                                                                                             | Not run                                                                                            | `.cache/ralf/training-reproduction/cgl/`                                                |
| PKU refinement    | package                    | `not-yet-run (staged evidence pending; tracked in the [RALF training reproduction issue #44](https://github.com/creative-graphic-design/design-generators/issues/44))`                                                                                                                                                                                | Not run             | No staged evidence                                                                                                                                             | Not run                                                                                            | `.cache/ralf/training-reproduction/cgl/`                                                |
| PKU relation      | package                    | `not-yet-run (staged evidence pending; tracked in the [RALF training reproduction issue #44](https://github.com/creative-graphic-design/design-generators/issues/44))`                                                                                                                                                                                | Not run             | No staged evidence                                                                                                                                             | Not run                                                                                            | `.cache/ralf/training-reproduction/cgl/`                                                |

### CGL Unconditional S5 Comparison

Each system trained seeds 1-3 to 30 epochs, and every resulting checkpoint was
evaluated with the original implementation's own chain: its inference
entrypoint with `cond_type=uncond`, `test_split=test`, and `top_k` sampling,
followed by its `eval.py` with the authors' FIDNet weights. Package
checkpoints enter that chain through the evaluator adapter, which removes the
single Lightning `model.` prefix and reuses the seed-matched vendor
`config.yaml`. Inference runs with `num_workers=0` because the entrypoint
holds every generated batch until it serializes results, and dataloader
workers would stage that accumulation in `/dev/shm` (measured 128 GiB in 40
minutes on this host); worker count does not change the outputs.

| Metric (test average)   | Package mean [min, max] | Vendor mean [min, max]  |
| ----------------------- | ----------------------- | ----------------------- |
| Layout FID              | 2.1271 [1.9648, 2.2164] | 2.1751 [2.0229, 2.2628] |
| Validity                | 0.9990 [0.9989, 0.9992] | 0.9991 [0.9988, 0.9992] |
| Alignment (LayoutGAN++) | 0.0021 [0.0020, 0.0022] | 0.0022 [0.0021, 0.0023] |
| Overlap (LayoutGAN++)   | 0.4017 [0.3970, 0.4042] | 0.4017 [0.3946, 0.4076] |
| Underlay (strict)       | 0.9761 [0.9746, 0.9769] | 0.9752 [0.9746, 0.9762] |
| Underlay (loose)        | 0.9925 [0.9920, 0.9934] | 0.9926 [0.9918, 0.9932] |
| Utilization             | 0.1970 [0.1963, 0.1978] | 0.1978 [0.1961, 0.1989] |
| Occlusion               | 0.1247 [0.1243, 0.1254] | 0.1252 [0.1249, 0.1257] |
| Unreadability           | 0.0177 [0.0177, 0.0178] | 0.0178 [0.0177, 0.0180] |
| Precision (layout)      | 0.9679 [0.9659, 0.9707] | 0.9658 [0.9643, 0.9678] |
| Recall (layout)         | 0.8917 [0.8869, 0.9000] | 0.8931 [0.8894, 0.8999] |
| Density (layout)        | 1.1270 [1.1138, 1.1398] | 1.1221 [1.1130, 1.1373] |
| Coverage (layout)       | 0.9492 [0.9459, 0.9526] | 0.9489 [0.9465, 0.9502] |
| R_shm (VGG distance)    | 14.281 [14.263, 14.294] | 14.326 [14.258, 14.373] |
| Overlay                 | 0.0051 [0.0047, 0.0059] | 0.0053 [0.0050, 0.0059] |

Every package mean falls inside the vendor three-seed range. The per-seed
score files (one YAML per run under each run root's `scores/` directory)
have these SHA-256 values: package seeds 1-3
`ee58b8ac4d25437e111173c31ebce39c18c288d565c8b0222d357248e95ce985`,
`2972dc798be946a8f0d073e8d18391121268d8e40e544ec50d98bd7e9d481577`,
`6a26d8a0a86ec4d335ed06f0d863548b9bc7996d2c157ea653fc6fd428d0ab35`; vendor
seeds 1-3
`c7e30d6171127a7d1d87c6b9b958d253bdabb5a8dbeadde646f0f649422539c6`,
`1f90ceba3cdac32a8250432e1a3d2fbf50eabb690bffadb5ca39bba4abb22c8e`,
`6113cedcc75854041ea693632c75f71dba0d42264eb9e46fa023cf10aef0ccc7`.

S4 run-010 compared the actual package `RalfDataModule.train_dataloader()` and
`val_dataloader()` with the vendor `DataLoader` and `collate_fn`. Train and
validation split membership matched exactly (48,544 and 6,002 ids,
respectively), with zero overlap. The package and vendor canonical stream
digest was the same: `664a421335f288c1a3beafc92f53d5d88ba685e5d393aea8fcc6eb484cdf3876`.
The loader digests were package
`2ef173685465a5810cfcaed76f2788cba85c2f48c85b5a83229cf004be1b4f3e` and vendor
`5970e735922f26706018496a90f4bf9cda40c6dea1dbde0f4e980b4446f92ab7`.

The corrected CGL S0 run-003 used the vendor-effective 30-epoch recipe and
passed with 44,386,946 parameters, 664 state-dict keys, and milestone 21. Its
candidate source digest is
`cf21004402441022542289debd19a6ac2d3cca5ad1fa3a8b406406222f921b6f`, its
effective config digest is
`be54c44b7e677a9aee75c7076ca2d20e04d4c56d58c3defc45ead4a452b93f61`, and its
pinned vendor revision is `c51db6032acbd0bd0ce72433becce08317e7874d`. The
recorded source digest identifies the dirty candidate tree used for this
diagnostic; it is not a claim that the artifact was generated from a clean
commit.

## Evaluator Adapter Diagnostic

The retained CPU diagnostic is a PASS for the vendor config, pickle loader, and
validity checks, and is not S-stage evidence. The narrow adapter under
`tests/vendor_parity` achieves that result by extracting the Lightning
`state_dict`, removing the single `model.` wrapper, writing a vendor raw
checkpoint beside its config, and mapping package `LayoutGenerationOutput`
normalized `xywh` boxes, labels, masks, and explicit sample ids to the vendor
pickle geometry fields. The retained JSON binds the reported PASS to the exact
source checkpoint, raw checkpoint, config, and pickle; their SHA-256 values
allow integrity checking.

| File                                                                     | Result or SHA-256                                                                        |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `.cache/ralf/training-reproduction/cgl/evaluator/run-001/evaluator.json` | PASS; contains the retained digest record                                                |
| Vendor raw checkpoint (`gen_final_model.pt`)                             | Included in the retained digest record                                                   |
| Vendor config (`config.yaml`)                                            | Loaded by `load_train_cfg`                                                               |
| Vendor pickle (`test_0.pkl`)                                             | PASS through `load_pkl` and `compute_validity`                                           |
| Vendor validity                                                          | PASS criterion met: validity `1.0` for both filtered samples; 664 state-dict keys loaded |

## Regeneration Metadata

The currently verified diagnostic environment is a pre-existing coherent
cu128 runtime supplied through an externally managed interpreter, with torch
`2.8.0+cu128`, on one selected Tesla V100-SXM2-32GB. The authoritative cache is
supplied read-only through `RALF_CACHE_DIR`; all accepted CGL evidence references
that environment variable. Current candidate metadata and retained artifact
records carry the relevant source, canonical-config, and pinned-vendor digests
needed to reproduce or audit the staged checks.

The currently selected authoritative cache is the user-provided local copy,
selected only through `RALF_CACHE_DIR`; no shared-mount dependency is part of
the rerun contract. Its content-role inventory is the CGL/PKU dataset trees,
retrieval-index tables, precomputed ResNet/FIDNet/DreamSim resources,
relationship and saliency data, evaluation features, and training logs. The
per-file SHA256 manifest is
`.cache/ralf/training-reproduction/cgl/provenance/local-cache-manifest-001.sha256`
(404 files; manifest SHA256
`17ce2b93df63ca8444b4be465f8b72d8a2ea9d2e480d98bb60f98de5a8150a96`). The
manifest is generated from the authoritative local copy and is not committed.

The repository-default torch and CUDA metadata remain unchanged for the V100
host constraint. RALF declares its runtime `jaxtyping` dependency in the
package metadata; `uv.lock` is intentionally unchanged in this checkpoint. To use the currently verified temporary interpreter,
set `RALF_AUDIT_PYTHON` and activate its containing environment before using
the member-scoped `uv run --active --no-sync` commands below. This is an
observed verification condition, not a package requirement. The package and
S0-S3 commands do not require `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`. The S4 vendor
loader command retains it because the pinned vendor path calls legacy
`torch.load` without an explicit `weights_only` argument while loading cached
retrieval/precomputed resources; it is scoped to that vendor-only boundary.

S4 run-010 evidence:

```text
artifact: .cache/ralf/training-reproduction/cgl/s4/runs/run-010/s4.json
artifact_sha256: 4c989f7e87444d2275d775c06fadcdcfbcff36acf18caca013f6ec156a7106c8
stdout: .cache/ralf/training-reproduction/cgl/s4/runs/run-010/stdout.log
stdout_sha256: c762713e38bdfff432a360550b7f96f2f9694af41bc9584cc2c4b117df36a349
package_canonical_stream_sha256: 664a421335f288c1a3beafc92f53d5d88ba685e5d393aea8fcc6eb484cdf3876
vendor_canonical_stream_sha256: 664a421335f288c1a3beafc92f53d5d88ba685e5d393aea8fcc6eb484cdf3876
package_loader_sha256: 2ef173685465a5810cfcaed76f2788cba85c2f48c85b5a83229cf004be1b4f3e
vendor_loader_sha256: 5970e735922f26706018496a90f4bf9cda40c6dea1dbde0f4e980b4446f92ab7
train_ids_sha256: 1bff3728181120eb8a1daae365a99a26788103501cb3835fc1360cf116307aed
validation_ids_sha256: 56aee7f12c369c9336e04665dc434cdf1b05455d6525e68d2a74584fefbdcdf2
```

The corrected 30-epoch S3 evidence spans two natural runs and one synchronized
run. The natural runs record a first divergence in the raw gradients of the
first compared step, so they do not pass on their own; the synchronized run
supplies the contract-internal agreement that makes the bounded pass:

```text
run-016_evidence_mode: natural
run-016_trace_sha256: 878c76d2fb1872d0cf6fd981fbd4d46f767062c22d9665da79e57734793bacf1
run-016_peak_memory_allocated_bytes: 14019053056
run-016_first_divergence: S3.epoch[0].batch[1].raw_gradients.encoder.extractor.body.bn1.weight (max_abs_diff 9.832845535129309e-06)
run-017_evidence_mode: natural
run-017_trace_sha256: f059356d9c3a32c5acf2a8f472e0388cb2bc43ec94ae5c35601afddb1f7569f9
run-017_peak_memory_allocated_bytes: 11647124992
run-017_first_divergence: S3.epoch[0].batch[1].raw_gradients.encoder.extractor.body.conv1.weight (max_abs_diff 4.539033398032188e-06)
run-017_runtime_seconds: 633.6163005884737
run-018_evidence_mode: synchronized
run-018_status: PASS
run-018_trace_sha256: 09ffe676ffab5252e31812fccbeac8095711124e3f15f0ec13086a2f656f617c
run-018_lockstep_steps: 90/90
run-018_first_divergence: null
run-018_peak_memory_allocated_bytes: 24506435072
run-018_runtime_seconds: 1734.4115353412926
run-018_scheduler_milestones: [21]
```

Superseded S3 run-012 evidence remains under
`.cache/ralf/training-reproduction/cgl/s3/runs/run-012/`; it is retained for
provenance but invalidated by the 30-epoch correction:

```text
trace_sha256: d29f0c259eaacad7affa8cffd9b2e55942fdf54a25be19afcc2d28af0ca44094
stdout_sha256: 45aa1a4122261d3608f19800a0c051ae7b277650a387af4c5c90231860504419
checkpoint_sha256: 8b5c76c0f105c25b3b061d77c1136ddc0a0ce90884cdfb3257126d72756064af
runtime_seconds: 3807.7891788799316
```

The package-path health check after the import/runtime changes is recorded
separately from accepted stage evidence:

```text
artifact: .cache/ralf/training-reproduction/cgl/import-env-20260815-001/s0.json
artifact_sha256: a76a0d6a0adb828dbee860b8bc1f4256e64ac7091bd8fbd2516b24ce768be294
status: PASS
selected_gpu: 1 (Tesla V100-SXM2-32GB)
torch: 2.8.0+cu128
vendor_revision: c51db6032acbd0bd0ce72433becce08317e7874d
```

This health check confirms the current member-scoped import path and static
state, but does not replace the corrected S0 or retained S1/S2 and accepted S4
artifacts above; the superseded S0 and S3 artifacts remain invalidated.

## Training Commands

Set the diagnostic paths explicitly before rerunning a staged check. The
commands use one GPU and fail closed when local parity assets are unavailable.

```bash
: "${RALF_AUDIT_PYTHON:?set RALF_AUDIT_PYTHON to the verified diagnostic interpreter}"
: "${RALF_CACHE_DIR:?set RALF_CACHE_DIR to the authoritative read-only cache}"
RALF_AUDIT_VENV="${RALF_AUDIT_PYTHON%/bin/python}"
source "$RALF_AUDIT_VENV/bin/activate"
```

Run the focused package training checks.

```bash
PARITY_REQUIRE=1 uv run --active --no-sync --package ralf --extra training --extra vendor \
  --with pytest --with 'beartype>=0.22.9,<0.23' pytest \
  models/ralf/tests/test_training.py \
  models/ralf/tests/vendor_parity/test_training_harness.py -q
```

Regenerate CGL S0-S2 into new `.cache` directories rather than overwriting
accepted artifacts.

```bash
CUDA_VISIBLE_DEVICES=1 PARITY_REQUIRE=1 \
  uv run --active --no-sync --package ralf --extra training --extra vendor python \
  models/ralf/tests/vendor_parity/run_training_stages.py \
  --stage S1 --dataset cgl --cache-dir "$RALF_CACHE_DIR" \
  --output .cache/ralf/training-reproduction/cgl/<new-run>/s1.json \
  --steps 1 --batch-size 32 --seed 1
```

Use the same command with `--stage S2` for the one-step optimizer check.

For that authorized run, set `RALF_GPU`, `RALF_CACHE_DIR`, and `RALF_S3_OUTPUT`
as shown in the Stage Evidence table and run the complete command below. The
run follows the protocol's layered S3 rule: the natural trajectory passes on its
own when every step stays inside the S0-S2 contract, and otherwise the retained
natural record plus a contract-internal synchronized layer is a bounded pass.
The runner records the
nested `traingen fit` command and its artifacts. The recipe supplies 30 epochs
and the runner supplies the production limits, callbacks, logger, and cache
paths.

```bash
: "${RALF_GPU:?set RALF_GPU to one selected V100 index}"
: "${RALF_CACHE_DIR:?set RALF_CACHE_DIR to the authoritative read-only cache}"
: "${RALF_S3_OUTPUT:?set RALF_S3_OUTPUT to a new repository-relative JSON path}"
CUDA_VISIBLE_DEVICES="$RALF_GPU" PARITY_REQUIRE=1 \
  uv run --active --no-sync --package ralf --extra training --extra vendor python \
  models/ralf/tests/vendor_parity/run_training_stages.py \
  --stage S3 --dataset cgl --cache-dir "$RALF_CACHE_DIR" \
  --output "$RALF_S3_OUTPUT" --batch-size 32 --seed 1
```

Do not substitute a package model with the original model in the package
Trainer.

Run the accepted CGL S4 loader-stream check into a fresh artifact directory.

```bash
CUDA_VISIBLE_DEVICES=1 PARITY_REQUIRE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  uv run --active --no-sync --package ralf --extra training --extra vendor python \
  models/ralf/tests/vendor_parity/run_training_stages.py \
  --stage S4 --dataset cgl --cache-dir "$RALF_CACHE_DIR" \
  --output .cache/ralf/training-reproduction/cgl/s4/runs/<new-run>/s4.json \
  --steps 8 --batch-size 32 --seed 1
```

The 300-step real-scale lockstep probe evidence is recorded under
[Stage Evidence](#stage-evidence), and the three-seed S5 training and
evaluation results are recorded under
[Reproduction Results](#reproduction-results). The coordinator's independent
parity rerun and verdict remain outstanding.

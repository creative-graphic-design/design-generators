---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutFormerPP
---

# LayoutFormer++ Training

The retained LayoutFormer++ candidate has accepted S0-S4 evidence for all
twelve RICO25 and PubLayNet recipe families. PubLayNet label-conditioned
`gen_t` also has documented-bounded-deviation practical status, not a `PASS`:
under one common 10,998-item standalone test protocol, the package three-seed
mean mIoU is `0.32663`, which is `0.0039` below the clean original-code
three-seed minimum. Five of seven reported metrics place the package mean
inside the clean original-code range. Alignment exceeds that range by only
`1.2e-6`. The authors' released-checkpoint mIoU is inside the clean
original-code range.

This S5 evidence covers only PubLayNet label-conditioned `gen_t`. The package
and clean original-code runs used three distinct numeric seeds, but their
entrypoints apply those seeds at different construction points. Their test
results are protocol-comparable, not seed-paired training trajectories. The
other eleven recipe families remain untested at S5, so no general
LayoutFormer++ trained-checkpoint reproduction is claimed.

Run commands from the repository root. Keep generated evidence, logs, data, and
checkpoints under `.cache/layoutformerpp/`.

## Install

```bash
uv sync --package layoutformerpp --extra training
```

Install the original-code adapter dependencies only for static parity checks.

```bash
uv sync --package layoutformerpp --extra training --extra vendor
```

The package runtime and training namespace do not import DeepSpeed or Detectron2.
The S0 vendor-parity harness reads the original requirements pin and constructs
the real DeepSpeed 0.5.10 `WarmupLR` class in an isolated `uv --with` overlay;
the original package import is incompatible with the currently verified Torch
because it imports the removed `torch._six` module. No dependency-design change
is claimed. S3 and S4 use bounded slices of the authoritative original processed
RICO25 and PubLayNet splits supplied through `LAYOUTFORMERPP_PARITY_DATA_ROOT`;
no full dataset download is part of this candidate.

## GPU Environment

The repository default Torch build is `cu130`, which is diagnostic-only on the
currently verified Tesla V100 setup. Use a temporary environment outside the
repository with the driver-compatible `cu126` wheels; do not change
`pyproject.toml` or `uv.lock` for this local runtime replacement.

```bash
export LAYOUTFORMERPP_PARITY_ENV="${LAYOUTFORMERPP_PARITY_ENV:?set a temporary path outside this repository}"
uv venv --python 3.11 "$LAYOUTFORMERPP_PARITY_ENV"
uv pip install --python "$LAYOUTFORMERPP_PARITY_ENV/bin/python" \
  torch==2.12.0 torchvision==0.27.0 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$LAYOUTFORMERPP_PARITY_ENV/bin/python" \
  -e lib/laygen \
  -e "lib/traingen[lightning]" \
  -e "lib/traingen-parity[lightning]" \
  -e "models/layoutformerpp[training,vendor]" \
  pytest
export LAYOUTFORMERPP_PARITY_PYTHON="$LAYOUTFORMERPP_PARITY_ENV/bin/python"
CUDA_VISIBLE_DEVICES=0 "$LAYOUTFORMERPP_PARITY_PYTHON" -c \
  'import torch; assert torch.__version__ == "2.12.0+cu126"; assert torch.version.cuda == "12.6"; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability(0) == (7, 0); torch.zeros(1, device="cuda:0")'
```

The verified setup is one physical V100 exposed as logical `cuda:0`; CPU is
diagnostic-only for these stages.

## Data

S0-S1 use the accepted source-shaped fixed fixtures. S3-S4 use the pinned
original processed split files through
`LAYOUTFORMERPP_PARITY_DATA_ROOT` and compare the original loader with the
package-local DataModule/DataLoader.

| Dataset | Source | Config or path |
| --- | --- | --- |
| RICO25 | `creative-graphic-design/Rico` | `ui-screenshots-and-hierarchies-with-semantic-annotations`; processed `pre_processed_20_25` slice used for S3-S4 |
| PubLayNet | `creative-graphic-design/PubLayNet` | processed `pre_processed_20_5` slice used for S3-S4 |

RICO25 persists separate public zero-based and sequence one-based maps. The maps
are joined by normalized semantic name and hashed; integer arithmetic is never
used for translation. The public/config/Hub slug is `rico25`; `rico` is accepted
only at named original CLI/cache boundaries.

## Configs

The twelve faithful LightningCLI YAMLs live under
`models/layoutformerpp/configs/training`. The package-local DataModule is the
existing production training path used by `traingen fit` and the bounded
loader-parity gate.

| Config | Dataset | Seed mode | Purpose |
| --- | --- | --- | --- |
| `rico25_{label,label_size,relation,refinement,completion,unconditional}.yaml` | RICO25 | original CLI late-seed behavior recorded | six faithful static recipes |
| `publaynet_{label,label_size,relation,refinement,completion,unconditional}.yaml` | PubLayNet | original CLI late-seed behavior recorded | six faithful static recipes |

PubLayNet relation is one family with task order `refinement,gen_ts,gen_t,completion,ugen,gen_r`
and package task-ID tuple `0,4,3,1,2,5`; its partition buckets are
`-1,-1,-2,0,0,-3`. Each single-task family also persists its package task ID:
refinement `0`, completion `1`, unconditional `2`, label `3`, label-size `4`,
and relation `5`. S0 compares those package-owned values with task IDs extracted
independently from the original implementation. Diagnostic deterministic and
smoke configs are deferred.

## Scheduler and Recipe Notes

The reference mode is plain PyTorch `basic`; the distributed DeepSpeed trainer
remains secondary, provenance-gated evidence. The basic path nevertheless
imports DeepSpeed's pinned `WarmupLR`, so S0 constructs that real scheduler class
through the vendor-only compatibility probe and compares it with the package
scheduler. Every faithful recipe uses Adam at `1e-4` with default betas and
epsilon, zero weight decay, accumulation one, and no clipping, EMA, or AMP.
`vendor_effective_cross_entropy` includes pad targets; the runtime model's
ordinary forward loss remains pad-masked.

The logarithmic WarmupLR starts at index `-1` without an eager scheduler step.
The first optimizer update uses `1e-4`; its post-update scheduler call writes
`0`; subsequent post-update values are `1e-4 * log(step + 1) / log(W)` until the
warmup maximum. Lightning receives the scheduler with `interval: step`, and the
module advances it exactly once after each optimizer update.

Validation occurs every 20 RICO25 epochs or 50 PubLayNet epochs. The selected
checkpoint is the minimum aggregate evaluation-loss epoch, matching the
original `best_epoch` decision. Original basic checkpoints contain model state
only, so exact optimizer/scheduler/RNG resume is not a parity claim.

## Seed Policy

The PubLayNet result uses `training-seed n=3` for each system and evaluation
seed `500` for every checkpoint. It is not a seed-paired claim. Lightning
applied package `seed_everything=0,1,2` before model and DataModule
construction. The original CLI constructs the tokenizer, model, optimizer,
scheduler, and datasets first; its trainer then applies `--seed 0,1,2` during
experiment setup. The official `publaynet_gen_t.sh` has no `--seed` argument;
the clean r6 commands added those three values.

The package runs are distinct. Their first recorded training losses are
`7.39870`, `6.61055`, and `7.31237`, and their selected checkpoints have
different SHA-256 values. All three recorded `data.init_args.seed=0`, but that
field is inert: the DataModule stores it and no runtime path reads it. The
production loader uses `generator=None` with `num_workers=0`, so its
`RandomSampler` derives its private shuffle seed from the global Torch state
after package model construction. The S4 regression records this RNG movement
and proves that reseeding immediately before loader construction produces a
different first sample order.

The systems therefore share evaluator code, constrained decoding, test split,
and evaluation seed, while model initialization and training streams are not
paired. A complete twelve-family training-seed claim would require 12 families
by 2 systems by 3 training seeds, or 72 full runs; PubLayNet relation is counted
once.

## Validation Stages

The stages below follow `docs/training-reproduction.md`.

| Stage | Scope | Purpose |
| --- | --- | --- |
| S0 | Static config and initialized state parity | all twelve recipe, topology, state, vocab, label-map, loss, optimizer, scheduler, seed-order, and checkpoint facts |
| S1 | Fixed-batch pre-optimizer trace parity | all twelve families pass on the selected CUDA device |
| S2 | One optimizer-step parity | accepted: authoritative scheduler, backward/optimizer state, post-step parameters, LR/cadence, RNG, and first divergence matched for all twelve recipes |
| S3 | Short deterministic multi-batch run | accepted: 12-family manual numerical lockstep passed; ten representative real-data `traingen fit` runs exercised production scheduler, logging, validation, and `ModelCheckpoint` wiring |
| S4 | Deterministic loader stream | accepted for isolated same-boundary reseeding; a production-order regression records that package model construction advances RNG before the loader while the original seed is applied after model construction |
| S5 | Full-run statistical comparison | `CHECK` for PubLayNet label-conditioned `gen_t`: documented bounded deviation under a common standalone test protocol; the other eleven families remain pending |

## Stage Evidence

| Stage | Command | Artifact | Result |
| --- | --- | --- | --- |
| S0 | `PARITY_REQUIRE=1 uv run --package layoutformerpp --extra training --extra vendor --no-sync pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s0 -rs -q` | `.cache/layoutformerpp/s0/static-parity.json` | PASS: 39 tests, all twelve families, zero skips; pinned original revision `1498ff300710b4fc204aece537582d37ca447fc7`; independent task-ID, topology, scheduler, loss, and state checks. |
| S1 | `CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 "$LAYOUTFORMERPP_PARITY_PYTHON" -m pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s1 -rs -q` | `.cache/layoutformerpp/s0/static-parity.json#s1` | PASS: 12 tests, zero skips on physical GPU 0 as logical `cuda:0` (`torch 2.12.0+cu126`, CUDA 12.6, SM 7.0, float32); `rtol=1e-4`, `atol=1e-5`; first divergence `null`; max absolute error `3.814697265625e-06`; max relative error `8.155166142387316e-08`. |
| S2 | `CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 "$LAYOUTFORMERPP_PARITY_PYTHON" -m pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s2 -rs -q` | `.cache/layoutformerpp/s0/static-parity.json#s2` | PASS: 12 tests, zero skips; one real backward and optimizer step per recipe matched gradients, clipping behavior, optimizer state, post-step parameters, authoritative scheduler cadence/LR, RNG, and first divergence within `rtol=1e-4`, `atol=1e-5`. |
| S3 | `CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT="${LAYOUTFORMERPP_PARITY_DATA_ROOT:?set authoritative original processed data root}" "$LAYOUTFORMERPP_PARITY_PYTHON" -m pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s3 -rs -q` | `.cache/layoutformerpp/s0/static-parity.json#s3` | PASS: 22 tests, zero skips: 12 manual numerical lockstep cases plus 10 production `traingen fit` representative cases (`rico25_label`, `rico25_label_size`, `rico25_relation`, `rico25_refinement`, `rico25_completion`, `rico25_unconditional`, `publaynet_label`, `publaynet_label_size`, `publaynet_relation`, `publaynet_refinement`). Every console run exited 0, reached `global_step=2` and two optimizer steps, recorded scheduler `last_epoch=1` and the expected post-step LR, delivered `train_loss`/`val_loss` to the CSV logger, and selected a real `ModelCheckpoint` file. The ordinary 12-YAML matrix guard proves shared Trainer/DataModule/model/checkpoint wiring for all recipes; manual numerical parity remains the separate all-family claim, while PubLayNet completion/unconditional are not separately claimed as console-fit runs. |
| S4 | `CUDA_VISIBLE_DEVICES="" PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT="${LAYOUTFORMERPP_PARITY_DATA_ROOT:?set authoritative original processed data root}" uv run --package layoutformerpp --extra training --extra vendor --no-sync pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s4 -rs -q` | `.cache/layoutformerpp/s0/static-parity.json#s4` | CHECK: 12 isolated same-boundary loader comparisons pass; the focused PubLayNet production-order regression records RNG after seed application, model construction, and the first loader batch, then proves that production order differs from the isolated loader reseed path. This bounds S4 and rejects a seed-paired trajectory claim. |
| S5 | `CUDA_VISIBLE_DEVICES=<gpu-index> /tmp/layoutformerpp-s1-cu126-BLnm3l/bin/python vendor/ms-layout-generation/LayoutFormer++/src/main.py --test --dataset publaynet --tasks gen_t ... --eval_seed 500 --eval_ckpt_tag <epoch_49\|final> --enable_task_measure --load_vocab` | `.cache/layoutformerpp/s5/eval/publaynet_label/` | CHECK: package, clean original-code r6, and authors' released checkpoints used the same constrained standalone evaluator and complete 10,998-item test split. PubLayNet `gen_t` has documented bounded deviation; the other eleven families remain pending. |

The loader-based 300-step real-scale lockstep diagnostic passed for the
`rico25_label` recipe after S0-S4. It recorded per-step loss, gradient norm,
parameter difference, scheduler/LR, loader order, and RNG/state hashes for 300
steps; first divergence was `null`. This remains diagnostic preflight only and
is not a trained-checkpoint, training-seed, or S5 claim.

The production `traingen fit` runs are wiring evidence only, not an additional
12-family numerical parity claim. They used the package
`LayoutFormerPPDataModule` and `LayoutFormerPPTrainingModule` with the
authoritative real-data root. The ten representatives cover both datasets,
all six conditioning paths, the distinct train/evaluation batch branches,
PubLayNet label-size flags, and the PubLayNet relation multitask partition
path. Native 20/50 validation cadence values are statically validated from
YAML; runtime validation/logging/checkpoint wiring was exercised at bounded
cadence 1. The ordinary 12-YAML recipe guard proves that all
recipe configs select the same Trainer/DataModule/model/checkpoint wiring
branch; the two PubLayNet families not directly run as console representatives
remain covered by the separate 12-family manual numerical gate only.

## Reproduction Results

PubLayNet label-conditioned `gen_t` has documented-bounded-deviation practical
status under `training-seed n=3` per system and one fixed evaluation seed. This
is the only family with S5 evidence. The clean comparison does not establish
strict seed pairing because the two training entrypoints apply their numeric
seeds at different construction points.

| Dataset | System | Status | Seed scope | Primary metrics | Loss evidence | Artifact summary |
| --- | --- | --- | --- | --- | --- | --- |
| RICO25 label | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 label | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 label-size | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 label-size | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 relation | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 relation | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 refinement | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 refinement | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 completion | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 completion | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 unconditional | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| RICO25 unconditional | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet label | clean original code r6 | `s5-practical-reproduction` | training-seed n=3; evaluation seed 500; not seed-paired to package construction | mIoU `0.33282` mean, `0.00365` sample std; full metrics below | epoch 49 selected by minimum original evaluation loss | `.cache/layoutformerpp/s5/publaynet_label/vendor-seed{0,1,2}-r6/`; `.cache/layoutformerpp/s5/eval/publaynet_label/vendor-r6-test/r1/` |
| PubLayNet label | package | `s5-practical-reproduction` | training-seed n=3; evaluation seed 500; not seed-paired to original construction | mIoU `0.32663` mean, `0.0039` below original seed minimum; full metrics below | epoch 49 selected by minimum package `val_loss` | `.cache/layoutformerpp/s5/publaynet_label/package-seed{0,1,2}/`; `.cache/layoutformerpp/s5/eval/publaynet_label/seed{0,1,2}-r1/` |
| PubLayNet label-size | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet label-size | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet relation | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet relation | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet refinement | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet refinement | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet completion | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet completion | package | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet unconditional | original | `not-yet-run (S0-S4 accepted; 300-step diagnostic only; S5 stopped)` | no training seed | not run | S0-S4 stage evidence; diagnostic pre-S5 probe only | `.cache/layoutformerpp/s0/` |
| PubLayNet unconditional | package | `not-yet-run (S0-S4 accepted; S5 pending)` | no training seed | not run | S0-S4 stage evidence only | `.cache/layoutformerpp/s0/` |

### Clean PubLayNet `gen_t` comparison

All seven evaluations used the pinned original evaluator with constrained
decoding, evaluation seed `500`, and the complete 10,998-item PubLayNet test
stream. The package rows evaluate the selected Lightning checkpoints after
stripping the `model.` prefix into the original checkpoint shape. The clean
original-code r6 and authors' release rows evaluate their original-shaped
checkpoints directly. Every row uses the checkpoint's own matching vocabulary;
all vocabularies have SHA-256
`b35377045c8815cdd073583e49e6fab855e62a5019723f6e8b40a9cbb8e87e61`.

| System | Seed | mIoU | FID | Alignment | Overlap | Bbox accuracy | Label accuracy | Violation rate | Metrics SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Package | 0 | 0.326212555 | 16.199593947 | 0.000179652 | 0.009041049 | 0.132921308 | 0.999999940 | 0 | `11a7eb8aa78d4e37bad35d946501cbd97e32f2c208043841d6f72af76f09d81f` |
| Package | 1 | 0.326852250 | 15.783900173 | 0.000205534 | 0.008726429 | 0.134756923 | 0.999999940 | 0 | `c421259884a03e8e7b32f63904c94f19ba609c7c014d8ac42b1e1ba27423d83a` |
| Package | 2 | 0.326826790 | 17.345766941 | 0.000193734 | 0.007683629 | 0.132045463 | 0.999999940 | 0 | `821bed52687f056ef720f289c4cb42d0031c14ff683ebb4c0acd0b7b2b38ac03` |
| Package mean | - | 0.326630532 | 16.443087020 | 0.000192973 | 0.008483702 | 0.133241231 | 0.999999940 | 0 | - |
| Clean original code r6 | 0 | 0.330498997 | 16.857438909 | 0.000185820 | 0.008018690 | 0.133104220 | 0.999999940 | 0 | `593bcfac1d72ca45e07b8aef8a0b32baf8e9b05a27c5c4125aa4dba8386ac2bc` |
| Clean original code r6 | 1 | 0.337028043 | 15.467105644 | 0.000187255 | 0.009885931 | 0.136743158 | 0.999999940 | 0 | `1c9e732b5477d5b22bbd01af36245881f12640804e6a91501a6328dc14ddfe47` |
| Clean original code r6 | 2 | 0.330921058 | 15.104894274 | 0.000191818 | 0.009708703 | 0.137130514 | 0.999999940 | 0 | `021ae24fb18dd4c59ac8f00143345c9da8eab99223d43cf91bad6bf006622d94` |
| Clean original code r6 mean | - | 0.332816033 | 15.809812942 | 0.000188298 | 0.009204441 | 0.135659297 | 0.999999940 | 0 | - |
| Authors' released checkpoint | - | 0.334915109 | 14.074759770 | 0.000197474 | 0.008215016 | 0.135372370 | 0.999999940 | 0 | `9b4efaa76b5b869d285da7c4c20e07b971144aba47e58eb096c916a62e3c13ed` |

| Metric | Package mean | Clean original-code mean | Original-code seed range | Package mean inside range? |
| --- | ---: | ---: | ---: | --- |
| mIoU | 0.326630532 | 0.332816033 | 0.330498997 to 0.337028043 | No, `0.003868465` below the minimum |
| FID | 16.443087020 | 15.809812942 | 15.104894274 to 16.857438909 | Yes |
| Alignment | 0.000192973 | 0.000188298 | 0.000185820 to 0.000191818 | No, `0.000001156` above the maximum |
| Overlap | 0.008483702 | 0.009204441 | 0.008018690 to 0.009885931 | Yes |
| Bbox accuracy | 0.133241231 | 0.135659297 | 0.133104220 to 0.137130514 | Yes |
| Label accuracy | 0.999999940 | 0.999999940 | 0.999999940 to 0.999999940 | Yes |
| Violation rate | 0 | 0 | 0 to 0 | Yes |

The package mean falls inside the clean original-code range for five of seven
metrics. Its mIoU is outside that range, and the result does not support an
unqualified parity `PASS`. The alignment miss is about `1.2e-6`, which is
negligible at the reported scale. The authors' released-checkpoint mIoU
`0.334915109` lies inside the clean r6 seed range and supports using r6 as the
original-code reference.

| System | Seed | Training checkpoint SHA-256 | Evaluated checkpoint SHA-256 |
| --- | ---: | --- | --- |
| Package | 0 | `8631d408ecc554688b3cffb2e759d0be5c8e96a773495371cbb7e2e57f36908c` | `f4a4390502231fd59a2ed21bf30c0960c13f7e7efde61575ca11097c4b3074f1` |
| Package | 1 | `fddbe8ea9ff3abeddab58e950a172404179692e97dbddc5f30aa640edbfa814d` | `9b0413b150dba714b0cda2b7cb7dc521640646a10ee7f90ab7a2b7519c361c34` |
| Package | 2 | `e705b2ff999d7e2c9315c15804568926a240d45f1841df184d166d958987195c` | `4fd630216025a972a52201db9c627b8681d1555fd3ed12b791ab5b0508a76d21` |
| Clean original code r6 | 0 | `3530818261703d77ef197158977c687f40cc494a30c0f8da0c1641b3a343b31b` | same as training checkpoint |
| Clean original code r6 | 1 | `e5a1c5909710ad74ab766b43e23d41b633136d866bb189ee8d6fe8bdf326f5e5` | same as training checkpoint |
| Clean original code r6 | 2 | `afef4f829a498b4ac05b59e3664353bfa1ff40d5e8d57f6168dbde8507220374` | same as training checkpoint |
| Authors' released checkpoint | - | `3c5ef9cdb28d606fda778f72ecf792609627f42660d7ec44f688c7af78fb8981` | same as release checkpoint |

### Invalidated r4/r5 original-code runs

The earlier `vendor-seed0-r4`, `vendor-seed1-r5`, and `vendor-seed2-r5` runs
set `gen_t_add_unk_token=true`. The official `publaynet_gen_t.sh` leaves that
flag false. Those runs trained on a different input serialization and produced
standalone mIoU values `0.200176`, `0.204640`, and `0.185452`. They are invalid
as original-code references and do not indicate a package failure. The clean r6
runs removed the flag, used the official effective settings, and reproduced the
authors' released-checkpoint mIoU range.

## Regeneration Metadata

The authoritative local manifest contains the worktree commit, exact
original-source revision, reviewed boundary hashes, RICO25 map hash,
recipe/test/document hashes, S0-S4 command results, the production Trainer
wiring smoke, the loader-based 300-step diagnostic, pinned DeepSpeed source
hash, and late-seed/captured-initialization contract at:

```text
.cache/layoutformerpp/s0/static-parity.json
```

The manifest's `manifest_generation` object is the single machine-readable
recipe for reproducing its aggregate Python/YAML digest; do not create a second
manifest or helper.

The [issue #265 evidence comment](https://github.com/creative-graphic-design/design-generators/issues/265#issuecomment-5264137469)
is the durable record for the rejected predecessor candidate, not the current
manifest. The current S0-S4 evidence uses the production package DataModule
and the pinned original loaders. The 300-step artifact is diagnostic only.
PubLayNet `gen_t` evaluation artifacts live under
`.cache/layoutformerpp/s5/eval/publaynet_label/`; S5 remains pending for the
other eleven families.

Official Lightning/PyTorch documentation and the pinned GitHub source were
consulted before the loader correction. The implementation follows the
documented DataModule/DataLoader ownership and the original split/sampler path;
vendor code remains gated under `tests/vendor_parity`.

## Training Commands

Rerun the ordinary static checks.

```bash
uv run --package layoutformerpp --extra training --no-sync pytest \
  models/layoutformerpp/tests -m "not vendor_parity and not integration" -q
```

Rerun S0 original-code parity with missing assets treated as failures.

```bash
PARITY_REQUIRE=1 \
  uv run --package layoutformerpp --extra training --extra vendor --no-sync pytest \
  models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py \
  -m "vendor_parity and training" -k s0 -rs -q
```

Run S1-S4 after the compatible environment setup above. The canonical
interpreter is `LAYOUTFORMERPP_PARITY_PYTHON`; set
`LAYOUTFORMERPP_PARITY_DATA_ROOT` to the authoritative processed source tree
for S3-S4. Keep `PARITY_REQUIRE=1`; missing source assets must fail the run.

`traingen fit` uses the package-local DataModule. Checkpoint conversion and S5
commands for the other eleven families remain deferred.

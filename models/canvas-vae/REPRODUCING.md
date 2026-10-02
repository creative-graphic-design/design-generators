# Reproducing CanvasVAE Agreement Checks

These commands rerun the agreement checks between this package and the [original CanvasVAE TensorFlow code](https://github.com/CyberAgentAILab/canvas-vae): RICO preparation, reference traces from the original trainer, the parity suite, checkpoint conversion, and a local `from_pretrained` smoke test. Run them from the repository root after `git submodule update --init vendor/canvas-vae`. Generated data, traces, and checkpoints stay under `.cache/canvas-vae/`.

Workflow order: download and prepare RICO, generate original-code references, run the parity suite, convert a checkpoint, then smoke-test `from_pretrained` local loading.

CanvasVAE has no released weights, so agreement is checked on the training path. The stages S0-S4 are defined in the [training reproduction protocol](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md); [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) records their results.

The `vendor` extra installs TensorFlow 2.15.1 and Apache Beam next to PyTorch. Original-code commands disable TF32 with `NVIDIA_TF32_OVERRIDE=0`; the currently verified setup runs every command on a CPU-only runtime with `CUDA_VISIBLE_DEVICES=""`. On a GPU host, set `CUDA_VISIBLE_DEVICES=<gpu-index>` to one free device instead.

## Prepare data

```bash
uv sync --package canvas-vae --extra training --extra vendor
uv run --package canvas-vae models/canvas-vae/scripts/download_rico.py
uv run --package canvas-vae models/canvas-vae/scripts/prepare_rico.py
```

`download_rico.py` fetches the RICO semantic-annotation archive (MD5 `5dd3372e2d99b342958136e394d4de79`) to `.cache/canvas-vae/data/semantic_annotations.zip`. `prepare_rico.py` writes the package splits to `.cache/canvas-vae/data/rico/`.

## Generate original-code references

```bash
export NVIDIA_TF32_OVERRIDE=0 CUDA_VISIBLE_DEVICES=""
uv run --package canvas-vae --extra vendor models/canvas-vae/scripts/generate_vendor_reference.py beam
uv run --package canvas-vae --extra vendor models/canvas-vae/scripts/generate_vendor_reference.py trace
uv run --package canvas-vae --extra vendor models/canvas-vae/scripts/generate_vendor_reference.py trace --output-dir .cache/canvas-vae/reference/trace-repeat
uv run --package canvas-vae --extra vendor models/canvas-vae/scripts/generate_vendor_reference.py stream
```

`beam` runs the original Apache Beam preprocessing into `.cache/canvas-vae/original/data/rico/`. `trace` exports static configuration, initial weights, the first 50 training batches with their posterior noise, a fixed-batch forward and one-step trace, and a 50-step trajectory to `.cache/canvas-vae/reference/trace/`; the repeated trace measures the original's run-to-run envelope. `stream` dumps every original record and the first 90 training, 6 validation, and 6 test batches of the original data streams to `.cache/canvas-vae/reference/stream/`.

## Run the parity suite

```bash
PARITY_REQUIRE=1 CUDA_VISIBLE_DEVICES="" uv run --package canvas-vae --extra training --extra vendor --with pytest pytest models/canvas-vae/tests/vendor_parity -m "vendor_parity and training" -rs
```

`PARITY_REQUIRE=1` turns missing references into failures. Measured differences are written to `.cache/canvas-vae/reference/reports/`.

## Convert a checkpoint and smoke-test loading

```bash
uv run --package canvas-vae --extra vendor models/canvas-vae/scripts/convert_original_checkpoint.py --checkpoint .cache/canvas-vae/reference/trace/initial/initial.ckpt --vocabulary .cache/canvas-vae/original/data/rico/vocabulary.json --output-dir .cache/canvas-vae/converted/initial
uv run --package canvas-vae models/canvas-vae/scripts/smoke_from_pretrained.py --checkpoint .cache/canvas-vae/converted/initial
```

The same converter accepts `final.ckpt` from an original training run and a Lightning `last.ckpt` from `traingen fit`.

# Reproducing CanvasVAE Agreement Checks

These commands rerun the agreement checks between this package and the [original CanvasVAE TensorFlow code](https://github.com/CyberAgentAILab/canvas-vae): RICO preparation, reference traces from the original trainer, the parity suite, checkpoint conversion, and a local `from_pretrained` smoke test. Run them from the repository root after `git submodule update --init vendor/canvas-vae`. Generated data, traces, and checkpoints stay under `.cache/canvas-vae/`.

Workflow order: download and prepare RICO, generate original-code references, run the parity suite, convert a checkpoint, then smoke-test `from_pretrained` local loading.

## Crello v1 data contract

This section is for maintainers reproducing the Crello v1 data path. It pins the input schema and deterministic IDs used to compare the original preprocessing with the package loader; it does not claim a Crello model or trained-encoder result.

Use the original v1 archive at `https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip` from vendor commit [`bc1e2072ba3a253f1b099e8b0c604f6051e787da`](https://github.com/CyberAgentAILab/canvas-vae/tree/bc1e2072ba3a253f1b099e8b0c604f6051e787da). The verified archive is 2,989,732,284 bytes with SHA-256 `f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e`; `count.json` lists 18,768 train, 2,315 validation, and 2,278 test documents. The archive and every derived image or embedding remain private.

The original [Crello document specification](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/canvas-vae/canvasvae/data/crello-document-spec.yml#L1-L99) defines context fields `id`, `group`, `format`, `category`, `canvas_width`, `canvas_height`, and `length`; `id` is retained as identity metadata and is not a model input. Sequence fields are `type`, `left`, `top`, `width`, `height`, `color` (three RGB int64 channels), `opacity`, and `image_embedding` (float32[256]). The source parser gives scalar columns shape `[1]` by default, so scalar context fields use `[batch, 1]` and scalar sequence fields use `[batch, length, 1]`; color and image embeddings keep `[batch, length, 3]` and `[batch, length, 256]` ([original DataSpec](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/canvas-vae/canvasvae/data/spec.py#L165-L183)). Geometry uses 64 bins over [0, 1], opacity uses 8 bins over [0, 1], and each color channel uses 16 bins over [0, 255]. String and integer vocabularies retain the source `vocabulary.json` ordering and lookup settings.

`document_id` is `crello-v1/{split}/{source id}`, where `source id` is the original context `id`; split names are `train`, `val`, and `test`. `image_id` is `crello-v1/{split}/{sha256(image_bytes)}` over the exact PNG bytes. This matches the original image transform's byte-based `Distinct` identity within each split ([image transform](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/preprocess/preprocess/transforms/generate_crello_image.py#L22-L60)). The PixelVAE training population is exactly the image transform's `imageElement`, `maskElement`, and `svgElement` subset. The document transform preserves each source document ID and replaces every element's ordered `image_bytes` sequence with one embedding, with no element-type filter ([document transform](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/preprocess/preprocess/transforms/generate_crello_document.py#L24-L49)). Therefore the encoder input and posterior-mean fixture population is every distinct PNG referenced by any element type in the document stage; this is a superset of the PixelVAE training population. Document order is split order `train`, `val`, `test`, then UTF-8 bytewise `document_id`; element order stays as stored in the SequenceExample. Traverse documents in that order and assign fixture rows to the first occurrence of each split-scoped `image_id`; record traversal order in the manifest and pass images to PixelVAE in the same order. This canonical ordering is required because Beam permits arbitrary processing order and its Direct Runner requires the data to fit in memory ([Beam Direct Runner](https://beam.apache.org/documentation/runners/direct/#using-the-direct-runner)).

For a batch padded to its longest document, `element_mask[b, t]` is true exactly when `t < length[b]`. Sequence-field masks for `type`, geometry, and `opacity` equal `element_mask`; `color_mask` additionally requires type `textElement` or `coloredBackground`; `image_embedding_mask` additionally requires type `svgElement`, `imageElement`, or `maskElement`, as declared by the source spec. Padding has false masks. Non-applicable values remain present in the parsed source record; masks identify where each field contributes a loss.

The canonical private fixture is an `.npy` array of contiguous little-endian float32 values with shape `[N, 256]`, one posterior-mean row per unique split-scoped `image_id`. A neighboring `manifest.json` records the format version, source archive SHA-256, frozen TensorFlow encoder-state SHA-256, ordered `image_id` and row pairs, per-row embedding SHA-256 values, and the `.npy` file SHA-256; `manifest.sha256` hashes the manifest itself. The values come from PR 1's deterministic untrained TensorFlow state; the fixture stores no source images. The original document transform decodes PNGs as RGBA before encoding ([document transform](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/preprocess/preprocess/transforms/generate_crello_document.py#L40-L49)). The v1 guide describes each element preview as 256×256 pixels; decode it without resizing, cropping, or compositing and reject any decoded shape other than 256×256×4 ([dataset schema](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/docs/crello-dataset.md#L74-L90)). Each manifest row is sent to PixelVAE in manifest order and output row `i` is paired with manifest entry `i`; the document processor looks up the split-scoped ID and restores the original element order.

CanvasVAE has no released weights, so agreement is checked on the training path. The stages S0-S4 are defined in the [training reproduction protocol](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md); [TRAINING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/canvas-vae/TRAINING.md) records their results.

The `vendor` extra installs CUDA-enabled TensorFlow 2.15.1 and Apache Beam for the original-code workflow. The separate `convert` extra installs `tensorflow-cpu==2.15.1`, pinned Apache Beam, PyYAML, and the shared parity helpers for the Crello CPU data/reference commands below and original TensorFlow checkpoint conversion. It installs no NVIDIA CUDA wheels, so PyTorch keeps the NCCL library that matches its own CUDA wheel when the Crello CPU job imports both frameworks. A Lightning `last.ckpt` converts without a TensorFlow extra. Current training configs keep validation-best weights in `best.ckpt`; a separate unmonitored callback with `save_top_k: 1` retains one rolling checkpoint under Lightning’s epoch/step name and updates `last.ckpt` at configured checkpoint events. The S5 executor verifies the configured final epoch and global step before copying or uploading that file. Use `vendor` and `convert` in separate environments. Original GPU training and evaluation set `NVIDIA_TF32_OVERRIDE=0` and `TF_ENABLE_ONEDNN_OPTS=0`; the currently verified parity setup uses a CPU-only runtime with `CUDA_VISIBLE_DEVICES=""`.

The pinned [Crello document transform](https://github.com/CyberAgentAILab/canvas-vae/blob/bc1e2072ba3a253f1b099e8b0c604f6051e787da/src/preprocess/preprocess/transforms/generate_crello_document.py#L40-L49) calls `tf.debugging.assert_all_finite(embeddings)` without a message. In [TensorFlow 2.15.1](https://github.com/tensorflow/tensorflow/blob/v2.15.1/tensorflow/python/ops/numerics.py#L46-L72), `verify_tensor_all_finite_v2(x, message, name=None)` requires `message`; the unmodified vendor call therefore raises `TypeError` before reaching `check_numerics`. `run_crello_reference.py` installs `traingen_parity.tensorflow_compat.install_assert_all_finite_compat`, which supplies a default message while retaining TensorFlow's finite-value assertion; non-finite embeddings still raise `tf.errors.InvalidArgumentError`. The vendor commit remains unchanged.

## Reproduce Crello data checks

Run these checks on a high-RAM CPU Colab because the original Beam Direct Runner holds its working data in process memory. The scripts use the local archive and private output directories; the embedding fixture includes derived data and must stay private. The dimension audit reports every non-256×256 image with its element types and stops before declaring the contract verified.

```bash
uv sync --package canvas-vae --extra convert
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/download_crello.py --archive .cache/canvas-vae/crello/crello-dataset-v1.zip --extract-to .cache/canvas-vae/crello/source
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/prepare_crello.py --source-dir .cache/canvas-vae/crello/source --output-dir .cache/canvas-vae/crello/package-run-1
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/prepare_crello.py --source-dir .cache/canvas-vae/crello/source --output-dir .cache/canvas-vae/crello/package-run-2
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/compare_crello_cache_runs.py --first-run .cache/canvas-vae/crello/package-run-1 --second-run .cache/canvas-vae/crello/package-run-2
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/run_crello_reference.py --vendor-root vendor/canvas-vae --source-dir .cache/canvas-vae/crello/source --encoder-path "$ENCODER_STATE_PATH" --output-dir .cache/canvas-vae/crello/reference
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/compare_crello_beam_runs.py --first-run .cache/canvas-vae/crello/reference/run-1 --second-run .cache/canvas-vae/crello/reference/run-2
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/write_crello_fixture.py --reference-run .cache/canvas-vae/crello/reference/run-1 --package-dir .cache/canvas-vae/crello/package-run-1 --fixture-dir .cache/canvas-vae/crello/fixture --encoder-state-sha256 "$ENCODER_STATE_SHA256"
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/compare_crello_s4.py --vendor-root vendor/canvas-vae --reference-run .cache/canvas-vae/crello/reference/run-1 --package-dir .cache/canvas-vae/crello/package-run-1 --fixture-dir .cache/canvas-vae/crello/fixture
```

Set `ENCODER_STATE_PATH` and `ENCODER_STATE_SHA256` to the private frozen untrained TensorFlow state from PR 1 and its SHA-256. The reference command runs both original transforms twice with Beam 2.76's in-process `FnApiRunner`; its comparison checks exact serialized document fields and filtered image IDs under stable IDs. The package replay comparison hashes every prepared file, including every cached PNG. The S4 command compares original and package values for all three splits, all context and sequence fields, all masks, canonical IDs and order, fixture embeddings, and repeated loader output. Keep the source archive, package runs, references, and fixture under private storage; do not commit or publish them.

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

An original TensorFlow checkpoint prefix such as `initial.ckpt` or `final.ckpt` needs the `convert` extra. A Lightning `last.ckpt` file uses PyTorch loading and needs no TensorFlow extra. For a full run, use only a `last.ckpt` that passes the executor's final-epoch and global-step check.

```bash
uv run --package canvas-vae --extra convert models/canvas-vae/scripts/convert_original_checkpoint.py --checkpoint .cache/canvas-vae/reference/trace/initial/initial.ckpt --vocabulary .cache/canvas-vae/original/data/rico/vocabulary.json --output-dir .cache/canvas-vae/converted/initial
uv run --package canvas-vae models/canvas-vae/scripts/smoke_from_pretrained.py --checkpoint .cache/canvas-vae/converted/initial
```

For a Lightning checkpoint from `traingen fit`, run the same converter with `uv run --package canvas-vae` and no TensorFlow extra.
